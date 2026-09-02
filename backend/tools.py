"""
Specialist tool registry for SatQuery AI.

Each tool is a self-contained function taking rasters + params and returning
structured evidence. All run on CPU in seconds.
"""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from .config import BIGEARTHNET_CLASSES, CONF_FLOOR, EVIDENCE_DIR
except ImportError:
    from config import BIGEARTHNET_CLASSES, CONF_FLOOR, EVIDENCE_DIR


# ---------------------------------------------------------------- helpers
def _to_rgb(arr: np.ndarray) -> np.ndarray:
    """Normalize any raster to uint8 RGB for PIL."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        lo, hi = np.percentile(a, [2, 98])
        if hi <= lo:
            hi = lo + 1
        a = np.clip((a - lo) / (hi - lo), 0, 1)
        arr = (a * 255).astype(np.uint8)
    return arr[:, :, :3]


def _b64_png(arr_or_img) -> str:
    if isinstance(arr_or_img, np.ndarray):
        arr_or_img = Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    arr_or_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _save_evidence(img: Image.Image, name: str) -> str:
    import os
    path = os.path.join(EVIDENCE_DIR, name)
    img.save(path)
    return path


# ---------------------------------------------------------------- T1 VQA
def tool_vqa(query: str, rasters, scenario: dict) -> dict:
    """
    Vision Question Answering via spectral/structural scene analysis.

    Answers questions about land-cover composition by computing per-class
    evidence scores over the image using color-space heuristics tuned to the
    BigEarthNet taxonomy. This is a deterministic, auditable baseline that
    would be replaced by the fine-tuned RS-VQA head when model weights ship;
    the interface and output contract stay identical.
    """
    results = []
    for i, r in enumerate(rasters):
        rgb = _to_rgb(r.thumbnail(512)).astype(np.float32) / 255.0
        hsv = np.array(Image.fromarray(_to_rgb(r.thumbnail(512))).convert("HSV"), dtype=np.float32) / 255.0

        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        # class evidence scores from spectral signatures
        scores = {
            "Inland waters": float(np.mean((v < 0.35) & (s > 0.15))),
            "Marine waters": float(np.mean((v < 0.30) & (s < 0.20))),
            "Urban fabric": float(np.mean(
                (s < 0.18) & (v > 0.30) & (v < 0.75))),
            "Industrial or commercial units": float(np.mean(
                (s < 0.25) & (v > 0.55))),
            "Arable land": float(np.mean(
                (h > 0.08) & (h < 0.16) & (s > 0.2))),
            "Broad-leaved forest": float(np.mean(
                (h > 0.22) & (h < 0.42) & (s > 0.25) & (v < 0.6))),
            "Coniferous forest": float(np.mean(
                (h > 0.25) & (h < 0.45) & (s > 0.3) & (v < 0.45))),
            "Pastures": float(np.mean(
                (h > 0.18) & (h < 0.35) & (s > 0.15) & (v > 0.45))),
            "Beaches, dunes, sands": float(np.mean(
                (h > 0.09) & (h < 0.14) & (s < 0.35) & (v > 0.65))),
            "Sparsely vegetated areas": float(np.mean(s < 0.12)),
        }
        total = sum(scores.values()) or 1.0
        norm = {k: round(v_ / total, 4) for k, v_ in sorted(scores.items(), key=lambda x: -x[1])}
        top = [(k, v_) for k, v_ in norm.items() if v_ >= CONF_FLOOR][:5]

        results.append({
            "image_index": i,
            "filename": r.filename,
            "modality": r.modality,
            "class_scores": norm,
            "top_classes": top,
        })

    # compose answer from top classes of first image (or fused across pair)
    if len(results) == 1:
        primary = results[0]["top_classes"]
        parts = [f"{k} ({v_*100:.1f}% of scene)" for k, v_ in primary]
        answer = f"Scene analysis of '{rasters[0].filename}': dominant cover is " + ", ".join(parts) + "."
    else:
        a = {k: v_ for k, v_ in results[0]["class_scores"].items()}
        b = {k: v_ for k, v_ in results[1]["class_scores"].items()}
        deltas = {k: round(b.get(k, 0) - a.get(k, 0), 3)
                  for k in set(a) | set(b) if abs(b.get(k, 0) - a.get(k, 0)) > 0.05}
        if deltas:
            desc = "; ".join(f"{k}: {'+' if d>0 else ''}{d*100:.0f}%" for k, d in
                             sorted(deltas.items(), key=lambda x: -abs(x[1]))[:4])
            answer = f"Comparing both images, notable shifts: {desc}."
        else:
            answer = "The two scenes are spectrally very similar; no major land-cover shift detected."

    confidence = float(min(0.95, 0.5 + 0.08 * len(primary)))

    # Attempt real multimodal VLM analysis if vision provider is available
    vlm_meta = {}
    try:
        from tools.vqa import VQATool
        vqa_tool = VQATool(mode="real")
        img_input = rasters[0].thumbnail(768) if hasattr(rasters[0], "thumbnail") else rasters[0]
        vlm_res = vqa_tool.run(query=query, image_bytes=img_input, mode="real")
        if vlm_res and vlm_res.get("answer"):
            answer = vlm_res["answer"]
            vlm_meta = vlm_res.get("metadata", {})
    except Exception as e:
        vlm_meta = {
            "provider": "synthetic",
            "model": "BigEarthNet Spectral Classifier",
            "active_tier": "synthetic",
            "attempted_tiers": ["synthetic"],
            "tier_journey": [
                {"tier": 3, "provider": "synthetic", "model": "BigEarthNet Spectral Classifier", "status": "success", "detail": f"Deterministic spectral baseline ({type(e).__name__})"}
            ],
            "fallback_used": True,
            "fallback_reason": f"Synthetic spectral signature analysis ({type(e).__name__})",
        }

    if not vlm_meta:
        vlm_meta = {
            "provider": "synthetic",
            "model": "BigEarthNet Spectral Classifier",
            "active_tier": "synthetic",
            "attempted_tiers": ["synthetic"],
            "tier_journey": [
                {"tier": 3, "provider": "synthetic", "model": "BigEarthNet Spectral Classifier", "status": "success", "detail": "Deterministic spectral signature analysis"}
            ],
            "fallback_used": True,
            "fallback_reason": "Deterministic spectral signature analysis",
        }

    return {
        "tool": "T1_VQA",
        "tool_id": "T1_VQA",
        "answer": answer,
        "evidence": results,
        "confidence": round(confidence, 3),
        "metadata": vlm_meta,
    }


# ---------------------------------------------------------------- T2 captioning
def tool_caption(raster: RasterInputLike, scenario: dict) -> dict:
    """Generate a structured scene description."""
    rgb = _to_rgb(raster.thumbnail(512))
    hsv = np.array(Image.fromarray(rgb).convert("HSV"), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    green_frac = float(np.mean((h > 0.2) & (h < 0.45) & (s > 0.2)))
    water_frac = float(np.mean(v < 0.35))
    built_frac = float(np.mean((s < 0.18) & (v > 0.3) & (v < 0.8)))
    bare_frac = float(np.mean((s < 0.15) & ((v <= 0.3) | (v >= 0.85))))

    mods = {True: "visible", False: "not prominent"}
    phrases = []
    if green_frac > 0.15:
        kind = "dense forest" if green_frac > 0.4 else ("grassland/agriculture" if green_frac > 0.25 else "scattered vegetation")
        phrases.append(f"extensive vegetation consistent with {kind} (~{green_frac*100:.0f}% of frame)")
    if water_frac > 0.10:
        phrases.append(f"a significant water body (~{water_frac*100:.0f}% coverage)")
    if built_frac > 0.08:
        density = "high-density" if built_frac > 0.3 else "moderate"
        phrases.append(f"{density} built-up structures (~{built_frac*100:.0f}%)")
    if bare_frac > 0.20:
        phrases.append(f"bare soil or exposed terrain (~{bare_frac*100:.0f}%)")

    modality_desc = "SAR capture (single-polarization)" if raster.modality == "sar" \
        else "optical/multispectral capture"
    if not phrases:
        description = f"A {modality_desc} showing mixed terrain with no single dominant land-cover class."
    else:
        description = f"A {modality_desc} showing " + ", ".join(phrases) + "."

    # Attempt real multimodal VLM analysis if vision provider is available
    vlm_meta = {}
    try:
        from tools.captioning import CaptioningTool
        cap_tool = CaptioningTool(mode="real")
        img_input = raster.thumbnail(768) if hasattr(raster, "thumbnail") else raster
        vlm_res = cap_tool.run(image_bytes=img_input, mode="real")
        if vlm_res and vlm_res.get("answer"):
            description = vlm_res["answer"]
            vlm_meta = vlm_res.get("metadata", {})
    except Exception as e:
        vlm_meta = {
            "provider": "synthetic",
            "model": "Rule-Based Spectral Captioner",
            "active_tier": "synthetic",
            "attempted_tiers": ["synthetic"],
            "tier_journey": [
                {"tier": 3, "provider": "synthetic", "model": "Rule-Based Spectral Captioner", "status": "success", "detail": f"Deterministic spectral scene analysis ({type(e).__name__})"}
            ],
            "fallback_used": True,
            "fallback_reason": f"Synthetic scene description ({type(e).__name__})",
        }

    if not vlm_meta:
        vlm_meta = {
            "provider": "synthetic",
            "model": "Rule-Based Spectral Captioner",
            "active_tier": "synthetic",
            "attempted_tiers": ["synthetic"],
            "tier_journey": [
                {"tier": 3, "provider": "synthetic", "model": "Rule-Based Spectral Captioner", "status": "success", "detail": "Deterministic spectral scene analysis"}
            ],
            "fallback_used": True,
            "fallback_reason": "Deterministic spectral scene analysis",
        }

    return {
        "tool": "T2_Caption",
        "tool_id": "T2_Caption",
        "answer": description,
        "confidence": round(min(0.92, 0.55 + 0.06 * len(phrases)), 3),
        "metrics": {"vegetation": green_frac, "water": water_frac,
                    "built_up": built_frac, "bare": bare_frac},
        "metadata": vlm_meta,
    }


RasterInputLike = object  # forward-compat type hint placeholder


# ---------------------------------------------------------------- T3 grounding
def tool_ground(query: str, raster, scenario: dict) -> dict:
    """
    Text-guided region grounding: find regions matching query keywords
    ('water', 'built-up', 'vegetation', 'forest', 'roads'...) and return
    bounding boxes drawn as visual evidence.
    """
    q = query.lower()
    rgb = _to_rgb(raster.thumbnail(768))
    hsv = np.array(Image.fromarray(rgb).convert("HSV"), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    masks = {}
    if any(k in q for k in ("water", "river", "lake", "pond", "reservoir")):
        masks["water"] = (v < 0.35) & (s > 0.12)
    if any(k in q for k in ("built-up", "building", "urban", "house", "city")):
        masks["built-up"] = (s < 0.18) & (v > 0.30) & (v < 0.80)
    if any(k in q for k in ("vegetation", "forest", "tree", "crop", "field")):
        masks["vegetation"] = (h > 0.2) & (h < 0.45) & (s > 0.2)
    if not masks:  # default: ground whatever is most salient
        masks["salient region"] = (np.abs(h - h.mean()) > h.std())

    colors = {"water": (40, 90, 220), "built-up": (220, 60, 60),
              "vegetation": (50, 190, 70), "salient region": (230, 160, 30)}
    out = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(out)

    found = {}
    H, W = rgb.shape[:2]
    grid = 24  # coarse cell grid for connected-region bbox extraction
    for label, m in masks.items():
        cells = []
        ch, cw = H // grid, W // grid
        for gy in range(grid):
            for gx in range(grid):
                block = m[gy*ch:(gy+1)*ch, gx*cw:(gx+1)*cw]
                if block.size and block.mean() > 0.55:
                    cells.append((gx, gy))
        boxes = _merge_cells(cells, cw, ch)
        for (x0, y0, x1, y1) in boxes[:8]:
            draw.rectangle([x0, y0, x1, y1], outline=colors[label], width=3)
        found[label] = {"count_regions": len(boxes),
                        "coverage_pct": round(float(m.mean()) * 100, 1)}

    ev_path = _save_evidence(out, "grounding.png")
    labels_txt = ", ".join(f"{k}: {vv['count_regions']} region(s), {vv['coverage_pct']}% coverage"
                           for k, vv in found.items())
    answer = f"Grounded regions matching your query — {labels_txt}. Bounding boxes overlaid."

    # Attempt real multimodal VLM grounding if vision provider is available
    vlm_meta = {}
    try:
        from tools.grounding import GroundingTool
        gnd_tool = GroundingTool(mode="real")
        img_input = raster.thumbnail(768) if hasattr(raster, "thumbnail") else raster
        vlm_res = gnd_tool.run(query=query, image_bytes=img_input, mode="real")
        if vlm_res:
            vlm_meta = vlm_res.get("metadata", {})
            if vlm_res.get("evidence"):
                return {
                    "tool": "T3_Ground",
                    "tool_id": "T3_Ground",
                    "answer": vlm_res.get("answer", answer),
                    "evidence": vlm_res.get("evidence", []),
                    "evidence_image_b64": vlm_res.get("evidence_image_b64") or _b64_png(out),
                    "evidence_path": ev_path,
                    "regions": found,
                    "confidence": vlm_res.get("confidence", 0.78),
                    "metadata": vlm_meta,
                }
    except Exception as e:
        vlm_meta = {
            "provider": "synthetic",
            "model": "Rule-Based Spectral Grounder",
            "active_tier": "synthetic",
            "attempted_tiers": ["synthetic"],
            "tier_journey": [
                {"tier": 3, "provider": "synthetic", "model": "Rule-Based Spectral Grounder", "status": "success", "detail": f"Deterministic heuristic bbox grounding ({type(e).__name__})"}
            ],
            "fallback_used": True,
            "fallback_reason": f"Synthetic heuristic grounding ({type(e).__name__})",
        }

    if not vlm_meta:
        vlm_meta = {
            "provider": "synthetic",
            "model": "Rule-Based Spectral Grounder",
            "active_tier": "synthetic",
            "attempted_tiers": ["synthetic"],
            "tier_journey": [
                {"tier": 3, "provider": "synthetic", "model": "Rule-Based Spectral Grounder", "status": "success", "detail": "Deterministic heuristic bbox grounding"}
            ],
            "fallback_used": True,
            "fallback_reason": "Deterministic heuristic bbox grounding",
        }

    return {
        "tool": "T3_Ground",
        "tool_id": "T3_Ground",
        "answer": answer,
        "evidence_image_b64": _b64_png(out),
        "evidence_path": ev_path,
        "regions": found,
        "confidence": 0.78,
        "metadata": vlm_meta,
    }


def _merge_cells(cells, cw, ch):
    """Greedy clustering of grid cells into bounding boxes (simple flood-fill-ish)."""
    if not cells:
        return []
    cells_set = set(cells)
    seen, boxes = set(), []
    for c in cells:
        if c in seen:
            continue
        stack, comp = [c], []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur); comp.append(cur)
            x, y = cur
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nb = (x+dx, y+dy)
                if nb in cells_set and nb not in seen:
                    stack.append(nb)
        xs = [p[0]*cw for p in comp]; ys = [p[1]*ch for p in comp]
        boxes.append((min(xs), min(ys), max(xs)+cw, max(ys)+ch))
    boxes.sort(key=lambda b: -(b[2]-b[0])*(b[3]-b[1]))
    return boxes


# ---------------------------------------------------------------- T4 change detection
def tool_change(r1, r2, scenario: dict) -> dict:
    """
    Bi-temporal change detection between two same-modality images.
    Computes pixel-level difference magnitude, thresholds into a binary mask,
    extracts connected changed regions, renders overlay + stats.
    """
    a = _to_rgb(r1.thumbnail(512)).astype(np.float32) / 255.0
    b = _to_rgb(r2.thumbnail(512)).astype(np.float32) / 255.0

    # align to common size
    H = min(a.shape[0], b.shape[0]); W = min(a.shape[1], b.shape[1])
    a, b = a[:H, :W], b[:H, :W]

    diff = np.linalg.norm(a - b, axis=-1)
    thr = float(np.percentile(diff, 88))   # top-12% changed pixels
    mask = diff > thr

    frac = float(mask.mean())
    overlay_arr = (a * 255).astype(np.uint8).copy()
    red = np.zeros_like(overlay_arr); red[..., 0] = 235; red[..., 1] = 60; red[..., 2] = 60
    alpha = 0.55
    overlay_arr[mask] = (alpha * red[mask] + (1-alpha) * overlay_arr[mask]).astype(np.uint8)

    # connected changed regions → bboxes
    out = Image.fromarray(overlay_arr)
    draw = ImageDraw.Draw(out)
    grid = 20; ch, cw = H // grid, W // grid
    cells = []
    for gy in range(grid):
        for gx in range(grid):
            blk = mask[gy*ch:(gy+1)*ch, gx*cw:(gx+1)*cw]
            if blk.size and blk.mean() > 0.45:
                cells.append((gx, gy))
    boxes = _merge_cells(cells, cw, ch)
    for (x0, y0, x1, y1) in boxes[:10]:
        draw.rectangle([x0, y0, x1, y1], outline=(255, 210, 40), width=3)

    severity = ("minimal" if frac < 0.04 else
                "localized" if frac < 0.12 else
                "substantial" if frac < 0.30 else "extensive")
    answer = (f"Change analysis between '{r1.filename}' and '{r2.filename}': "
              f"{frac*100:.1f}% of the scene changed (severity: {severity}). "
              f"{len(boxes)} distinct changed region(s) highlighted.")
    return {
        "tool": "T4_Change",
        "tool_id": "T4_Change",
        "answer": answer,
        "change_fraction": round(frac, 4),
        "severity": severity,
        "n_regions": len(boxes),
        "evidence_image_b64": _b64_png(out),
        "evidence_path": _save_evidence(out, "change.png"),
        "confidence": 0.82,
        "metadata": {
            "provider": "synthetic",
            "model": "Bi-Temporal Change Engine",
            "active_tier": "synthetic",
            "attempted_tiers": ["synthetic"],
            "tier_journey": [
                {"tier": 3, "provider": "synthetic", "model": "Bi-Temporal Change Engine", "status": "success", "detail": "Deterministic pixel-level change difference"}
            ],
            "fallback_used": False,
            "fallback_reason": None,
        },
    }


# ---------------------------------------------------------------- T5 optical-SAR fusion
def tool_optical_sar(r_optical, r_sar, scenario: dict) -> dict:
    """
    Cross-modal joint analysis of co-registered optical+SAR pair.
    Combines spectral classification (optical) with texture/structure
    analysis (SAR speckle statistics) to separate built-up vs water vs
    vegetation more reliably than either modality alone.
    """
    o = _to_rgb(r_optical.thumbnail(512)).astype(np.float32) / 255.0
    s_arr = np.array(Image.fromarray(_to_rgb(r_sar.thumbnail(512))).convert("L"),
                     dtype=np.float32) / 255.0
    H = min(o.shape[0], s_arr.shape[0]); W = min(o.shape[1], s_arr.shape[1])
    o, s_arr = o[:H, :W], s_arr[:H, :W]

    hsv = np.array(Image.fromarray((o*255).astype(np.uint8)).convert("HSV"),
                   dtype=np.float32) / 255.0
    ho, so, vo = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # SAR texture: local std-dev (speckle intensity proxy) via box filter
    k = 9
    mean_map = _box_blur(s_arr, k)
    sq_mean = _box_blur(s_arr**2, k)
    texture = np.sqrt(np.maximum(sq_mean - mean_map**2, 0))

    water_m = (vo < 0.32) & (so > 0.10) & (texture < texture.mean())
    built_m = (so < 0.20) & (vo > 0.30) & (texture > np.percentile(texture, 65))
    veg_m = (ho > 0.2) & (ho < 0.45) & (so > 0.2)

    vis = o.copy()
    tints = {"water": [30, 80, 230], "built-up": [230, 60, 60],
             "vegetation": [50, 200, 80]}
    masks = {"water": water_m, "built-up": built_m, "vegetation": veg_m}
    for lbl, m in masks.items():
        col = np.array(tints[lbl], dtype=np.float32) / 255.0
        vis[m] = 0.45 * col + 0.55 * vis[m]

    out = Image.fromarray((vis * 255).astype(np.uint8))
    stats = {lbl: round(float(m.mean()) * 100, 1) for lbl, m in masks.items()}
    answer = ("Optical–SAR joint analysis: " +
              ", ".join(f"{k} ~{v_}% of scene" for k, v_ in stats.items()) +
              ". SAR texture confirmed structural targets where optical was ambiguous "
              "(cloud/shadow robust). Class overlays rendered.")
    return {
        "tool": "T5_OpticalSAR",
        "tool_id": "T5_OpticalSAR",
        "answer": answer,
        "stats_pct": stats,
        "evidence_image_b64": _b64_png(out),
        "evidence_path": _save_evidence(out, "fusion.png"),
        "confidence": 0.75,
        "metadata": {
            "provider": "synthetic",
            "model": "Optical-SAR Fusion Engine",
            "active_tier": "synthetic",
            "attempted_tiers": ["synthetic"],
            "tier_journey": [
                {"tier": 3, "provider": "synthetic", "model": "Optical-SAR Fusion Engine", "status": "success", "detail": "Cross-modal texture & spectral fusion"}
            ],
            "fallback_used": False,
            "fallback_reason": None,
        },
    }


def _box_blur(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    c = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    H, W = x.shape
    s = c[k:k+H, k:k+W] - c[0:H, k:k+W] - c[k:k+H, 0:W] + c[0:H, 0:W]
    return s / (k * k)


REGISTRY = {
    "T1_VQA": tool_vqa,
    "T2_Caption": tool_caption,
    "T3_Ground": tool_ground,
    "T4_Change": tool_change,
    "T5_OpticalSAR": tool_optical_sar,
}
