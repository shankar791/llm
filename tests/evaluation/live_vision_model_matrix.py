"""
Step 12D Live Vision Model Capability Matrix Probe for SatQuery AI.
Sequentially benchmarks OpenRouter vision candidates:
1. google/gemma-4-26b-a4b-it:free
2. google/gemma-4-31b-it:free
3. nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free

Distinguishes:
- UPSTREAM_PROVIDER_429: Record RATE_LIMITED, continue to next candidate model.
- ACCOUNT_RATE_LIMIT ("free-models-per-day"): Record ACCOUNT_RATE_LIMIT, halt all remaining tests.

Tasks tested per model:
A. T1 VQA -> (if PASS) -> B. T2 Caption -> (if PASS) -> C. T3 Grounding

Uses real repository satellite images:
- backend/real_data/opt_0611.png
- backend/real_data/opt_0810.png

Gated with: VISION_INTEGRATION_TEST=true
"""
from __future__ import annotations
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai.vision.openrouter_qwen import _encode_image_to_data_url, _is_account_level_rate_limit, _parse_and_validate_grounding

CANDIDATE_MODELS = [
    ("Gemma 4 26B", "google/gemma-4-26b-a4b-it:free"),
    ("Gemma 4 31B", "google/gemma-4-31b-it:free"),
    ("Nemotron 3 Nano Omni", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
]

VQA_PROMPT = "What objects and major land-cover types are visible in this satellite image?"
CAPTION_PROMPT = "Describe this satellite scene concisely, mentioning major land-cover types and visible infrastructure."
GROUND_PROMPT = "Locate the major buildings visible in this satellite image."

SYSTEM_PROMPT_VQA = "You are an expert remote sensing vision assistant. Answer concisely and factually."
SYSTEM_PROMPT_CAPTION = "You are an expert remote sensing analyst. Describe major land-cover types and visible infrastructure."
SYSTEM_PROMPT_GROUND = (
    "You are a remote-sensing grounding model. Locate the requested objects and respond ONLY with JSON:\n"
    '{"objects": [{"label": "building", "box": [x0, y0, x1, y1]}]}.\n'
    "Coordinates must be normalized in [0.0, 1.0] where (x0, y0) is top-left and (x1, y1) is bottom-right."
)


def _extract_error_metadata(error_str: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract error_type and provider_name from OpenRouter error JSON payload."""
    if not error_str:
        return None, None
    try:
        data = json.loads(error_str)
        err = data.get("error", {})
        metadata = err.get("metadata", {})
        provider_name = metadata.get("provider_name")
        error_type = metadata.get("provider_error_code") or err.get("code") or "error"
        return str(error_type), provider_name
    except Exception:
        return "upstream_error", None


def _execute_http(payload: dict, api_key: str, timeout: float = 45.0) -> tuple[int, dict, float, str]:
    """Execute single HTTP POST to OpenRouter API (max 1 request per task per model)."""
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/satquery-ai/satquery",
        "X-Title": "SatQuery AI",
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            lat = (time.perf_counter() - t0) * 1000.0
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body, lat, ""
    except urllib.error.HTTPError as e:
        lat = (time.perf_counter() - t0) * 1000.0
        err_body = e.read().decode("utf-8", errors="replace")
        return e.code, {}, lat, err_body
    except Exception as e:
        lat = (time.perf_counter() - t0) * 1000.0
        return 0, {}, lat, str(e)


def run_model_probe(
    model_slug: str,
    api_key: str,
    data_url1: str,
    data_url2: str,
    w2: int,
    h2: int,
) -> Dict[str, Any]:
    """
    Run sequential probes on a model:
    1. VQA
    2. If VQA PASS -> Caption
    3. If Caption PASS -> Grounding
    If upstream 429 occurs on any task -> record RATE_LIMITED, do not classify as model failure, return to continue next model.
    If account-level limit occurs -> record ACCOUNT_RATE_LIMIT and set account_quota_exhausted=True.
    """
    results: Dict[str, Any] = {
        "model": model_slug,
        "account_quota_exhausted": False,
        "vqa": {"status": "NOT_RUN", "latency_ms": 0.0, "error_type": None, "provider_name": None, "fallback_used": False},
        "caption": {"status": "NOT_RUN", "latency_ms": 0.0, "error_type": None, "provider_name": None, "fallback_used": False},
        "grounding": {"status": "NOT_RUN", "latency_ms": 0.0, "error_type": None, "provider_name": None, "fallback_used": False},
        "overall_status": "ERROR",
        "avg_latency_ms": 0.0,
    }

    latencies: List[float] = []

    # ----------------------------------------------------
    # 1. T1 VQA Probe
    # ----------------------------------------------------
    vqa_payload = {
        "model": model_slug,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_VQA},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VQA_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url1}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "reasoning": {"effort": "low"},
    }

    status, resp_data, lat, err = _execute_http(vqa_payload, api_key)
    latencies.append(lat)
    results["vqa"]["latency_ms"] = round(lat, 1)

    if status == 200:
        choice = resp_data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "").strip()
        finish_reason = choice.get("finish_reason")
        results["vqa"]["status"] = "PASS" if content else "ERROR"
        results["vqa"]["content"] = content
        results["vqa"]["finish_reason"] = finish_reason
    elif status == 429:
        err_type, prov_name = _extract_error_metadata(err)
        results["vqa"]["error_type"] = err_type
        results["vqa"]["provider_name"] = prov_name

        if _is_account_level_rate_limit(err):
            results["vqa"]["status"] = "ACCOUNT_RATE_LIMIT"
            results["account_quota_exhausted"] = True
            results["overall_status"] = "ACCOUNT_RATE_LIMIT"
            return results
        else:
            # Upstream provider rate limit: model is rate limited, but not a failure
            results["vqa"]["status"] = "RATE_LIMITED"
            results["overall_status"] = "RATE_LIMITED"
            return results
    else:
        err_type, prov_name = _extract_error_metadata(err)
        results["vqa"]["status"] = "ERROR"
        results["vqa"]["error_type"] = err_type
        results["vqa"]["provider_name"] = prov_name
        results["overall_status"] = "ERROR"
        return results

    time.sleep(1.0)

    # ----------------------------------------------------
    # 2. T2 Caption Probe (Only if VQA PASS)
    # ----------------------------------------------------
    cap_payload = {
        "model": model_slug,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_CAPTION},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url1}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 768,
        "reasoning": {"effort": "low"},
    }

    status, resp_data, lat, err = _execute_http(cap_payload, api_key)
    latencies.append(lat)
    results["caption"]["latency_ms"] = round(lat, 1)

    if status == 200:
        choice = resp_data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "").strip()
        finish_reason = choice.get("finish_reason")
        results["caption"]["status"] = "PASS" if content else "ERROR"
        results["caption"]["content"] = content
        results["caption"]["finish_reason"] = finish_reason
    elif status == 429:
        err_type, prov_name = _extract_error_metadata(err)
        results["caption"]["error_type"] = err_type
        results["caption"]["provider_name"] = prov_name

        if _is_account_level_rate_limit(err):
            results["caption"]["status"] = "ACCOUNT_RATE_LIMIT"
            results["account_quota_exhausted"] = True
            results["overall_status"] = "ACCOUNT_RATE_LIMIT"
            return results
        else:
            results["caption"]["status"] = "RATE_LIMITED"
            results["overall_status"] = "RATE_LIMITED"
            return results
    else:
        err_type, prov_name = _extract_error_metadata(err)
        results["caption"]["status"] = "ERROR"
        results["caption"]["error_type"] = err_type
        results["caption"]["provider_name"] = prov_name
        results["overall_status"] = "ERROR"
        return results

    time.sleep(1.0)

    # ----------------------------------------------------
    # 3. T3 Grounding Probe (Only if Caption PASS)
    # ----------------------------------------------------
    ground_payload = {
        "model": model_slug,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_GROUND},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GROUND_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url2}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }

    status, resp_data, lat, err = _execute_http(ground_payload, api_key)
    latencies.append(lat)
    results["grounding"]["latency_ms"] = round(lat, 1)

    if status == 200:
        content = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        results["grounding"]["content"] = content

        # Strictly validate coordinates; never fabricate
        parsed_res = _parse_and_validate_grounding(content, w2, h2)
        if parsed_res and len(parsed_res.objects) > 0:
            results["grounding"]["status"] = "PASS"
            results["grounding"]["box_count"] = len(parsed_res.objects)
            results["grounding"]["boxes"] = [
                {"label": b.label, "box": b.box, "pixel_box": b.to_pixel_box(w2, h2)}
                for b in parsed_res.objects
            ]
            results["grounding"]["coordinate_validity"] = "VALID_NORMALIZED"
            results["grounding"]["semantically_plausible"] = True
        else:
            # Model emitted text or unsupported schema
            results["grounding"]["status"] = "UNSUPPORTED"
            results["grounding"]["box_count"] = 0
            results["grounding"]["coordinate_validity"] = "NO_VALID_BOXES"
            results["grounding"]["semantically_plausible"] = False
    elif status == 429:
        err_type, prov_name = _extract_error_metadata(err)
        results["grounding"]["error_type"] = err_type
        results["grounding"]["provider_name"] = prov_name

        if _is_account_level_rate_limit(err):
            results["grounding"]["status"] = "ACCOUNT_RATE_LIMIT"
            results["account_quota_exhausted"] = True
            results["overall_status"] = "ACCOUNT_RATE_LIMIT"
            return results
        else:
            results["grounding"]["status"] = "RATE_LIMITED"
            results["overall_status"] = "RATE_LIMITED"
            return results
    else:
        err_type, prov_name = _extract_error_metadata(err)
        results["grounding"]["status"] = "ERROR"
        results["grounding"]["error_type"] = err_type
        results["grounding"]["provider_name"] = prov_name

    if latencies:
        results["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)

    # Determine overall status
    v_s = results["vqa"]["status"]
    c_s = results["caption"]["status"]
    g_s = results["grounding"]["status"]

    if v_s == "PASS" and c_s == "PASS" and g_s == "PASS":
        results["overall_status"] = "PASS"
    elif v_s == "PASS" and c_s == "PASS" and g_s == "UNSUPPORTED":
        results["overall_status"] = "PARTIAL"
    elif any(s == "ACCOUNT_RATE_LIMIT" for s in [v_s, c_s, g_s]):
        results["overall_status"] = "ACCOUNT_RATE_LIMIT"
    elif any(s == "RATE_LIMITED" for s in [v_s, c_s, g_s]):
        results["overall_status"] = "RATE_LIMITED"
    else:
        results["overall_status"] = "ERROR"

    return results


def run_live_vision_matrix() -> Dict[str, Any]:
    """Execute live vision matrix across all candidate models."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("VISION_API_KEY")
    if not api_key:
        print("[SKIPPED] OPENROUTER_API_KEY is not set. Live vision matrix requires API key.")
        return {"status": "SKIPPED_NO_API_KEY"}

    is_gated = os.environ.get("VISION_INTEGRATION_TEST", "").lower() in {"true", "1", "yes"}
    if not is_gated and not sys.stdout.isatty():
        print("[SKIPPED] Set VISION_INTEGRATION_TEST=true to run live vision matrix.")
        return {"status": "SKIPPED_NOT_GATED"}

    opt1_path = "backend/real_data/opt_0611.png"
    opt2_path = "backend/real_data/opt_0810.png"

    if not os.path.exists(opt1_path) or not os.path.exists(opt2_path):
        print(f"[ERROR] Required satellite test images missing: {opt1_path}, {opt2_path}")
        return {"status": "MISSING_IMAGES"}

    img1 = Image.open(opt1_path)
    img2 = Image.open(opt2_path)
    data_url1, (w1, h1) = _encode_image_to_data_url(img1)
    data_url2, (w2, h2) = _encode_image_to_data_url(img2)

    matrix_results: Dict[str, Any] = {"models": {}}
    table_rows: List[Dict[str, str]] = []

    print("\n==================================================================")
    print("STEP 12D — LIVE OPENROUTER VISION MODEL CAPABILITY MATRIX PROBES")
    print("==================================================================\n")

    for display_name, model_slug in CANDIDATE_MODELS:
        print(f"\n--- Probing Candidate: {display_name} ({model_slug}) ---")
        probe_res = run_model_probe(model_slug, api_key, data_url1, data_url2, w2, h2)
        matrix_results["models"][model_slug] = probe_res

        vqa_s = probe_res["vqa"]["status"]
        cap_s = probe_res["caption"]["status"]
        grd_s = probe_res["grounding"]["status"]
        overall = probe_res["overall_status"]

        print(f"  VQA:       {vqa_s} ({probe_res['vqa']['latency_ms']}ms)")
        print(f"  Caption:   {cap_s} ({probe_res['caption']['latency_ms']}ms)")
        print(f"  Grounding: {grd_s} ({probe_res['grounding']['latency_ms']}ms) - Boxes: {probe_res['grounding'].get('box_count', 0)}")
        print(f"  Status:    {overall}")

        table_rows.append({
            "name": display_name,
            "vqa": vqa_s,
            "caption": cap_s,
            "grounding": grd_s,
            "status": overall,
        })

        # Check for account-level quota exhaustion
        if probe_res.get("account_quota_exhausted"):
            print("\n[ALERT] Account-level daily free quota ('free-models-per-day') exhausted. Halting remaining live probes.")
            break
        elif overall == "RATE_LIMITED":
            print(f"[INFO] Upstream 429 encountered for {display_name}. Model is rate-limited on upstream pool. Continuing to next candidate...")

        time.sleep(1.5)

    # Print clean Markdown matrix table
    print("\n\n### Step 12D Live Model Capability Matrix\n")
    print("| Model | VQA | Caption | Grounding | Status |")
    print("|---|---|---|---|---|")
    for r in table_rows:
        print(f"| {r['name']} | {r['vqa']} | {r['caption']} | {r['grounding']} | {r['status']} |")

    # Save structured results to JSON (strictly excluding API keys)
    out_path = "tests/evaluation/live_vision_model_matrix_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(matrix_results, f, indent=2)
    print(f"\n[Matrix results saved to {out_path}]")

    return matrix_results


if __name__ == "__main__":
    run_live_vision_matrix()
