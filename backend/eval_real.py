"""
Evaluate SatQuery AI on REAL satellite data over Vignan University (Deshmukhi,
Pochampally, Telangana ~17.36N 78.62E):
  - Sentinel-2 L2A true colour, 2026-06-11 (3% cloud)
  - Sentinel-2 L2A true colour, 2026-08-10 (17% cloud, cloudy SE quadrant)
  - Sentinel-1 GRD VV, 2026-08-10 (same-day SAR)
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rasterio_utils import RasterInput
from agent import execute

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_data")
OUT = os.path.join(DATA, "results")
os.makedirs(OUT, exist_ok=True)

def load(name):
    with open(os.path.join(DATA, name), "rb") as f:
        return RasterInput(name, f.read())

opt_jun = load("opt_0611.png")
opt_aug = load("opt_0810.png")
sar_aug = load("sar_0810.png")

print(f"opt_0611: {opt_jun.metadata}")
print(f"opt_0810: {opt_aug.metadata}")
print(f"sar_0810: {sar_aug.metadata}")

CASES = [
    ("1_single_vqa", "Describe the land-cover and major objects visible in this image.",
     [opt_jun]),
    ("2_grounding_water", "Highlight the water body in this image.", [opt_jun]),
    ("3_bitemporal_change", "What changed between these two dates, and where did the change occur?",
     [opt_jun, opt_aug]),
    ("4_optical_sar_fusion", "Use the optical and SAR images together to identify built-up and water-covered regions.",
     [opt_aug, sar_aug]),
    ("5_cloud_penetration", "Part of the optical image is covered by clouds. Use the SAR image to describe what lies beneath the clouds.",
     [opt_aug, sar_aug]),
]

report = []
for name, query, rasters in CASES:
    t0 = time.time()
    res = execute(query, rasters)
    dt = (time.time() - t0) * 1000
    print("\n" + "=" * 70)
    print(f"[{name}]")
    print(f"  scenario : {res['scenario']}")
    print("  tools    :", [o["tool"] for o in res["outputs"]])
    print(f"  conf     : {res.get('confidence')} | {dt:.0f} ms")
    print(f"  answer   : {res['answer'][:400]}")
    # save evidence images
    for i, ev in enumerate(res.get("evidence_images_b64", [])):
        p = os.path.join(OUT, f"{name}_ev{i}.png")
        if ev.startswith("data:image/png;base64,"):
            import base64
            with open(p, "wb") as f:
                f.write(base64.b64decode(ev.split(",", 1)[1]))
    report.append({"case": name, "query": query, "scenario": res["scenario"],
                   "tools": [o["tool"] for o in res["outputs"]], "confidence": res.get("confidence"),
                   "ms": round(dt), "answer": res["answer"],
                   "trace": res.get("trace")})

with open(os.path.join(OUT, "report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("\n\nREPORT SAVED:", os.path.join(OUT, "report.json"))
