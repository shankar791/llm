"""
STEP 12 Live Smoke Test for OpenRouter Free Models:
- NLP: z-ai/glm-5.2:free
- Vision: google/gemma-4-26b-a4b-it:free
Uses real repository satellite images:
- backend/real_data/opt_0611.png
- backend/real_data/opt_0810.png
"""
from __future__ import annotations
import json
import os
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from PIL import Image
from ai.vision.openrouter_qwen import _encode_image_to_data_url


def run_live_smoke_test():
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY") or os.environ.get("VISION_API_KEY")
    if not api_key:
        print("[ERROR] No API key found. Please export OPENROUTER_API_KEY or LLM_API_KEY to execute real API requests.")
        return {"status": "NO_API_KEY"}

    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/satquery-ai/satquery",
        "X-Title": "SatQuery AI",
    }

    results = {}

    def _execute_http(payload: dict) -> tuple[int, dict, float, str]:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
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

    # ============================================================
    # 1. NLP LIVE TEST — z-ai/glm-5.2:free
    # ============================================================
    nlp_model = "z-ai/glm-5.2:free"
    print(f"\n[1] Testing NLP Model: {nlp_model} on Intent Classification...")

    intent_payload = {
        "model": nlp_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a JSON classifier for SatQuery AI. Classify the user query into a JSON object with fields: "
                    "'task' ('vqa' | 'caption' | 'ground' | 'change' | 'fusion'), 'target', 'requires_temporal_pair' (boolean)."
                ),
            },
            {
                "role": "user",
                "content": "Identify new construction between 2020 and 2024.",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }

    status, resp_data, lat, err = _execute_http(intent_payload)
    results["nlp_intent"] = {
        "model": nlp_model,
        "status": status,
        "latency_ms": round(lat, 2),
        "raw_response": resp_data,
        "error": err,
    }
    if status == 200:
        content = resp_data["choices"][0]["message"]["content"].strip()
        print(f" -> Intent Response ({lat:.1f}ms): {content}")
        results["nlp_intent"]["content"] = content
    else:
        print(f" -> Intent Failed: HTTP {status}, error: {err}")

    time.sleep(1.0)

    # Synthesis Test
    print(f"\n[2] Testing NLP Model: {nlp_model} on Synthesis...")
    synth_payload = {
        "model": nlp_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the final response synthesizer for SatQuery AI. "
                    "Synthesize a grounded answer using ONLY the supplied facts: "
                    "area_ha = 12.4, polygon_count = 3, change_fraction_pct = 7.0. "
                    "Return a JSON object with 'answer' preserving these exact numbers."
                ),
            },
            {
                "role": "user",
                "content": "Summarize the detected construction change.",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    status, resp_data, lat, err = _execute_http(synth_payload)
    results["nlp_synthesis"] = {
        "model": nlp_model,
        "status": status,
        "latency_ms": round(lat, 2),
        "raw_response": resp_data,
        "error": err,
    }
    if status == 200:
        content = resp_data["choices"][0]["message"]["content"].strip()
        print(f" -> Synthesis Response ({lat:.1f}ms): {content}")
        results["nlp_synthesis"]["content"] = content
    else:
        print(f" -> Synthesis Failed: HTTP {status}, error: {err}")

    time.sleep(1.0)

    # ============================================================
    # 2. VISION LIVE TEST — google/gemma-4-26b-a4b-it:free
    # ============================================================
    vision_model = "google/gemma-4-26b-a4b-it:free"
    opt1_path = "backend/real_data/opt_0611.png"
    opt2_path = "backend/real_data/opt_0810.png"

    if not os.path.exists(opt1_path) or not os.path.exists(opt2_path):
        print(f"[ERROR] Image files missing: {opt1_path}, {opt2_path}")
        return results

    img1 = Image.open(opt1_path)
    img2 = Image.open(opt2_path)
    data_url1, (w1, h1) = _encode_image_to_data_url(img1)
    data_url2, (w2, h2) = _encode_image_to_data_url(img2)

    # T1 VQA
    print(f"\n[3] Testing Vision Model (T1 VQA): {vision_model} on opt_0611.png...")
    vqa_payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What objects and major land-cover types are visible in this satellite image?"},
                    {"type": "image_url", "image_url": {"url": data_url1}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    status, resp_data, lat, err = _execute_http(vqa_payload)
    results["vision_vqa"] = {
        "model": vision_model,
        "status": status,
        "latency_ms": round(lat, 2),
        "error": err,
    }
    if status == 200:
        content = resp_data["choices"][0]["message"]["content"].strip()
        print(f" -> T1 VQA Response ({lat:.1f}ms): {content}")
        results["vision_vqa"]["content"] = content
    else:
        print(f" -> T1 VQA Failed: HTTP {status}, error: {err}")

    time.sleep(1.5)

    # T2 Caption
    print(f"\n[4] Testing Vision Model (T2 Caption): {vision_model} on opt_0611.png...")
    cap_payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this satellite scene concisely, mentioning major land-cover types and visible infrastructure."},
                    {"type": "image_url", "image_url": {"url": data_url1}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    status, resp_data, lat, err = _execute_http(cap_payload)
    results["vision_caption"] = {
        "model": vision_model,
        "status": status,
        "latency_ms": round(lat, 2),
        "error": err,
    }
    if status == 200:
        content = resp_data["choices"][0]["message"]["content"].strip()
        print(f" -> T2 Caption Response ({lat:.1f}ms): {content}")
        results["vision_caption"]["content"] = content
    else:
        print(f" -> T2 Caption Failed: HTTP {status}, error: {err}")

    time.sleep(1.5)

    # T3 Grounding
    print(f"\n[5] Testing Vision Model (T3 Grounding): {vision_model} on opt_0810.png...")
    ground_payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a remote-sensing grounding model. Locate the requested objects and respond ONLY with JSON: "
                    '{"objects": [{"label": "building", "box": [x0, y0, x1, y1]}]}. Coordinates must be normalized in [0.0, 1.0].'
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Locate the major buildings visible in this satellite image."},
                    {"type": "image_url", "image_url": {"url": data_url2}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }

    status, resp_data, lat, err = _execute_http(ground_payload)
    results["vision_grounding"] = {
        "model": vision_model,
        "status": status,
        "latency_ms": round(lat, 2),
        "error": err,
    }
    if status == 200:
        content = resp_data["choices"][0]["message"]["content"].strip()
        print(f" -> T3 Grounding Response ({lat:.1f}ms): {content}")
        results["vision_grounding"]["content"] = content
        
        # Coordinate & Box Validation
        try:
            parsed = json.loads(content)
            boxes = parsed.get("objects", [])
            valid_boxes = []
            for b in boxes:
                coords = b.get("box", [])
                if len(coords) == 4 and coords[0] <= coords[2] and coords[1] <= coords[3]:
                    valid_boxes.append(coords)
            results["vision_grounding"]["valid_boxes_count"] = len(valid_boxes)
            results["vision_grounding"]["total_boxes_count"] = len(boxes)
            print(f" -> Valid Bounding Boxes parsed: {len(valid_boxes)}/{len(boxes)}")
        except Exception as parse_e:
            results["vision_grounding"]["parse_error"] = str(parse_e)
            print(f" -> Grounding JSON parse error: {parse_e}")
    else:
        print(f" -> T3 Grounding Failed: HTTP {status}, error: {err}")

    # Save summary
    with open("tests/evaluation/live_smoke_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_live_smoke_test()
