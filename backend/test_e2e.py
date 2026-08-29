"""End-to-end verification of all 5 representative SIH26167 query flows."""
import json
import urllib.request
import uuid

BASE = "http://127.0.0.1:8000/api/query"
IMG = "C:/Users/Y.shankar/satquery-ai/backend/test_images/"


def multipart(query, paths):
    boundary = uuid.uuid4().hex
    body = bytearray()
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"query\"\r\n\r\n{query}\r\n".encode()
    for p in paths:
        data = open(p, "rb").read()
        name = p.split("/")[-1]
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
                 f"filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n").encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def run(label, query, images):
    body, ctype = multipart(query, [IMG + i for i in images])
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    print("=" * 70)
    print(f"[{label}]")
    if "error" in d:
        print("  ERROR:", d["error"])
        return
    print("  scenario :", d["scenario"])
    print("  tools    :", [o["tool"] for o in d["outputs"]])
    print("  conf     :", d["confidence"], "| time:", d["trace"]["steps"][-1]["total_ms"], "ms")
    ans = d["answer"].replace("\n", " ")
    print("  answer   :", ans[:260] + ("..." if len(ans) > 260 else ""))
    print("  evidence imgs:", len(d.get("evidence_images_b64", [])))


run("1. Single-image VQA (mandatory)", 
    "Describe the land-cover and major objects visible in this image.",
    ["optical_t0.png"])

run("2. Grounding",
    "Highlight the water body referred to in the query.",
    ["optical_t0.png"])

run("3. Bi-temporal change (mandatory)",
    "What changed between these two dates, and where did the change occur?",
    ["optical_t0.png", "optical_t1.png"])

run("4. Optical-SAR fusion (cross-modal)",
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
    ["optical_t1.png", "sar_t1.png"])

run("5. Built-up change quantification",
    "Has the built-up area increased, decreased, or remained unchanged?",
    ["optical_t0.png", "optical_t1.png"])

print("\nALL FLOWS EXECUTED.")
