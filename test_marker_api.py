"""
test_marker_api.py — fire ONE test POST at the map marker endpoint and print
the response. Run it from a machine that can reach the server.

    python test_marker_api.py                      # real sample images below
    python test_marker_api.py --url http://127.0.0.1:8000/marker/boat   # a mock
    python test_marker_api.py --low crop.png --high snap.png            # your imgs

By default it sends a real low-res crop + a real high-res scan. If either
default path is missing (e.g. the groundstation doesn't have sorted_crops),
that image falls back to a synthesized 224x224 test tile so the call still
runs. A successful POST creates a REAL marker on the live map.
"""

import argparse
import os

import cv2
import numpy as np
import requests

REAL_URL = "http://10.107.30.63:30932/marker/boat"
BBOX = [90, 4, 120, 170]          # interpreted here as [x, y, w, h]

# Real sample images shipped in the repo.
DEFAULT_LOW  = "sorted_crops/man_made__c07/c0001_p18249.png"  # 224x224 crop
DEFAULT_HIGH = "sonar_web/data/images/000019.png"             # colorized scan


def synth_png():
    """Fallback 224x224 tile: grey field, a bright box at BBOX, 'TL' label."""
    img = np.full((224, 224), 90, np.uint8)
    x, y, w, h = BBOX
    cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
    cv2.putText(img, "TL", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    return cv2.imencode(".png", img)[1].tobytes()


def load_img(path, label):
    if path and os.path.exists(path):
        print(f"  {label} = {path}")
        with open(path, "rb") as f:
            return f.read()
    print(f"  {label} = <synthesized>  ({path} not found)")
    return synth_png()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=REAL_URL)
    ap.add_argument("--low", help=f"low_res_img (default: {DEFAULT_LOW})")
    ap.add_argument("--high", help=f"high_res_img (default: {DEFAULT_HIGH})")
    args = ap.parse_args()

    print(f"POST {args.url}")
    lo = load_img(args.low or DEFAULT_LOW, "low_res_img ")
    hi = load_img(args.high or DEFAULT_HIGH, "high_res_img")

    payload = {"lat": "38.142122", "lon": "-76.528510", "bbox": str(BBOX)}
    files = [
        ("low_res_img",  ("low.png",  lo, "image/png")),
        ("high_res_img", ("high.png", hi, "image/png")),
    ]
    print(f"  payload = {payload}")
    try:
        r = requests.post(args.url, data=payload, files=files, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"  CONNECTION FAILED: {e}")
        print("  -> not on the same network as the server, or wrong URL/port")
        return
    print(f"  HTTP {r.status_code}")
    print(f"  response: {r.text[:1000]}")


if __name__ == "__main__":
    main()
