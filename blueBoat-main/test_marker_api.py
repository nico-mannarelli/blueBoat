"""
test_marker_api.py — fire ONE test POST at the map marker endpoint and print
the response. Run it from a machine that can reach the server (the ground-
station), not necessarily a dev laptop.

    python test_marker_api.py                      # hits the real server
    python test_marker_api.py --url http://127.0.0.1:8000/marker/boat   # a mock
    python test_marker_api.py --low crop.png --high snap.png            # real imgs

Self-contained: with no --low/--high it synthesizes a 224x224 test image
(a bright box drawn at the bbox, plus corner markers), so you need no sample
files. On the map you can then SEE whether the server's drawn box lands on
our bright box — that confirms the bbox format ([x,y,w,h] vs [x0,y0,x1,y1]).

What it proves: the endpoint is reachable, it accepts our multipart format
(low_res_img + high_res_img + bbox + lat/lon), and what it sends back.
"""

import argparse

import cv2
import numpy as np
import requests

REAL_URL = "http://10.107.30.63:30932/marker/boat"
BBOX = [90, 4, 120, 170]          # interpreted here as [x, y, w, h]


def synth_png():
    """A 224x224 test image: mid-grey field, a bright rectangle drawn at BBOX
    (as x,y,w,h), and 'TL' text top-left so orientation is unmistakable."""
    img = np.full((224, 224), 90, np.uint8)
    x, y, w, h = BBOX
    cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
    cv2.putText(img, "TL", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    return cv2.imencode(".png", img)[1].tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=REAL_URL)
    ap.add_argument("--low", help="low_res_img path (default: synthesized)")
    ap.add_argument("--high", help="high_res_img path (default: synthesized)")
    args = ap.parse_args()

    lo = open(args.low, "rb").read() if args.low else synth_png()
    hi = open(args.high, "rb").read() if args.high else synth_png()

    payload = {
        "lat": "38.142122",
        "lon": "-76.528510",
        "bbox": str(BBOX),          # "[90, 4, 120, 170]"
    }
    files = [
        ("low_res_img",  ("low.png",  lo, "image/png")),
        ("high_res_img", ("high.png", hi, "image/png")),
    ]
    print(f"POST {args.url}")
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
