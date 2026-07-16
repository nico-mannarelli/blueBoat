"""
crop_saver.py
Save classifier-ready crops of sonar contacts, plus a JSON sidecar per crop.

Each crop is a 224x224 grayscale PNG cut around a detection's bounding box
with generous padding (the acoustic shadow behind an object is often more
diagnostic than the object highlight itself, so we keep it in frame).

Two ways to use it:

1. Offline harvest from a SonarView .xtf recording (no boat needed):
       python crop_saver.py scan.xtf --out crops
   Every detection the real pipeline finds in the replay becomes one PNG +
   one .json sidecar in --out. Hand-sort the PNGs into class folders
   (log/ rock/ man_made/ background/), then build the classifier library:
       python sonar_classifier.py build --crops <sorted-dir> --lib library.npz

2. From a live detection callback (RESCAN phase): call save_crop() with the
   waterfall frame and the detection box.
"""

import argparse
import json
import os

import cv2
import numpy as np

CROP_SIZE = 224        # output side length (matches the classifier input)
CROP_PAD  = 1.6        # box is expanded to pad*max(w,h) square before cropping
STRETCH_PCT = (2, 98)  # percentile contrast stretch so crops are actually
                       # visible — raw dB-scaled waterfall is dim, muddy gray


def _stretch(crop):
    lo, hi = np.percentile(crop, STRETCH_PCT)
    if hi <= lo:
        return crop
    out = (crop.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return out.clip(0, 255).astype(np.uint8)


def save_crop(gray, box, out_dir, cid, meta=None,
              pad=CROP_PAD, size=CROP_SIZE):
    """Cut a padded square crop around `box` and write PNG + JSON sidecar.

    gray : 2-D uint8 waterfall image the detection came from.
    box  : (x, y, w, h) in waterfall pixel coords.
    cid  : contact id (from DetectionLog) — used in the filename so crops
           trace back to contacts.
    meta : dict merged into the sidecar (lat/lon, source, score, ping...).

    Returns the written PNG path.
    """
    x, y, w, h = (int(v) for v in box)
    H, W = gray.shape[:2]

    # square window: pad * the larger box side, centred on the detection
    half = int(max(w, h) * pad / 2) or 1
    cx, cy = x + w // 2, y + h // 2
    x0, x1 = max(0, cx - half), min(W, cx + half)
    y0, y1 = max(0, cy - half), min(H, cy + half)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = _stretch(gray[y0:y1, x0:x1])
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    os.makedirs(out_dir, exist_ok=True)
    ping = (meta or {}).get("ping_number", 0)
    stem = f"c{cid:04d}_p{ping}"
    png_path = os.path.join(out_dir, stem + ".png")
    cv2.imwrite(png_path, crop)

    sidecar = {"contact_id": cid, "bbox": [x, y, w, h],
               "crop_window": [x0, y0, x1 - x0, y1 - y0]}
    sidecar.update(meta or {})
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    return png_path


# ---- offline harvest from an .xtf recording --------------------------------

def harvest(xtf_path, out_dir, channel="both", detector="cfar"):
    """Replay an .xtf through the real detector and save a crop per
    detection. Reuses replay_xtf's machinery so results match production."""
    # imported here so `import crop_saver` stays light for live use
    from replay_xtf import iter_pings, georeference
    from mavlink_client import VehicleState
    from detection_log import DetectionLog
    from sonar_detect import WaterfallDetector

    vehicle = VehicleState()
    log = DetectionLog()
    saved = [0]

    def on_detection(objects, ping, image):
        if image is None:
            return
        for obj in objects:
            if not all(k in obj for k in ("x", "y", "w", "h")):
                continue
            nadir = ping.get("nadir_col", 0)
            range_m = abs(obj["x"] - nadir) * ping["mm_per_sample"] / 1000.0
            latlon = georeference(dict(obj, range_m=range_m), ping, vehicle)
            c = log.add(latlon[0], latlon[1], range_m=range_m,
                        size=(obj["w"], obj["h"]),
                        source=obj.get("source", detector),
                        score=obj.get("score", 0.0),
                        ping_number=ping["ping_number"]) if latlon else None
            meta = {
                "lat": latlon[0] if latlon else None,
                "lon": latlon[1] if latlon else None,
                "range_m": round(range_m, 1),
                "source": obj.get("source", detector),
                "score": round(float(obj.get("score", 0.0)), 2),
                "ping_number": ping["ping_number"],
                "xtf": os.path.basename(xtf_path),
            }
            p = save_crop(image, (obj["x"], obj["y"], obj["w"], obj["h"]),
                          out_dir, c["id"] if c else 0, meta)
            if p:
                saved[0] += 1
                print(f"[crop] {p}  contact=#{c['id'] if c else '-'}")

    det = WaterfallDetector(max_rows=500, detect_every=50, display_every=0,
                            detector=detector, on_detection=on_detection)

    first = None
    for ping, interval, lat, lon in iter_pings(xtf_path, channel=channel):
        if lat and lon:
            vehicle.update_gps(lat=lat, lon=lon, alt_m=0)
        vehicle.update_heading(ping["vehicle_heading_deg"])
        det.add_ping(ping)

    print(f"[crop] done: {saved[0]} crop(s) from {len(log)} unique contact(s) "
          f"-> {out_dir}")
    return saved[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("xtf", help=".xtf recording to harvest crops from")
    ap.add_argument("--out", default="crops", help="output folder")
    ap.add_argument("--channel", default="both", help="both | 0 | 1")
    ap.add_argument("--detector", default="cfar",
                    help="cfar | classical | blob | both (match your survey)")
    args = ap.parse_args()
    harvest(args.xtf, args.out, channel=args.channel, detector=args.detector)


if __name__ == "__main__":
    main()
