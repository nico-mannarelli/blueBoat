"""
simulate.py
Feed synthetic ping dicts directly into WaterfallDetector to verify the
waterfall image pipeline and tune MSER parameters — no hardware required.

Scenario: 200 pings of background noise. Between pings 80-150 a bright
"target" blob is inserted at sample index 100 (~5 m range). Run it and check:
  - waterfall.png looks right (bright stripe mid-frame)
  - detections print for the target region
  - "clear" prints for noise-only pings
"""

import sys
import numpy as np
import cv2

from sonar_detect import WaterfallDetector

NUM_PINGS = 200
NUM_SAMPLES = 200
START_MM = 0
LENGTH_MM = 10_000   # 10 m range
MM_PER_SAMPLE = LENGTH_MM / NUM_SAMPLES


def _ping(ping_number, samples):
    return {
        "ping_number": ping_number,
        "start_mm": START_MM,
        "length_mm": LENGTH_MM,
        "num_results": NUM_SAMPLES,
        "vehicle_heading_deg": float(ping_number % 360),
        "min_pwr_db": -90.0,
        "max_pwr_db": -40.0,
        "samples_db": samples,
        "mm_per_sample": MM_PER_SAMPLE,
    }


def noise_ping(ping_number):
    samples = np.random.uniform(-80.0, -65.0, NUM_SAMPLES).tolist()
    return _ping(ping_number, samples)


def target_ping(ping_number, target_sample=100, half_width=8, strength=-48.0):
    samples = np.random.uniform(-80.0, -65.0, NUM_SAMPLES)
    lo = max(0, target_sample - half_width)
    hi = min(NUM_SAMPLES, target_sample + half_width)
    samples[lo:hi] = strength
    return _ping(ping_number, samples.tolist())


# ---- detection callback ----------------------------------------------------
last_image = [None]
all_detections = []


def on_detection(objects, ping, image):
    last_image[0] = image
    tag = f"ping #{ping['ping_number']}"
    if not objects:
        print(f"[detect] {tag}: clear")
        return
    print(f"[detect] {tag}: {len(objects)} object(s)")
    for obj in objects:
        range_mm = ping["start_mm"] + obj["y"] * ping["mm_per_sample"]
        print(
            f"    along-track={obj['x']}px  range~{range_mm/1000:.1f}m"
            f"  size={obj['w']}x{obj['h']}px"
        )
    all_detections.extend(objects)


# ---- run -------------------------------------------------------------------
detector = WaterfallDetector(
    max_rows=500,
    detect_every=50,
    on_detection=on_detection,
)

print(f"[simulate] feeding {NUM_PINGS} pings  (target at pings 80-150, sample 100)")
for i in range(NUM_PINGS):
    ping = target_ping(i) if 80 <= i <= 150 else noise_ping(i)
    detector.add_ping(ping)

print(f"\n[simulate] done — {len(all_detections)} total MSER detections")

if last_image[0] is not None:
    out = "waterfall.png"
    cv2.imwrite(out, last_image[0])
    print(f"[simulate] waterfall image saved → {out}")
else:
    print("[simulate] no image produced (fewer than detect_every pings ran?)")
    sys.exit(1)
