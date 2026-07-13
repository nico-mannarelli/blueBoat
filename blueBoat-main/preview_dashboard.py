"""
preview_dashboard.py
Headless smoke-test / preview for the dashboard. Drives the REAL
WaterfallDetector with synthetic dB pings (a noisy seabed plus a bright
along-track target) and writes a dashboard PNG — no GUI, no XTF, no hardware.

    python preview_dashboard.py          # writes dashboard_preview.png

Useful for eyeballing dashboard changes and confirming the
detector -> detection_log -> dashboard wiring still renders end to end.
"""
import math
import numpy as np
import cv2

from sonar_detect import WaterfallDetector
from sonar_dashboard import SonarDashboard
from detection_log import DetectionLog

N_COLS = 400
MM_PER_SAMPLE = 25_000.0 / N_COLS      # 25 m slant range
MIN_DB, MAX_DB = -90.0, -40.0


class _Vehicle:
    """Minimal VehicleState stand-in (avoids importing websocket)."""
    def __init__(self):
        self._lat = self._lon = self._hdg = None

    def update_gps(self, lat, lon, alt_m, timestamp_ms=None):
        self._lat, self._lon = lat, lon

    def update_heading(self, h):
        self._hdg = h % 360.0

    @property
    def fix(self):
        if self._lat is None or self._hdg is None:
            return None
        return {"lat": self._lat, "lon": self._lon, "alt_m": 0,
                "heading_deg": self._hdg, "timestamp_ms": 0}


def _make_ping(i, rng):
    col = np.arange(N_COLS)
    seabed = (-55.0 - 18.0 * (col / N_COLS)) + rng.normal(0, 1.1, N_COLS)
    seabed = np.convolve(seabed, np.ones(3) / 3, mode="same")
    if 30 < i < 230:
        seabed[147:153] += 18.0                 # along-track "log"
    if 118 <= i <= 126:
        seabed[250:257] += 15.0                 # a rock
    samples = np.clip(seabed, MIN_DB, MAX_DB)
    lat = 37.42100 + i * 1.5e-6
    lon = -122.17600 + i * 0.6e-6
    hdg = 35.0 + 3 * math.sin(i / 40)
    ping = {
        "type": "ping", "ping_number": 1000 + i, "channel_number": 1,
        "start_mm": 0, "length_mm": 25000.0, "num_results": N_COLS,
        "timestamp_ms": i * 50, "vehicle_heading_deg": hdg,
        "transducer_heading_deg": hdg, "min_pwr_db": MIN_DB, "max_pwr_db": MAX_DB,
        "samples_db": samples.tolist(), "mm_per_sample": MM_PER_SAMPLE,
        "nadir_col": 0, "side": "stbd",
    }
    return ping, lat, lon, hdg


def main(out="dashboard_preview.png"):
    rng = np.random.default_rng(7)
    vehicle = _Vehicle()
    log = DetectionLog(merge_radius_m=3.0)
    dash = SonarDashboard(title="OmniScan 450 - Preview", palette="blue",
                          source_label="synthetic", mode="REPLAY 4x")
    last = {}

    def georef(obj, ping):
        fix = vehicle.fix
        if not fix:
            return None
        r = abs(obj["x"]) * ping["mm_per_sample"] / 1000.0
        brg = math.radians((ping["transducer_heading_deg"] + 90.0) % 360)
        dlat = r * math.cos(brg) / 111_111.0
        dlon = r * math.sin(brg) / (111_111.0 * math.cos(math.radians(fix["lat"])))
        return fix["lat"] + dlat, fix["lon"] + dlon, r

    def on_detection(objects, ping, image):
        for o in objects:
            g = georef(o, ping)
            if g:
                o["range_m"] = g[2]
                c = log.add(g[0], g[1], range_m=g[2], size=(o["w"], o["h"]),
                            source=o.get("source", "cfar"),
                            score=o.get("score", 0.0), ping_number=ping["ping_number"])
                o["cid"] = c["id"]
        last["frame"] = dash.render(image, objects, ping, vehicle, log)

    def on_frame(objects, ping, image):
        last["frame"] = dash.render(image, objects, ping, vehicle, log)

    det = WaterfallDetector(max_rows=500, detect_every=50, display_every=10,
                            on_detection=on_detection, on_frame=on_frame)

    for i in range(260):
        ping, lat, lon, hdg = _make_ping(i, rng)
        vehicle.update_gps(lat=lat, lon=lon, alt_m=0)
        vehicle.update_heading(hdg)
        log.add_fix(lat, lon)
        det.add_ping(ping)

    if "frame" in last:
        cv2.imwrite(out, last["frame"])
        print(f"wrote {out}   ({len(log)} contact(s))")
    else:
        print("no frame produced")


if __name__ == "__main__":
    main()
