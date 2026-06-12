"""
main.py
Wires the pipeline together:

    mavlink2rest WebSocket  ──────────────────────────────┐
                                                          ▼
    SonarLink WebSocket  →  OmniScanParser  →  WaterfallDetector
                                                          │
                                               on_frame()       ← live display (every 5 pings)
                                               handle_detection() ← MSER + georeference (every 50 pings)

Vehicle state (GPS + heading) comes from two sources, in priority order:
  1. mavlink2rest WebSocket (direct, ATTITUDE for heading — primary)
  2. MAVLINK_WRAPPER packets embedded in the SonarLink stream (fallback)

Run against the boat:
    python main.py

Run against the mock server:
    HOST=127.0.0.1 python main.py
"""

import math
import os

import cv2

from sonar_ws import SonarLinkClient
from sonar_parse import OmniScanParser
from sonar_detect import WaterfallDetector
from mavlink_client import MAVLinkClient, VehicleState

HOST     = os.environ.get("HOST", "192.168.2.2")
PORT     = int(os.environ.get("SONAR_PORT", 7077))
MAV_PORT = int(os.environ.get("MAV_PORT",   6040))

_WINDOW = "OmniScan 450 — Live Waterfall"

# ---- shared state ----------------------------------------------------------

vehicle = VehicleState()   # updated by mavlink_client thread
parser  = OmniScanParser()


# ---- georeferencing --------------------------------------------------------

def georeference(detection, ping):
    """Convert a waterfall bounding box to approximate lat/lon.

    Range = start_mm + sample_column * mm_per_sample.
    Bearing = transducer_heading_deg recorded per ping.
    Dead-reckoned from latest vehicle fix.
    Returns (lat, lon) or None if no fix yet.
    """
    fix = vehicle.fix
    if fix is None:
        return None

    # detection["x"] is the column (sample index) = range axis
    range_m = (ping["start_mm"] + detection["x"] * ping["mm_per_sample"]) / 1000.0
    bearing = math.radians(ping["transducer_heading_deg"])
    ref_lat = fix["lat"]
    ref_lon = fix["lon"]

    dlat = range_m * math.cos(bearing) / 111_111.0
    dlon = range_m * math.sin(bearing) / (111_111.0 * math.cos(math.radians(ref_lat)))

    return ref_lat + dlat, ref_lon + dlon


# ---- live display ----------------------------------------------------------

def on_frame(objects, ping, image):
    """Draw the latest waterfall with detection bounding boxes and show it."""
    display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for obj in objects:
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)

    fix = vehicle.fix
    gps_str = f"gps=({fix['lat']:.5f},{fix['lon']:.5f})" if fix else "no-fix"
    label = f"ping #{ping['ping_number']}  {gps_str}  {len(objects)} detection(s)"
    cv2.putText(display, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow(_WINDOW, display)
    cv2.waitKey(1)


# ---- detection callback ----------------------------------------------------

def handle_detection(objects, ping, image):
    # Update the display immediately when detection runs
    if image is not None:
        on_frame(objects, ping, image)

    if not objects:
        print(f"[detect] ping #{ping['ping_number']}: clear")
        return

    fix     = vehicle.fix
    gps_tag = (
        f"  gps=({fix['lat']:.6f}, {fix['lon']:.6f})"
        f"  hdg={fix['heading_deg']:.1f}°"
        if fix else "  gps=no-fix"
    )
    print(f"[detect] ping #{ping['ping_number']}: {len(objects)} object(s){gps_tag}")

    for obj in objects:
        # obj["x"] = column = sample index = range direction
        range_mm = ping["start_mm"] + obj["x"] * ping["mm_per_sample"]
        latlon   = georeference(obj, ping)
        loc      = f"({latlon[0]:.6f}, {latlon[1]:.6f})" if latlon else "no-fix"
        print(
            f"    range~{range_mm/1000:.1f}m  "
            f"bearing={ping['transducer_heading_deg']:.1f}°  "
            f"size={obj['w']}x{obj['h']}px  "
            f"latlon={loc}"
        )

    # TODO: write detections to GeoJSON / post to operator dashboard


# ---- sonar pipeline --------------------------------------------------------

detector = WaterfallDetector(
    max_rows=500,
    detect_every=50,
    display_every=5,
    on_detection=handle_detection,
    on_frame=on_frame,
)


def on_bytes(raw):
    for packet in parser.feed(raw):
        if packet["type"] == "ping":
            detector.add_ping(packet)
        elif packet["type"] == "gps" and not vehicle.ready:
            # MAVLINK_WRAPPER fallback: only use if mavlink2rest hasn't
            # provided a fix yet (e.g. running without BlueOS)
            vehicle.update_gps(
                lat=packet["lat"],
                lon=packet["lon"],
                alt_m=packet["alt_m"],
                timestamp_ms=packet["timestamp_ms"],
            )
            vehicle.update_heading(packet["heading_deg"])


# ---- startup ---------------------------------------------------------------

def main():
    print(f"[main] sonar    →  {HOST}:{PORT}")
    print(f"[main] mavlink  →  {HOST}:{MAV_PORT}")

    mav = MAVLinkClient(host=HOST, port=MAV_PORT, state=vehicle)
    mav.start()

    sonar = SonarLinkClient(host=HOST, port=PORT, on_bytes=on_bytes)
    try:
        sonar.run_forever(auto_reconnect=True)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
