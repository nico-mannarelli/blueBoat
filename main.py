"""
main.py
Wires the pipeline together:

    mavlink2rest WebSocket  ──────────────────────────────┐
                                                          ▼
    SonarLink WebSocket  →  OmniScanParser  →  WaterfallDetector
                                                          │
                                               handle_detection()
                                               georeference() → lat/lon

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

from sonar_ws import SonarLinkClient
from sonar_parse import OmniScanParser
from sonar_detect import WaterfallDetector
from mavlink_client import MAVLinkClient, VehicleState

HOST     = os.environ.get("HOST", "192.168.2.2")
PORT     = int(os.environ.get("SONAR_PORT", 7077))
MAV_PORT = int(os.environ.get("MAV_PORT",   6040))

# ---- shared state ----------------------------------------------------------

vehicle = VehicleState()   # updated by mavlink_client thread
parser  = OmniScanParser()


# ---- georeferencing --------------------------------------------------------

def georeference(detection, ping):
    """Convert a waterfall bounding box to approximate lat/lon.

    Range = start_mm + sample_row * mm_per_sample.
    Bearing = transducer_heading_deg recorded per ping.
    Dead-reckoned from latest vehicle fix.
    Returns (lat, lon) or None if no fix yet.
    """
    fix = vehicle.fix
    if fix is None:
        return None

    range_m = (ping["start_mm"] + detection["y"] * ping["mm_per_sample"]) / 1000.0
    bearing = math.radians(ping["transducer_heading_deg"])
    ref_lat = fix["lat"]
    ref_lon = fix["lon"]

    dlat = range_m * math.cos(bearing) / 111_111.0
    dlon = range_m * math.sin(bearing) / (111_111.0 * math.cos(math.radians(ref_lat)))

    return ref_lat + dlat, ref_lon + dlon


# ---- detection callback ----------------------------------------------------

def handle_detection(objects, ping, _image):
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
        range_mm = ping["start_mm"] + obj["y"] * ping["mm_per_sample"]
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
    on_detection=handle_detection,
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

    # Start mavlink2rest client in background thread
    mav = MAVLinkClient(host=HOST, port=MAV_PORT, state=vehicle)
    mav.start()

    # Run sonar client in main thread (blocking)
    sonar = SonarLinkClient(host=HOST, port=PORT, on_bytes=on_bytes)
    sonar.run_forever(auto_reconnect=True)


if __name__ == "__main__":
    main()
