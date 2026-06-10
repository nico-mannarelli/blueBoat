"""
main.py
Wires the three modules together:

    SonarLink WebSocket  ->  OmniScanParser  ->  WaterfallDetector

The parser yields two packet types:
  "ping"  — sonar profile, fed straight into the detector
  "gps"   — MAVLink GLOBAL_POSITION_INT from the vehicle, kept as latest fix

When the detector fires, handle_detection() fuses each bounding box with
the latest GPS fix to produce an approximate lat/lon for the anomaly.

Run it:
    python main.py

To run against the mock server instead of the boat:
    HOST=127.0.0.1 python main.py
"""

import math
import os

from sonar_ws import SonarLinkClient
from sonar_parse import OmniScanParser
from sonar_detect import WaterfallDetector

HOST = os.environ.get("HOST", "192.168.2.2")
PORT = 7077

parser     = OmniScanParser()
latest_gps = None   # most recent {"lat", "lon", "alt_m", "heading_deg"}


# ---- georeferencing --------------------------------------------------------

def georeference(detection, ping):
    """Convert a waterfall bounding box to approximate lat/lon.

    The OmniScan 450 records the transducer heading per ping. A detection
    at sample row Y in the waterfall is at:
      - range = start_mm + Y * mm_per_sample
      - bearing = transducer_heading_deg of the ping it came from

    We then dead-reckon from the latest GPS fix.
    Returns (lat, lon) in decimal degrees, or None if no GPS fix yet.
    """
    if latest_gps is None:
        return None

    range_m  = (ping["start_mm"] + detection["y"] * ping["mm_per_sample"]) / 1000.0
    bearing  = math.radians(ping["transducer_heading_deg"])
    ref_lat  = latest_gps["lat"]
    ref_lon  = latest_gps["lon"]

    dlat = range_m * math.cos(bearing) / 111_111.0
    dlon = range_m * math.sin(bearing) / (111_111.0 * math.cos(math.radians(ref_lat)))

    return ref_lat + dlat, ref_lon + dlon


# ---- detection callback ----------------------------------------------------

def handle_detection(objects, ping, _image):
    if not objects:
        print(f"[detect] ping #{ping['ping_number']}: clear")
        return

    gps_tag = (
        f"  gps=({latest_gps['lat']:.6f}, {latest_gps['lon']:.6f})"
        if latest_gps else "  gps=none"
    )
    print(f"[detect] ping #{ping['ping_number']}: {len(objects)} object(s){gps_tag}")

    for obj in objects:
        range_mm = ping["start_mm"] + obj["y"] * ping["mm_per_sample"]
        latlon   = georeference(obj, ping)
        loc      = f"({latlon[0]:.6f}, {latlon[1]:.6f})" if latlon else "no-fix"
        print(
            f"    range~{range_mm/1000:.1f}m  "
            f"heading={ping['transducer_heading_deg']:.1f}deg  "
            f"size={obj['w']}x{obj['h']}px  "
            f"latlon={loc}"
        )

    # TODO: write detections to GeoJSON or post to operator dashboard


# ---- pipeline --------------------------------------------------------------

detector = WaterfallDetector(
    max_rows=500,
    detect_every=50,
    on_detection=handle_detection,
)


def on_bytes(raw):
    global latest_gps
    for packet in parser.feed(raw):
        if packet["type"] == "gps":
            latest_gps = packet
        elif packet["type"] == "ping":
            detector.add_ping(packet)


def main():
    client = SonarLinkClient(host=HOST, port=PORT, on_bytes=on_bytes)
    print(f"[main] connecting to {HOST}:{PORT}")
    client.run_forever(auto_reconnect=True)


if __name__ == "__main__":
    main()
