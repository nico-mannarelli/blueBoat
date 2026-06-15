"""
main.py
Wires the pipeline together:

    mavlink2rest WebSocket  ──────────────────────────────┐
                                                          ▼
    SonarLink WebSocket  →  OmniScanParser  →  WaterfallDetector(s)
                                                          │
                                               on_frame()        ← live display
                                               handle_detection() ← detect + georeference

Channels
--------
The OmniScan 450 side-scan sends each channel as its own os_mono_profile
message, tagged with channel_number (0 = port, 1 = starboard — matching the
two channels SonarView writes to XTF). We must NOT interleave them into one
waterfall, so each channel_number gets its own detector and display window.

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

# Side-scan look direction per channel, as an offset from vehicle heading.
# NOTE: port=0 / starboard=1 mapping mirrors the XTF channel order but should
# be confirmed against the live stream on the boat.
CHANNEL_SIDE_OFFSET = {0: -90.0, 1: +90.0}

# ---- shared state ----------------------------------------------------------

vehicle   = VehicleState()   # updated by mavlink_client thread
parser    = OmniScanParser()
detectors = {}               # channel_number -> WaterfallDetector


# ---- georeferencing --------------------------------------------------------

def georeference(detection, ping):
    """Convert a side-scan detection to approximate lat/lon, or None.

    Range  = detection column (sample index) * mm_per_sample.
    Bearing = vehicle heading + 90° (starboard) or - 90° (port), per channel.
    Dead-reckoned from the latest vehicle fix.
    """
    fix = vehicle.fix
    if fix is None:
        return None

    range_m = (ping["start_mm"] + detection["x"] * ping["mm_per_sample"]) / 1000.0
    side    = CHANNEL_SIDE_OFFSET.get(ping.get("channel_number", 0), 0.0)
    bearing = math.radians((ping["transducer_heading_deg"] + side) % 360.0)
    ref_lat, ref_lon = fix["lat"], fix["lon"]

    dlat = range_m * math.cos(bearing) / 111_111.0
    dlon = range_m * math.sin(bearing) / (111_111.0 * math.cos(math.radians(ref_lat)))
    return ref_lat + dlat, ref_lon + dlon


# ---- live display ----------------------------------------------------------

def _window_name(channel):
    return f"OmniScan 450 — channel {channel}"


def draw(objects, ping, image):
    channel = ping.get("channel_number", 0)
    display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for obj in objects:
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        if obj.get("source") == "hough":
            cx, cy, r = x + w // 2, y + h // 2, w // 2
            cv2.circle(display, (cx, cy), r, (255, 255, 0), 1)
        else:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)

    fix = vehicle.fix
    gps_str = f"gps=({fix['lat']:.5f},{fix['lon']:.5f})" if fix else "no-fix"
    label = (f"ch{channel}  ping #{ping['ping_number']}  {gps_str}  "
             f"{len(objects)} detection(s)")
    cv2.putText(display, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow(_window_name(channel), display)
    cv2.waitKey(1)


# ---- detection callback ----------------------------------------------------

def handle_detection(objects, ping, image):
    if image is not None:
        draw(objects, ping, image)

    channel = ping.get("channel_number", 0)
    if not objects:
        print(f"[detect] ch{channel} ping #{ping['ping_number']}: clear")
        return

    fix     = vehicle.fix
    gps_tag = (
        f"  gps=({fix['lat']:.6f}, {fix['lon']:.6f})  hdg={fix['heading_deg']:.1f}°"
        if fix else "  gps=no-fix"
    )
    print(f"[detect] ch{channel} ping #{ping['ping_number']}: "
          f"{len(objects)} object(s){gps_tag}")

    for obj in objects:
        range_mm = ping["start_mm"] + obj["x"] * ping["mm_per_sample"]
        latlon   = georeference(obj, ping)
        loc      = f"({latlon[0]:.6f}, {latlon[1]:.6f})" if latlon else "no-fix"
        print(
            f"    range~{range_mm/1000:.1f}m  "
            f"size={obj['w']}x{obj['h']}px  "
            f"src={obj.get('source','?')}  "
            f"latlon={loc}"
        )

    # TODO: write detections to GeoJSON / post to operator dashboard


# ---- sonar pipeline --------------------------------------------------------

def _get_detector(channel):
    """Lazily create one detector per channel so port/starboard never mix."""
    det = detectors.get(channel)
    if det is None:
        det = WaterfallDetector(
            max_rows=500,
            detect_every=50,
            display_every=5,
            on_detection=handle_detection,
            on_frame=draw,
        )
        detectors[channel] = det
    return det


def on_bytes(raw):
    for packet in parser.feed(raw):
        if packet["type"] == "ping":
            _get_detector(packet.get("channel_number", 0)).add_ping(packet)
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
