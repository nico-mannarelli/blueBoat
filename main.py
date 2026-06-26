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
from sonar_dashboard import SonarDashboard
from detection_log import DetectionLog
from mavlink_client import MAVLinkClient, VehicleState
from planner_handoff import send_to_planner
from export_coords import export_from_log

HOST     = os.environ.get("HOST", "192.168.2.2")
PORT     = int(os.environ.get("SONAR_PORT", 7077))
MAV_PORT = int(os.environ.get("MAV_PORT",   6040))

# Display palette: "blue" (default), "amber" (SonarView-style), or "gray". Cosmetic only.
PALETTE  = os.environ.get("SONAR_PALETTE", "blue")

# Detection on/off. NO_DETECT=1 shows clean imagery only — no markers, no
# contact list. Leave it off (default) when you need the coord list for handoff.
NO_DETECT = os.environ.get("NO_DETECT", "0") not in ("0", "", "false", "False")
# Detector: "blob" (default — large-shape selective, best precision on textured
# seabed), "cfar", "both", "roi", "blob_cfar". DETECT_GAMMA drives the blob
# feature image (gamma>1 pushes dim speckle to black; 1.8 is the tuned value).
DETECTOR     = os.environ.get("DETECTOR", "blob")
DETECT_GAMMA = float(os.environ.get("DETECT_GAMMA", "1.8"))
# If set, write the contact list as a Python array (coords = [(lat, lon, 0), ...])
# to this path at the end of the run. e.g. COORDS_OUT=contacts_coords.py
COORDS_OUT = os.environ.get("COORDS_OUT")
# Optional: keep only the N largest contacts by detection area (e.g.
# COORDS_LARGEST=50) and drop contacts seen on fewer than COORDS_MIN_HITS pings.
COORDS_LARGEST = os.environ.get("COORDS_LARGEST")
COORDS_LARGEST = int(COORDS_LARGEST) if COORDS_LARGEST else None
COORDS_MIN_HITS = int(os.environ.get("COORDS_MIN_HITS", "1"))

# Autonomous handoff: at end of survey, send the contacts to the
# revisit planner (see planner_handoff.py). On by default; SEND_TO_PLANNER=0
# disables it. We send the largest PLANNER_LARGEST contacts (default 50), each
# seen on >= PLANNER_MIN_HITS pings (default 2, to drop one-ping flickers).
SEND_TO_PLANNER = os.environ.get("SEND_TO_PLANNER", "1") not in ("0", "", "false", "False")
PLANNER_LARGEST = int(os.environ.get("PLANNER_LARGEST", "50"))
PLANNER_MIN_HITS = int(os.environ.get("PLANNER_MIN_HITS", "2"))

# Side-scan look direction per channel, as an offset from vehicle heading.
# NOTE: port=0 / starboard=1 mapping mirrors the XTF channel order but should
# be confirmed against the live stream on the boat.
CHANNEL_SIDE_OFFSET = {0: -90.0, 1: +90.0}

# ---- shared state ----------------------------------------------------------

vehicle    = VehicleState()   # updated by mavlink_client thread
parser     = OmniScanParser()
detectors  = {}               # channel_number -> WaterfallDetector
dashboards = {}               # channel_number -> SonarDashboard
log        = DetectionLog(merge_radius_m=3.0)   # de-duplicated mission contacts


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
    return f"OmniScan 450 - channel {channel}"


def _dashboard(channel):
    """One dashboard window per channel; all share the single contact log."""
    d = dashboards.get(channel)
    if d is None:
        d = SonarDashboard(title=_window_name(channel), palette=PALETTE,
                           source_label=f"{HOST}:{PORT}  ch{channel}", mode="LIVE")
        dashboards[channel] = d
    return d


def draw(objects, ping, image):
    channel = ping.get("channel_number", 0)
    fix = vehicle.fix
    if fix:
        log.add_fix(fix["lat"], fix["lon"])
    frame = _dashboard(channel).render(image, objects, ping, vehicle, log)
    cv2.imshow(_window_name(channel), frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        raise KeyboardInterrupt


# ---- detection callback ----------------------------------------------------

def handle_detection(objects, ping, image):
    # annotate each detection with range + contact id and fold it into the
    # de-duplicated contact log (log.contacts is the mission's hand-off list)
    for obj in objects:
        range_mm = ping["start_mm"] + obj["x"] * ping["mm_per_sample"]
        obj["range_m"] = range_mm / 1000.0
        latlon = georeference(obj, ping)
        if latlon:
            c = log.add(latlon[0], latlon[1], range_m=obj["range_m"],
                        size=(obj["w"], obj["h"]),
                        source=obj.get("source", "cfar"),
                        score=obj.get("score", 0.0),
                        ping_number=ping["ping_number"])
            obj["cid"] = c["id"]

    if image is not None:
        draw(objects, ping, image)

    channel = ping.get("channel_number", 0)
    if not objects:
        print(f"[detect] ch{channel} ping #{ping['ping_number']}: clear")
        return

    fix     = vehicle.fix
    gps_tag = (
        f"  gps=({fix['lat']:.6f}, {fix['lon']:.6f})  hdg={fix['heading_deg']:.1f}deg"
        if fix else "  gps=no-fix"
    )
    print(f"[detect] ch{channel} ping #{ping['ping_number']}: "
          f"{len(objects)} object(s){gps_tag}")

    for obj in objects:
        latlon = georeference(obj, ping)
        loc    = f"({latlon[0]:.6f}, {latlon[1]:.6f})" if latlon else "no-fix"
        print(
            f"    range~{obj.get('range_m', 0):.1f}m  "
            f"size={obj['w']}x{obj['h']}px  "
            f"src={obj.get('source','?')}  contact=#{obj.get('cid','-')}  "
            f"latlon={loc}"
        )

    # The full mission contact list is `log` — handed to the revisit planner at
    # end of survey via planner_handoff.py.


# ---- sonar pipeline --------------------------------------------------------

def _get_detector(channel):
    """Lazily create one detector per channel so port/starboard never mix."""
    det = detectors.get(channel)
    if det is None:
        det = WaterfallDetector(
            max_rows=500,
            detect_every=50,
            display_every=5,
            detector="off" if NO_DETECT else DETECTOR,
            detect_gamma=DETECT_GAMMA,
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
        print(f"\n[main] mission complete: {len(log)} unique contact(s)")
        for r in log.to_records():
            print(f"    #{r['id']:02d}  ({r['lat']:.6f}, {r['lon']:.6f})  "
                  f"range~{r['range_m']}m  {r['source']}  seen {r['hits']}x")
        if len(log):
            print("[main] hand off log.to_records() / to_geojson() / to_csv() "
                  "to your revisit planner")
        if COORDS_OUT:
            path, n = export_from_log(log, path=COORDS_OUT,
                                      largest=COORDS_LARGEST,
                                      min_hits=COORDS_MIN_HITS)
            print(f"[main] wrote {n} coord(s) to {path} "
                  f"(import with: from {os.path.splitext(os.path.basename(path))[0]} "
                  "import coords)")
        # End-of-survey: hand the largest contacts to the revisit planner so it
        # can plan a revisit run after this mission (see planner_handoff.py).
        if SEND_TO_PLANNER and len(log):
            coords = log.to_coords(largest=PLANNER_LARGEST, min_hits=PLANNER_MIN_HITS)
            print(f"[main] sending {len(coords)} contact(s) to revisit planner "
                  f"(largest {PLANNER_LARGEST} by area, hits >= {PLANNER_MIN_HITS})")
            send_to_planner(coords)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
