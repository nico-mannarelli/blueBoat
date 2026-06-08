"""
main.py
Wires the three modules together:

    SonarLink WebSocket  ->  OmniScanParser  ->  WaterfallDetector

Run it:
    python main.py

The detector calls handle_detection() whenever it finds candidate objects.
That's the single place where you'll later add: convert (x,y) -> range/bearing,
fuse with GPS from MAVLink, and fire a MAVLink command to the drone.
"""

from sonar_ws import SonarLinkClient
from sonar_parse import OmniScanParser
from sonar_detect import WaterfallDetector

HOST = "192.168.2.2"
PORT = 7077

parser = OmniScanParser()


def handle_detection(objects, ping, image):
    if not objects:
        print(f"[detect] ping #{ping['ping_number']}: clear")
        return

    print(f"[detect] ping #{ping['ping_number']}: {len(objects)} object(s)")
    for obj in objects:
        # y in the image is distance from the transducer; convert to mm.
        # (x is along-track position, i.e. which ping -- needs GPS to localize)
        range_mm = ping["start_mm"] + obj["y"] * ping["mm_per_sample"]
        print(
            f"    at along-track={obj['x']}, "
            f"range~{range_mm/1000:.1f} m, "
            f"size={obj['w']}x{obj['h']} px, "
            f"heading={ping['vehicle_heading_deg']:.1f} deg"
        )
    # TODO: fuse range/bearing + latest GPS fix -> object lat/long
    # TODO: send MAVLink command to drone


detector = WaterfallDetector(
    max_rows=500,
    detect_every=50,
    on_detection=handle_detection,
)


def on_bytes(raw):
    # Bytes in -> pings out -> into the detector. No global state, no spaghetti.
    for ping in parser.feed(raw):
        detector.add_ping(ping)


def main():
    client = SonarLinkClient(host=HOST, port=PORT, on_bytes=on_bytes)
    print("[main] starting. Make sure SonarView is running with the sonar on.")
    client.run_forever(auto_reconnect=True)


if __name__ == "__main__":
    main()
