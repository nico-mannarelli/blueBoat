"""
replay_xtf.py
Replay a SonarView .xtf recording through the detection pipeline.

Feeds real ping data into the same WaterfallDetector used in production,
so detection results match what you'd see from the live boat.

Usage:
    python replay_xtf.py scan.xtf               # real-time speed
    python replay_xtf.py scan.xtf --speed 4     # 4x faster
    python replay_xtf.py scan.xtf --speed 0     # no delay (max speed)
    python replay_xtf.py scan.xtf --probe       # print file info and exit
"""

import argparse
import math
import sys
import time

import cv2
import numpy as np
import pyxtf
from pyxtf import XTFHeaderType, XTFSampleFormat

from mavlink_client import VehicleState
from sonar_detect import WaterfallDetector

MIN_PWR_DB = -90.0
MAX_PWR_DB = -40.0
_WINDOW    = "OmniScan 450 — XTF Replay"

# ---- sample conversion -----------------------------------------------------

def _to_db(raw_samples, sample_format):
    """Map raw XTF sample values to dB matching our pipeline's ping dict format."""
    arr = np.asarray(raw_samples, dtype=np.float64)
    span = MAX_PWR_DB - MIN_PWR_DB

    if sample_format == XTFSampleFormat.word:       # uint16  0-65535
        return (MIN_PWR_DB + (arr / 65535.0) * span).tolist()
    elif sample_format == XTFSampleFormat.byte:     # uint8   0-255
        return (MIN_PWR_DB + (arr / 255.0) * span).tolist()
    else:
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.zeros_like(arr)
        return (MIN_PWR_DB + arr * span).tolist()


# ---- georeferencing (mirrors main.py) --------------------------------------

def georeference(detection, ping, vehicle):
    fix = vehicle.fix
    if fix is None:
        return None
    range_m = (ping["start_mm"] + detection["x"] * ping["mm_per_sample"]) / 1000.0
    bearing = math.radians(ping["transducer_heading_deg"])
    ref_lat, ref_lon = fix["lat"], fix["lon"]
    dlat = range_m * math.cos(bearing) / 111_111.0
    dlon = range_m * math.sin(bearing) / (111_111.0 * math.cos(math.radians(ref_lat)))
    return ref_lat + dlat, ref_lon + dlon


# ---- display (mirrors main.py) ---------------------------------------------

def make_on_frame(vehicle):
    def on_frame(objects, ping, image):
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
        label = f"ping #{ping['ping_number']}  {gps_str}  {len(objects)} detection(s)"
        cv2.putText(display, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(_WINDOW, display)
        cv2.waitKey(1)
    return on_frame


def make_on_detection(vehicle):
    def on_detection(objects, ping, image):
        if image is not None:
            make_on_frame(vehicle)(objects, ping, image)

        if not objects:
            print(f"[detect] ping #{ping['ping_number']}: clear")
            return

        fix     = vehicle.fix
        gps_tag = (
            f"  gps=({fix['lat']:.6f},{fix['lon']:.6f})  hdg={fix['heading_deg']:.1f}°"
            if fix else "  gps=no-fix"
        )
        print(f"[detect] ping #{ping['ping_number']}: {len(objects)} object(s){gps_tag}")
        for obj in objects:
            range_mm = ping["start_mm"] + obj["x"] * ping["mm_per_sample"]
            latlon   = georeference(obj, ping, vehicle)
            loc      = f"({latlon[0]:.6f},{latlon[1]:.6f})" if latlon else "no-fix"
            print(
                f"    range~{range_mm/1000:.1f}m  "
                f"bearing={ping['transducer_heading_deg']:.1f}°  "
                f"size={obj['w']}x{obj['h']}px  "
                f"src={obj.get('source','?')}  "
                f"latlon={loc}"
            )
    return on_detection


# ---- XTF iteration ---------------------------------------------------------

def iter_pings(path):
    """Yield (ping_dict, ping_interval_sec) for every sonar ping in the file."""
    file_header, packets = pyxtf.xtf_read(path)

    if XTFHeaderType.sonar not in packets:
        raise ValueError("No sonar channel found in XTF file.")

    sonar_packets = packets[XTFHeaderType.sonar]

    # Determine sample format from the first channel header
    first = sonar_packets[0]
    chan_hdr   = first.ping_chan_headers[0]
    sample_fmt = XTFSampleFormat(chan_hdr.SampleFormat) if hasattr(chan_hdr, "SampleFormat") else XTFSampleFormat.word

    prev_time = None

    for i, pkt in enumerate(sonar_packets):
        ph       = pkt.ping_header
        ch       = pkt.ping_chan_headers[0]
        raw_data = pkt.data[0]

        num_samples = int(ch.NumSamples)
        slant_m     = float(ch.SlantRange)
        if slant_m <= 0:
            slant_m = 10.0   # fallback: 10 m range

        length_mm    = slant_m * 1000.0
        mm_per_sample = length_mm / num_samples if num_samples > 0 else 1.0
        samples_db   = _to_db(raw_data[:num_samples], sample_fmt)

        # Fractional seconds into the day for interval estimation
        t = float(ph.Hour) * 3600 + float(ph.Minute) * 60 + float(ph.Second) + float(ph.HSeconds) / 100.0
        interval = (t - prev_time) if (prev_time is not None and t > prev_time) else 0.05
        interval = min(interval, 1.0)   # cap at 1 s in case of time jumps
        prev_time = t

        ping = {
            "type":                   "ping",
            "ping_number":            int(ph.PingNumber) if ph.PingNumber else i,
            "start_mm":               0,
            "length_mm":              length_mm,
            "num_results":            num_samples,
            "timestamp_ms":           int(t * 1000),
            "vehicle_heading_deg":    float(ph.ShipGyro),
            "transducer_heading_deg": float(ph.SensorHeading) if float(ph.SensorHeading) else float(ph.ShipGyro),
            "min_pwr_db":             MIN_PWR_DB,
            "max_pwr_db":             MAX_PWR_DB,
            "samples_db":             samples_db,
            "mm_per_sample":          mm_per_sample,
        }

        lat = float(ph.SensorYcoordinate) or float(ph.ShipYcoordinate)
        lon = float(ph.SensorXcoordinate) or float(ph.ShipXcoordinate)

        yield ping, interval, lat, lon


# ---- probe -----------------------------------------------------------------

def probe(path):
    """Print file metadata without running detection."""
    file_header, packets = pyxtf.xtf_read(path)

    print(f"\nFile: {path}")
    print(f"Sonar type:  {file_header.SonarType}")
    print(f"Channels:    {file_header.NumberOfSonarChannels}")
    for i in range(file_header.NumberOfSonarChannels):
        ci = file_header.ChanInfo[i]
        print(f"  Channel {i}: {ci.NumSamples} samples  {ci.SlantRange:.1f} m range")

    if XTFHeaderType.sonar in packets:
        pings = packets[XTFHeaderType.sonar]
        print(f"Pings:       {len(pings)}")
        ph = pings[0].ping_header
        print(f"First ping:  lat={ph.ShipYcoordinate:.6f}  lon={ph.ShipXcoordinate:.6f}  hdg={ph.ShipGyro:.1f}°")
        ph = pings[-1].ping_header
        print(f"Last ping:   lat={ph.ShipYcoordinate:.6f}  lon={ph.ShipXcoordinate:.6f}  hdg={ph.ShipGyro:.1f}°")
    print()


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Replay a SonarView XTF file through the detector.")
    ap.add_argument("xtf", help="Path to .xtf file")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier (0 = no delay, default 1.0)")
    ap.add_argument("--probe", action="store_true",
                    help="Print file info and exit without running detection")
    args = ap.parse_args()

    if args.probe:
        probe(args.xtf)
        return

    vehicle = VehicleState()
    on_frame     = make_on_frame(vehicle)
    on_detection = make_on_detection(vehicle)

    detector = WaterfallDetector(
        max_rows=500,
        detect_every=50,
        display_every=5,
        on_detection=on_detection,
        on_frame=on_frame,
    )

    print(f"[replay] {args.xtf}")
    print(f"[replay] speed={args.speed}x  (Ctrl-C to stop)")

    try:
        for ping, interval, lat, lon in iter_pings(args.xtf):
            # Update vehicle state from XTF navigation data
            if lat and lon:
                vehicle.update_gps(lat=lat, lon=lon, alt_m=0)
            vehicle.update_heading(ping["vehicle_heading_deg"])

            detector.add_ping(ping)

            if args.speed > 0:
                time.sleep(interval / args.speed)

    except KeyboardInterrupt:
        print("\n[replay] stopped")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
