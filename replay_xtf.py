"""
replay_xtf.py
Replay a SonarView .xtf recording through the detection pipeline.

Feeds real ping data into the same WaterfallDetector used in production,
so detection results match what you'd see from the live boat.

SonarView records the OmniScan 450 in side-scan mode: two channels
(port = channel 0, starboard = channel 1), uint16 samples. We combine
them into a full swath image (port reversed | starboard) with nadir at
the centre column, which is the standard side-scan waterfall view.

Navigation comes from the Sensor* fields in each ping header:
    SensorYcoordinate = latitude
    SensorXcoordinate = longitude
    SensorHeading     = heading (deg)

Usage:
    python replay_xtf.py scan.xtf                 # real-time speed
    python replay_xtf.py scan.xtf --speed 4       # 4x faster
    python replay_xtf.py scan.xtf --speed 0       # no delay (max speed)
    python replay_xtf.py scan.xtf --channel 1     # single channel only
    python replay_xtf.py scan.xtf --probe         # print file info and exit
"""

import argparse
import itertools
import math
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

def _to_db(arr, sample_format):
    """Map raw XTF sample values to dB. Absolute scale is unimportant — the
    detector re-normalises per window — but we keep relative ordering."""
    arr = np.asarray(arr, dtype=np.float64)
    span = MAX_PWR_DB - MIN_PWR_DB
    if sample_format == XTFSampleFormat.word:       # uint16
        return MIN_PWR_DB + (arr / 65535.0) * span
    elif sample_format == XTFSampleFormat.byte:     # uint8
        return MIN_PWR_DB + (arr / 255.0) * span
    lo, hi = float(arr.min()), float(arr.max())
    norm = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return MIN_PWR_DB + norm * span


# ---- georeferencing (side-scan aware) --------------------------------------

def georeference(detection, ping, vehicle):
    """Side-scan georeferencing.

    For a combined swath the nadir is at ping['nadir_col']. A detection's
    horizontal offset from nadir gives its range; whether it's left or right
    of nadir decides which side of the track (heading ± 90°) it lies on.
    For a single channel nadir_col is 0, so range = x * mm_per_sample and the
    detection is on ping['side'].
    """
    fix = vehicle.fix
    if fix is None:
        return None

    nadir_col = ping.get("nadir_col", 0)
    offset    = detection["x"] - nadir_col          # +starboard, -port
    range_m   = abs(offset) * ping["mm_per_sample"] / 1000.0

    heading = ping["transducer_heading_deg"]
    if nadir_col > 0:
        # combined swath: side from offset sign
        side_bearing = heading + 90.0 if offset >= 0 else heading - 90.0
    else:
        # single channel: caller tells us which side
        side_bearing = heading + (90.0 if ping.get("side", "stbd") == "stbd" else -90.0)

    bearing = math.radians(side_bearing % 360.0)
    ref_lat, ref_lon = fix["lat"], fix["lon"]
    dlat = range_m * math.cos(bearing) / 111_111.0
    dlon = range_m * math.sin(bearing) / (111_111.0 * math.cos(math.radians(ref_lat)))
    return ref_lat + dlat, ref_lon + dlon


# ---- display ---------------------------------------------------------------

def _draw(objects, ping, image, vehicle):
    display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # nadir line
    nadir_col = ping.get("nadir_col", 0)
    if nadir_col > 0:
        cv2.line(display, (nadir_col, 0), (nadir_col, display.shape[0]),
                 (60, 60, 60), 1)

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


def make_on_frame(vehicle):
    return lambda objects, ping, image: _draw(objects, ping, image, vehicle)


def make_on_detection(vehicle):
    def on_detection(objects, ping, image):
        if image is not None:
            _draw(objects, ping, image, vehicle)

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
            nadir   = ping.get("nadir_col", 0)
            range_m = abs(obj["x"] - nadir) * ping["mm_per_sample"] / 1000.0
            latlon  = georeference(obj, ping, vehicle)
            loc     = f"({latlon[0]:.6f},{latlon[1]:.6f})" if latlon else "no-fix"
            print(
                f"    range~{range_m:.1f}m  size={obj['w']}x{obj['h']}px  "
                f"src={obj.get('source','?')}  latlon={loc}"
            )
    return on_detection


# ---- XTF iteration ---------------------------------------------------------

def _seconds_of_day(p):
    return (float(p.Hour) * 3600 + float(p.Minute) * 60
            + float(p.Second) + float(p.HSeconds) / 100.0)


def iter_pings(path, channel="both"):
    """Yield (ping_dict, interval_sec, lat, lon) for every sonar ping.

    channel: "both" (combined swath), 0 (port only), or 1 (starboard only).
    """
    fh, packets = pyxtf.xtf_read(path)
    if XTFHeaderType.sonar not in packets:
        raise ValueError("No sonar channel found in XTF file.")
    sonar = packets[XTFHeaderType.sonar]

    sample_fmt = XTFSampleFormat(fh.ChanInfo[0].SampleFormat)
    prev_t = None

    for i, p in enumerate(sonar):
        chans   = p.ping_chan_headers
        n_chan  = len(p.data)

        # slant range from channel header (metres)
        slant_m = float(chans[0].SlantRange) or 25.0

        if channel == "both" and n_chan >= 2:
            port = _to_db(p.data[0], sample_fmt)
            stbd = _to_db(p.data[1], sample_fmt)
            samples_db = np.concatenate([port[::-1], stbd])
            num        = len(samples_db)
            nadir_col  = len(port)
            mm_per_sample = (slant_m * 1000.0) / len(port)
            side = None
        else:
            idx = int(channel) if channel != "both" else 0
            idx = min(idx, n_chan - 1)
            samples_db    = _to_db(p.data[idx], sample_fmt)
            num           = len(samples_db)
            nadir_col     = 0
            mm_per_sample = (slant_m * 1000.0) / num
            side          = "stbd" if idx == 1 else "port"

        t        = _seconds_of_day(p)
        interval = (t - prev_t) if (prev_t is not None and t > prev_t) else 0.05
        interval = min(interval, 1.0)
        prev_t   = t

        heading = float(p.SensorHeading) or float(p.ShipGyro)

        ping = {
            "type":                   "ping",
            "ping_number":            int(p.PingNumber) if p.PingNumber else i,
            "start_mm":               0,
            "length_mm":              slant_m * 1000.0,
            "num_results":            num,
            "timestamp_ms":           int(t * 1000),
            "vehicle_heading_deg":    heading,
            "transducer_heading_deg": heading,
            "min_pwr_db":             MIN_PWR_DB,
            "max_pwr_db":             MAX_PWR_DB,
            "samples_db":             samples_db.tolist(),
            "mm_per_sample":          mm_per_sample,
            "nadir_col":              nadir_col,
        }
        if side:
            ping["side"] = side

        lat = float(p.SensorYcoordinate) or float(p.ShipYcoordinate)
        lon = float(p.SensorXcoordinate) or float(p.ShipXcoordinate)
        yield ping, interval, lat, lon


# ---- probe -----------------------------------------------------------------

def probe(path):
    fh, packets = pyxtf.xtf_read(path)
    print(f"\nFile: {path}")
    print(f"Sonar type:  {fh.SonarType}")
    print(f"Channels:    {fh.NumberOfSonarChannels}")
    for i in range(max(1, fh.NumberOfSonarChannels)):
        ci = fh.ChanInfo[i]
        print(f"  Channel {i}: fmt={XTFSampleFormat(ci.SampleFormat).name}  "
              f"bytes/sample={ci.BytesPerSample}  type={ci.TypeOfChannel}")

    if XTFHeaderType.sonar in packets:
        sonar = packets[XTFHeaderType.sonar]
        print(f"Pings:       {len(sonar)}")
        ch0 = sonar[0].ping_chan_headers[0]
        print(f"Samples/ping: {ch0.NumSamples}   Slant range: {ch0.SlantRange:.1f} m")
        for label, p in (("First", sonar[0]), ("Last", sonar[-1])):
            print(f"{label} ping: lat={p.SensorYcoordinate:.6f}  "
                  f"lon={p.SensorXcoordinate:.6f}  hdg={p.SensorHeading:.1f}°  "
                  f"time={p.Hour:02d}:{p.Minute:02d}:{p.Second:02d}")
        lats = [pp.SensorYcoordinate for pp in sonar if pp.SensorYcoordinate]
        lons = [pp.SensorXcoordinate for pp in sonar if pp.SensorXcoordinate]
        if lats:
            print(f"Lat range:   {min(lats):.6f} .. {max(lats):.6f}")
            print(f"Lon range:   {min(lons):.6f} .. {max(lons):.6f}")
        dur = _seconds_of_day(sonar[-1]) - _seconds_of_day(sonar[0])
        print(f"Duration:    {dur:.0f} s  (~{len(sonar)/dur:.1f} pings/s)" if dur > 0 else "")
    print()


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Replay a SonarView XTF file through the detector.")
    ap.add_argument("xtf", help="Path to .xtf file")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier (0 = no delay, default 1.0)")
    ap.add_argument("--channel", default="both", choices=["both", "0", "1"],
                    help="Channel(s) to process (default: both = combined swath)")
    ap.add_argument("--probe", action="store_true",
                    help="Print file info and exit")
    args = ap.parse_args()

    if args.probe:
        probe(args.xtf)
        return

    vehicle = VehicleState()
    stream  = iter_pings(args.xtf, args.channel)

    # Peek the first ping so we can size the nadir mask to this file. For the
    # combined swath the nadir/water-column gap sits at nadir_col; suppress it
    # so it doesn't generate false detections.
    try:
        first = next(stream)
    except StopIteration:
        print("[replay] no pings in file")
        return
    nadir_col = first[0].get("nadir_col", 0)
    mask_band = (nadir_col, 70) if nadir_col > 0 else None

    detector = WaterfallDetector(
        max_rows=500,
        detect_every=50,
        display_every=5,
        mask_band=mask_band,
        on_detection=make_on_detection(vehicle),
        on_frame=make_on_frame(vehicle),
    )

    print(f"[replay] {args.xtf}")
    print(f"[replay] channel={args.channel}  speed={args.speed}x  "
          f"mask_band={mask_band}  (Ctrl-C to stop)")

    try:
        for ping, interval, lat, lon in itertools.chain([first], stream):
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
