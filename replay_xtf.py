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
import os
import time

import cv2
import numpy as np
import pyxtf
from pyxtf import XTFHeaderType, XTFSampleFormat

from mavlink_client import VehicleState
from sonar_detect import WaterfallDetector
from sonar_dashboard import SonarDashboard
from detection_log import DetectionLog

MIN_PWR_DB = -90.0
MAX_PWR_DB = -40.0
_WINDOW    = "OmniScan 450 - XTF Replay"
# Display palette: "blue" (default), "amber" (SonarView-style), or "gray". Cosmetic only.
PALETTE    = os.environ.get("SONAR_PALETTE", "amber")


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
# The dashboard (sonar_dashboard.py) composes the whole window; the detection
# log (detection_log.py) collects de-duplicated contacts that drive both the
# sidebar list and the end-of-mission hand-off. Detection is
# unchanged — these only consume what the pipeline already produces.

def _show(dashboard, log, objects, ping, image, vehicle):
    frame = dashboard.render(image, objects, ping, vehicle, log)
    cv2.imshow(dashboard.window, frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        raise KeyboardInterrupt


def make_on_frame(dashboard, log, vehicle):
    return lambda objects, ping, image: _show(
        dashboard, log, objects, ping, image, vehicle)


def make_on_detection(dashboard, log, vehicle):
    def on_detection(objects, ping, image):
        # annotate each detection with range + contact id and fold it into the
        # de-duplicated contact log before drawing
        nadir = ping.get("nadir_col", 0)
        for obj in objects:
            range_m = abs(obj["x"] - nadir) * ping["mm_per_sample"] / 1000.0
            obj["range_m"] = range_m
            latlon = georeference(obj, ping, vehicle)
            if latlon:
                c = log.add(latlon[0], latlon[1], range_m=range_m,
                            size=(obj["w"], obj["h"]),
                            source=obj.get("source", "cfar"),
                            score=obj.get("score", 0.0),
                            ping_number=ping["ping_number"])
                obj["cid"] = c["id"]

        if image is not None:
            _show(dashboard, log, objects, ping, image, vehicle)

        if not objects:
            print(f"[detect] ping #{ping['ping_number']}: clear")
            return

        fix     = vehicle.fix
        gps_tag = (
            f"  gps=({fix['lat']:.6f},{fix['lon']:.6f})  hdg={fix['heading_deg']:.1f}deg"
            if fix else "  gps=no-fix"
        )
        print(f"[detect] ping #{ping['ping_number']}: {len(objects)} object(s){gps_tag}")
        for obj in objects:
            latlon = georeference(obj, ping, vehicle)
            loc    = f"({latlon[0]:.6f},{latlon[1]:.6f})" if latlon else "no-fix"
            print(
                f"    range~{obj.get('range_m', 0):.1f}m  size={obj['w']}x{obj['h']}px  "
                f"src={obj.get('source','?')}  contact=#{obj.get('cid','-')}  latlon={loc}"
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
    ap.add_argument("--no-detect", action="store_true",
                    help="Turn detection off: show clean imagery only, no markers, no contacts")
    ap.add_argument("--detector", default="cfar",
                    choices=["both", "cfar", "classical", "roi", "blob", "blob_cfar"],
                    help="Detector to visualize (default: cfar)")
    ap.add_argument("--cfar-k", type=float, default=None, dest="cfar_k")
    ap.add_argument("--cfar-min-abs-db", type=float, default=None, dest="cfar_min_abs_db",
                    help="Absolute-brightness gate: drop CFAR boxes whose mean "
                         "dB-excess over the window global median is below this.")
    ap.add_argument("--cfar-min-contrast", type=float, default=None, dest="cfar_min_contrast",
                    help="Drop CFAR boxes whose local contrast score is below this.")
    ap.add_argument("--cfar-size-before-close", action="store_true",
                    dest="cfar_size_before_close",
                    help="Size-gate raw components before the morphological close.")
    ap.add_argument("--cfar-train-y", type=int, default=None, dest="cfar_train_y",
                    help="Along-track train half-height (>guard-y for a 2-D ring), e.g. 55")
    ap.add_argument("--cfar-guard-y", type=int, default=None, dest="cfar_guard_y",
                    help="Along-track guard half-height, e.g. 25")
    ap.add_argument("--min-area", type=int, default=None, dest="min_area")
    ap.add_argument("--nadir-guard", type=int, default=None, dest="nadir_guard")
    ap.add_argument("--edge-guard", type=int, default=None, dest="edge_guard")
    ap.add_argument("--dbscan-eps", type=float, default=None, dest="dbscan_eps",
                    help="DBSCAN neighbour radius in px (roi + cfar-confirm, default 20)")
    ap.add_argument("--dbscan-min", type=int, default=None, dest="dbscan_min",
                    help="min keypoints to form a cluster (roi + cfar-confirm, default 5)")
    ap.add_argument("--cfar-confirm", action="store_true", dest="cfar_confirm",
                    help="Keep a CFAR box only if its centre sits in a dense DBSCAN "
                         "feature cluster (precision gate). Use with --detector cfar.")
    ap.add_argument("--confirm-min-feat", type=int, default=None, dest="confirm_min_feat",
                    help="Min features a confirming cluster must hold (default 3*dbscan-min). "
                         "Raise to reject diffuse seabed speckle.")
    ap.add_argument("--kp-cap", type=int, default=None, dest="kp_cap",
                    help="Max features in the cloud fed to DBSCAN (default 100).")
    ap.add_argument("--fast-threshold", type=int, default=None, dest="fast_threshold")
    # ---- blob / contour shape detector (--detector blob) ----
    ap.add_argument("--detect-gamma", type=float, default=None, dest="detect_gamma",
                    help="Global gamma on the detection image (>1 suppresses small "
                         "bright speckle; replaces CLAHE). Recommended for blob mode, e.g. 1.8")
    ap.add_argument("--blob-pct", type=float, default=None, dest="blob_pct",
                    help="Keep pixels above this percentile of the feature image (default 98)")
    ap.add_argument("--blob-min-area", type=int, default=None, dest="blob_min_area")
    ap.add_argument("--blob-max-area", type=int, default=None, dest="blob_max_area")
    ap.add_argument("--blob-min-aspect", type=float, default=None, dest="blob_min_aspect")
    ap.add_argument("--blob-max-aspect", type=float, default=None, dest="blob_max_aspect")
    ap.add_argument("--blob-min-solidity", type=float, default=None, dest="blob_min_solidity")
    ap.add_argument("--blob-min-extent", type=float, default=None, dest="blob_min_extent")
    ap.add_argument("--blob-min-contrast", type=float, default=None, dest="blob_min_contrast")
    # ---- display tone (cosmetic only — does NOT affect detection) ----
    ap.add_argument("--contrast", type=float, default=1.12,
                    help="Display contrast gain around mid-grey (1.0=none, "
                         ">1 harder, <1 softer). Cosmetic only. Default 1.12")
    ap.add_argument("--gamma", type=float, default=0.85,
                    help="Display gamma (<1 lifts faint seabed). Default 0.85")
    ap.add_argument("--brightness", type=float, default=0.0,
                    help="Display brightness shift in [-1,1]. Default 0")
    ap.add_argument("--coords", metavar="PATH", default="contacts.py",
                    help="At the end of the run, write the contacts as an "
                         "importable Python array (coords = [(lat, lon, 0), ...]) "
                         "to this file. Default: contacts.py. Use --no-coords to skip.")
    ap.add_argument("--no-coords", dest="coords", action="store_const", const=None,
                    help="Do not write the coords file.")
    ap.add_argument("--coords-largest", type=int, default=None, dest="coords_largest",
                    help="With --coords, keep only the N largest contacts by "
                         "detection area (e.g. 50). Biggest shapes first.")
    ap.add_argument("--coords-min-hits", type=int, default=1, dest="coords_min_hits",
                    help="With --coords, drop contacts seen on fewer than N pings "
                         "(default 1; use 2 to drop one-ping flickers).")
    ap.add_argument("--populate", metavar="PATH", default=None,
                    help="Fill the hard-coded waypoint array in PATH (e.g. the "
                         "mission controller's mavlink.py) with this run's "
                         "contacts as (lat, lon) pairs, leaving the rest of that "
                         "file untouched (a .bak backup is written first). "
                         "Respects --coords-largest / --coords-min-hits.")
    ap.add_argument("--populate-var", default="WAYPOINTS", dest="populate_var",
                    help="Name of the array to fill in --populate's file "
                         "(default: WAYPOINTS).")
    ap.add_argument("--run-after", metavar="CMD", default=None, dest="run_after",
                    help="Shell command to run once the scan finishes and the "
                         "waypoints are written, e.g. "
                         "--run-after 'python mavlink.py' to launch the mission "
                         "controller automatically after the scan.")
    args = ap.parse_args()

    if args.probe:
        probe(args.xtf)
        return

    vehicle = VehicleState()
    log     = DetectionLog(merge_radius_m=3.0)
    mode    = f"REPLAY {args.speed:g}x" if args.speed > 0 else "REPLAY MAX"
    dashboard = SonarDashboard(title=_WINDOW, palette=PALETTE,
                               source_label=os.path.basename(args.xtf), mode=mode,
                               contrast=args.contrast, gamma=args.gamma,
                               brightness=args.brightness)
    stream  = iter_pings(args.xtf, args.channel)

    # Peek the first ping so we can size the nadir mask to this file. For the
    # combined swath the nadir/water-column gap sits at nadir_col; suppress it
    # so it doesn't generate false detections.
    try:
        first = next(stream)
    except StopIteration:
        print("[replay] no pings in file")
        return
    # The nadir / water-column stripe down the middle of the combined swath is
    # a bright fuzzy zone with no real seabed. Blank it on the *display* image
    # (mask_band) and exclude it from detection (nadir_guard) using the SAME
    # half-width, so --nadir-guard widens the ignored zone everywhere at once.
    nadir_col = first[0].get("nadir_col", 0)
    nadir_half = args.nadir_guard if args.nadir_guard is not None else 70
    mask_band = (nadir_col, nadir_half) if nadir_col > 0 else None

    det_kwargs = {}
    if args.cfar_k is not None:        det_kwargs["cfar_k"] = args.cfar_k
    if args.cfar_min_abs_db is not None:   det_kwargs["cfar_min_abs_db"] = args.cfar_min_abs_db
    if args.cfar_min_contrast is not None: det_kwargs["cfar_min_contrast"] = args.cfar_min_contrast
    if args.cfar_size_before_close:        det_kwargs["cfar_size_before_close"] = True
    if args.cfar_train_y is not None:  det_kwargs["cfar_train_y"] = args.cfar_train_y
    if args.cfar_guard_y is not None:  det_kwargs["cfar_guard_y"] = args.cfar_guard_y
    if args.min_area is not None:      det_kwargs["cfar_min_area"] = args.min_area
    det_kwargs["nadir_guard"] = nadir_half   # keep CFAR guard == display mask
    if args.edge_guard is not None:    det_kwargs["edge_guard"] = args.edge_guard
    if args.dbscan_eps is not None:    det_kwargs["dbscan_epsilon"] = args.dbscan_eps
    if args.dbscan_min is not None:    det_kwargs["dbscan_min_points"] = args.dbscan_min
    if args.cfar_confirm:              det_kwargs["cfar_confirm"] = True
    if args.confirm_min_feat is not None: det_kwargs["cfar_confirm_min_feat"] = args.confirm_min_feat
    if args.kp_cap is not None:        det_kwargs["kp_cap"] = args.kp_cap
    if args.fast_threshold is not None: det_kwargs["fast_threshold"] = args.fast_threshold
    if args.detect_gamma is not None:     det_kwargs["detect_gamma"] = args.detect_gamma
    if args.blob_pct is not None:         det_kwargs["blob_pct"] = args.blob_pct
    if args.blob_min_area is not None:    det_kwargs["blob_min_area"] = args.blob_min_area
    if args.blob_max_area is not None:    det_kwargs["blob_max_area"] = args.blob_max_area
    if args.blob_min_aspect is not None:  det_kwargs["blob_min_aspect"] = args.blob_min_aspect
    if args.blob_max_aspect is not None:  det_kwargs["blob_max_aspect"] = args.blob_max_aspect
    if args.blob_min_solidity is not None: det_kwargs["blob_min_solidity"] = args.blob_min_solidity
    if args.blob_min_extent is not None:  det_kwargs["blob_min_extent"] = args.blob_min_extent
    if args.blob_min_contrast is not None: det_kwargs["blob_min_contrast"] = args.blob_min_contrast

    detector = WaterfallDetector(
        max_rows=500,
        detect_every=50,
        display_every=5,
        detector="off" if args.no_detect else args.detector,
        mask_band=mask_band,
        on_detection=make_on_detection(dashboard, log, vehicle),
        on_frame=make_on_frame(dashboard, log, vehicle),
        **det_kwargs,
    )

    print(f"[replay] {args.xtf}")
    print(f"[replay] channel={args.channel}  speed={args.speed}x  "
          f"detector={'off' if args.no_detect else args.detector}  "
          f"mask_band={mask_band}  params={det_kwargs}  (Ctrl-C to stop)")

    interrupted = False
    try:
        for ping, interval, lat, lon in itertools.chain([first], stream):
            if lat and lon:
                vehicle.update_gps(lat=lat, lon=lon, alt_m=0)
                log.add_fix(lat, lon)
            vehicle.update_heading(ping["vehicle_heading_deg"])
            detector.add_ping(ping)
            if args.speed > 0:
                time.sleep(interval / args.speed)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[replay] stopped")
    finally:
        _summarize(log)
        if args.coords:
            from export_coords import export_from_log
            path, n = export_from_log(log, path=args.coords,
                                      largest=args.coords_largest,
                                      min_hits=args.coords_min_hits)
            mod = os.path.splitext(os.path.basename(path))[0]
            print(f"[replay] wrote {n} coord(s) to {path} "
                  f"(import with: from {mod} import coords)")
        if args.populate:
            from export_coords import populate_coords_in_file
            coords = log.to_coords(largest=args.coords_largest,
                                   min_hits=args.coords_min_hits)
            path, n, mode = populate_coords_in_file(
                coords, args.populate, var=args.populate_var, dims=2)
            if mode == "in-place":
                print(f"[replay] populated {n} point(s) into {args.populate_var} "
                      f"in {path}; original backed up to {path}.bak")
            else:
                print(f"[replay] wrote {n} point(s) to {path} ({mode})")
        cv2.destroyAllWindows()
        if args.run_after and not interrupted:
            import subprocess
            print(f"[replay] running: {args.run_after}")
            subprocess.run(args.run_after, shell=True)
        elif args.run_after and interrupted:
            print("[replay] scan interrupted — skipping --run-after "
                  f"({args.run_after!r}) so no mission launches on an aborted run.")


def _summarize(log):
    """Print the de-duplicated contact list at the end of a run. This list is
    what you hand off to another program."""
    print(f"\n[replay] mission complete: {len(log)} unique contact(s)")
    for r in log.to_records():
        print(f"    #{r['id']:02d}  ({r['lat']:.6f}, {r['lon']:.6f})  "
              f"range~{r['range_m']}m  {r['source']}  seen {r['hits']}x")
    if len(log):
        print("[replay] hand off log.to_records() / to_geojson() / to_csv() "
              "to your revisit planner")


if __name__ == "__main__":
    main()
