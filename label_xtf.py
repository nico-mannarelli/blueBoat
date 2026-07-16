"""
label_xtf.py
Hand-label true contacts in an XTF scan to produce ground-truth `.hits` files
for tuning and scoring the detector.

Why label
---------
The detector's parameters (cfar_k, cfar_min_area, fast_threshold, shadow_db, …)
are currently hand-picked. With a handful of labeled scans you can measure
precision/recall and tune them objectively instead of by eye — and the same
labels later train a classifier. This tool produces those labels.

What a label is
---------------
One label = the position of a real contact, in the SAME image coordinates the
detector reports, so scoring is a direct comparison:

    channel  x  y  w  h

    x   = sample column in the (combined-swath) waterfall — range across track
    y   = ABSOLUTE ping number. The live display scrolls, so a display row is
          meaningless later; the ping number is stable forever.
    w,h = box size in px (across-track samples × along-track rows). 0,0 = a
          clicked point / centre.
    channel = 0 (port, left of nadir) or 1 (starboard); single channel = 0.

This is OpenSidescan's `.hits` convention (`channel x y`) plus an optional box;
a plain 3-column `channel x y` file also loads. For strict per-channel
OpenSidescan interop convert the column with abs(x - nadir_col).

The labeler shows the whole scan one page at a time (rather than the live
scrolling view) so you can mark every contact — completeness is what makes the
recall number meaningful.

Usage
-----
    python label_xtf.py scan.xtf                 # combined swath -> scan.hits
    python label_xtf.py scan.xtf --channel 1     # single channel
    python label_xtf.py scan.xtf --out my.hits   # explicit output
    python label_xtf.py scan.xtf --width 1400    # display width (px)
    python label_xtf.py scan.xtf --stride 2      # label every 2nd ping (huge scans)

An existing output file is loaded on start, so you can resume.

Controls
--------
    left-click            drop a point label
    left-drag             draw a box label
    right-click           delete the nearest label on this page
    n / SPACE             next page          p     previous page
    u                     undo last label    s     save
    q / ESC               save and quit

Classifier-crop mode (--crops DIR)
----------------------------------
    python label_xtf.py scan.xtf --crops sorted_crops
After drawing a box (or clicking a point), press its class key to ALSO save a
224x224 contrast-stretched crop into DIR/<class>/ — see CROP_CLASSES below
(1=log 2=rock 3=man_made 4=background 5=debris 6=pipe_cable 7=tire 8=wreck
9=fish). The .hits labels are still written as usual, so one labeling session
feeds both the detector scoring AND the classifier library:
    python sonar_classifier.py build --crops sorted_crops --lib library.npz
"""

import argparse
import math
import os

import numpy as np
import cv2

from sonar_detect import WaterfallDetector
from sonar_display import colorize

PALETTE = os.environ.get("SONAR_PALETTE", "blue")
MARK = (0, 230, 255)   # BGR amber for label markers


# ---- data prep -------------------------------------------------------------

def collect_pings(path, channel="both"):
    """Read every ping from an XTF file as the same ping dicts the detector
    consumes. Imported lazily so this module loads without pyxtf installed."""
    from replay_xtf import iter_pings
    return [ping for ping, _interval, _lat, _lon in iter_pings(path, channel)]


def waterfall_from_pings(pings):
    """Stack a list of ping dicts into the display waterfall the detector sees.

    Returns (gray_uint8 HxW, ping_numbers[H], nadir_col). Row i of the image is
    pings[i], whose absolute ping number is ping_numbers[i] — that mapping is
    what lets a click become a stable (sample, ping-number) label."""
    rows = [np.asarray(p["samples_db"], dtype=np.float64) for p in pings]
    gray = WaterfallDetector(detector="off")._build_display_image(rows)
    ping_numbers = [int(p["ping_number"]) for p in pings]
    nadir_col = int(pings[0].get("nadir_col", 0))
    return gray, ping_numbers, nadir_col


# ---- pure coordinate / serialisation helpers (unit-tested) -----------------

def pixel_to_full(px, py, page_top_row, scale):
    """Display-pixel on the current page -> full-res image (col, row)."""
    col = int(round(px / scale))
    row = page_top_row + int(round(py / scale))
    return col, row


def make_label(col0, row0, col1, row1, ping_numbers, nadir_col):
    """Build a label dict from two full-res corners. Stores the CENTRE as (x,y):
    x = sample column, y = absolute ping number of the centre row; w,h = box px
    (0,0 for a point). channel from which side of nadir the centre falls on."""
    x0, x1 = sorted((col0, col1))
    r0, r1 = sorted((row0, row1))
    w, h = x1 - x0, r1 - r0
    cx = x0 + w // 2
    crow = max(0, min(len(ping_numbers) - 1, r0 + h // 2))
    ch = 0 if (nadir_col <= 0 or cx < nadir_col) else 1
    return {"channel": ch, "x": int(cx), "y": int(ping_numbers[crow]),
            "w": int(w), "h": int(h)}


def ping_row_index(ping_numbers):
    """ping number -> row index, for drawing stored labels back onto a page."""
    return {pn: i for i, pn in enumerate(ping_numbers)}


def labels_to_hits(labels):
    lines = ["# channel x y w h   (x=sample col, y=ping number; w=h=0 => point)"]
    for L in labels:
        lines.append(f"{L['channel']} {L['x']} {L['y']} {L['w']} {L['h']}")
    return "\n".join(lines) + "\n"


def parse_hits(text):
    """Tolerant reader: skips blanks/comments, accepts `channel x y` (3 col,
    OpenSidescan) or `channel x y w h` (5 col, ours)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 3:
            continue
        ch, x, y = int(p[0]), int(float(p[1])), int(float(p[2]))
        w = int(float(p[3])) if len(p) > 3 else 0
        h = int(float(p[4])) if len(p) > 4 else 0
        out.append({"channel": ch, "x": x, "y": y, "w": w, "h": h})
    return out


def load_hits(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return parse_hits(f.read())
    return []


def save_hits(path, labels):
    with open(path, "w") as f:
        f.write(labels_to_hits(labels))


# ---- interactive labeler (GUI) ---------------------------------------------

# Classifier classes for --crops mode: press the key after drawing a box to
# save that crop into crops_dir/<class>/. Classes are just folder names —
# add/rename freely; the classifier builds its library from whatever folders
# exist. Thin classes are fine (a few examples still make a centroid, and
# extra classes look good in demo/video output even before they're well fed).
CROP_CLASSES = {
    "1": "log",
    "2": "rock",
    "3": "man_made",
    "4": "background",
    "5": "debris",
    "6": "pipe_cable",
    "7": "tire",
    "8": "wreck",
    "9": "fish",
}
POINT_BOX_PX = 60   # box size used when a click (not a drag) gets a class key


def run_labeler(gray, ping_numbers, nadir_col, out_path,
                palette=PALETTE, disp_width=1280, disp_height=900,
                crops_dir=None):
    color = colorize(gray, palette=palette, scale=1)   # BGR, full-res
    H, W = color.shape[:2]
    scale = disp_width / W
    rows_per_page = max(1, int(disp_height / scale))
    pages = max(1, math.ceil(H / rows_per_page))
    labels = load_hits(out_path)
    pn_row = ping_row_index(ping_numbers)
    state = {"page": 0, "drag": None, "crops_saved": 0}

    def top_row():
        return state["page"] * rows_per_page

    def render():
        top = top_row()
        bot = min(H, top + rows_per_page)
        crop = color[top:bot]
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        disp = cv2.resize(crop, (int(round(crop.shape[1] * scale)),
                                 int(round(crop.shape[0] * scale))),
                          interpolation=interp)
        if nadir_col > 0:
            nx = int(nadir_col * scale)
            cv2.line(disp, (nx, 0), (nx, disp.shape[0]), (70, 70, 70), 1)
        for i, L in enumerate(labels):
            row = pn_row.get(L["y"])
            if row is None or not (top <= row < bot):
                continue
            dx, dy = int(L["x"] * scale), int((row - top) * scale)
            if L["w"] > 0 or L["h"] > 0:
                w, h = int(L["w"] * scale), int(L["h"] * scale)
                cv2.rectangle(disp, (dx - w // 2, dy - h // 2),
                              (dx + w // 2, dy + h // 2), MARK, 1)
            cv2.drawMarker(disp, (dx, dy), MARK, cv2.MARKER_CROSS, 12, 1)
            cv2.putText(disp, str(i + 1), (dx + 6, dy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, MARK, 1, cv2.LINE_AA)
        if state["drag"]:
            (sx, sy), (ex, ey) = state["drag"]
            cv2.rectangle(disp, (sx, sy), (ex, ey), (0, 180, 255), 1)
        bar = (f"page {state['page'] + 1}/{pages}  labels {len(labels)}  "
               f"{os.path.basename(out_path)}   "
               f"[click/drag add | right-click del | n/p page | u undo | s save | q quit]")
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(disp, bar, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if crops_dir:
            menu = "  ".join(f"{k}={v}" for k, v in CROP_CLASSES.items())
            bar2 = (f"crops {state['crops_saved']} -> {crops_dir}   "
                    f"press key after box: {menu}")
            cv2.rectangle(disp, (0, 20), (disp.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(disp, bar2, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (120, 255, 120), 1, cv2.LINE_AA)
        return disp

    def save_class_crop(cls):
        """Crop the most recent label out of the full-res gray into
        crops_dir/<cls>/ (224x224, padded, contrast-stretched)."""
        if not labels:
            return
        from crop_saver import save_crop
        L = labels[-1]
        row = pn_row.get(L["y"])
        if row is None:
            return
        w = L["w"] or POINT_BOX_PX
        h = L["h"] or POINT_BOX_PX
        box = (L["x"] - w // 2, row - h // 2, w, h)
        meta = {"class": cls, "ping_number": L["y"], "channel": L["channel"],
                "xtf": os.path.basename(out_path)}
        p = save_crop(gray, box, os.path.join(crops_dir, cls),
                      state["crops_saved"] + 1, meta)
        if p:
            state["crops_saved"] += 1
            print(f"[label] crop #{state['crops_saved']} -> {p}")

    def on_mouse(event, x, y, flags, param):
        top = top_row()
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = [(x, y), (x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["drag"][1] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            (sx, sy), (ex, ey) = state["drag"]
            state["drag"] = None
            c0, r0 = pixel_to_full(sx, sy, top, scale)
            c1, r1 = pixel_to_full(ex, ey, top, scale)
            labels.append(make_label(c0, r0, c1, r1, ping_numbers, nadir_col))
        elif event == cv2.EVENT_RBUTTONDOWN:
            col, row = pixel_to_full(x, y, top, scale)
            best, bd = None, 1e9
            for i, L in enumerate(labels):
                lr = pn_row.get(L["y"])
                if lr is None:
                    continue
                d = math.hypot(L["x"] - col, lr - row)
                if d < bd:
                    bd, best = d, i
            if best is not None and bd < 40 / scale:
                labels.pop(best)

    win = "label - " + os.path.basename(out_path)
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            cv2.imshow(win, render())
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("q"), 27):
                save_hits(out_path, labels)
                break
            elif k in (ord("n"), ord(" ")):
                state["page"] = min(pages - 1, state["page"] + 1)
            elif k == ord("p"):
                state["page"] = max(0, state["page"] - 1)
            elif k == ord("u") and labels:
                labels.pop()
            elif k == ord("s"):
                save_hits(out_path, labels)
                print(f"[label] saved {len(labels)} -> {out_path}")
            elif crops_dir and chr(k) in CROP_CLASSES:
                save_class_crop(CROP_CLASSES[chr(k)])
    finally:
        cv2.destroyAllWindows()
    return labels


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Hand-label contacts in an XTF scan into a .hits file.")
    ap.add_argument("xtf", help="Path to .xtf file")
    ap.add_argument("--channel", default="both", choices=["both", "0", "1"],
                    help="Channel(s) to label (default: both = combined swath)")
    ap.add_argument("--out", default=None,
                    help="Output .hits path (default: <xtf>.hits)")
    ap.add_argument("--width", type=int, default=1280, help="Display width (px)")
    ap.add_argument("--height", type=int, default=900, help="Display height (px)")
    ap.add_argument("--stride", type=int, default=1,
                    help="Label every Nth ping — for very long scans (default 1)")
    ap.add_argument("--crops", default=None, metavar="DIR",
                    help="classifier-crop mode: after drawing a box, press a "
                         "class key (1-9) to save a 224x224 crop into "
                         "DIR/<class>/ for sonar_classifier.py build")
    args = ap.parse_args()

    out = args.out or (os.path.splitext(args.xtf)[0] + ".hits")
    pings = collect_pings(args.xtf, args.channel)
    if args.stride > 1:
        pings = pings[::args.stride]
    if not pings:
        print("[label] no pings in file")
        return

    gray, ping_numbers, nadir_col = waterfall_from_pings(pings)
    print(f"[label] {len(pings)} pings  image {gray.shape[1]}x{gray.shape[0]}  "
          f"nadir_col={nadir_col}  -> {out}")
    run_labeler(gray, ping_numbers, nadir_col, out,
                palette=PALETTE, disp_width=args.width, disp_height=args.height,
                crops_dir=args.crops)
    print(f"[label] done: {out}")


if __name__ == "__main__":
    main()
