"""
score_detector.py
Score the detector against a hand-labeled scan: run it over the XTF, match its
contacts to the `.hits` labels, and print precision / recall / F1.

This is the objective number that replaces tuning by eye. Change a detector
parameter (cfar_k, cfar_min_area, fast_threshold, …), re-run, and watch
precision/recall move — keep the settings that give the best F1 on your labels.

    python score_detector.py 2026-06-16-15-21.xtf
    python score_detector.py scan.xtf --hits scan.hits      # explicit labels
    python score_detector.py scan.xtf --cfar-k 7 --min-area 120   # try a setting
    python score_detector.py scan.xtf --detector cfar       # cfar only vs "both"
    python score_detector.py scan.xtf --limit 4000          # first N pings (quick)

How it works
------------
The detector runs streaming (exactly as in production), so every contact is
captured in absolute image coordinates: x = sample column, ping = absolute ping
number — the same space the labels live in. The same physical target fires on
many overlapping windows, so raw detections are de-duplicated into contacts
before matching. A contact matches a label if it lands within a tolerance of
the label's centre (the label's own box size widens that tolerance).

Definitions: a label with a matching contact is a true positive (drives recall);
a contact with no matching label is a false positive (drives precision).
Duplicate detections of the same real target are not punished as false
positives.
"""

import argparse
import os

from sonar_detect import WaterfallDetector


# ---- run the detector over a scan -----------------------------------------

def run_detector(pings, quiet=False, **det_kwargs):
    """Stream every ping through the detector and return raw detections as
    dicts {x (sample centre), ping (absolute #), w, h, source, score}."""
    dets = []
    order = []           # ping numbers in fed order; index = row in the window
    state = {"i": -1}

    def on_det(objects, ping, image):
        h_win = image.shape[0] if image is not None else 0
        cur = state["i"]                       # index of the latest fed ping
        for o in objects:
            cy = o["y"] + o["h"] / 2.0         # box centre row in the window
            # bottom window row (h_win-1) is the current ping; map up from there
            idx = int(round(cur - (h_win - 1 - cy)))
            idx = max(0, min(len(order) - 1, idx))
            dets.append({
                "x": int(o["x"] + o["w"] / 2), "ping": order[idx],
                "w": int(o["w"]), "h": int(o["h"]),
                "source": o.get("source", "?"), "score": float(o.get("score", 0.0)),
                "shadow": float(o.get("shadow", 0.0)),
            })

    det = WaterfallDetector(on_detection=on_det, on_frame=None, **det_kwargs)
    n = len(pings)
    for k, p in enumerate(pings):
        order.append(int(p["ping_number"]))
        state["i"] = len(order) - 1
        det.add_ping(p)
        if not quiet and n >= 2000 and k % 1000 == 0 and k:
            print(f"  ... {k}/{n} pings, {len(dets)} raw detections")
    return dets


def dedup(dets, dpx=30, dping=80):
    """Cluster raw detections of the same target (nearby in sample & ping) into
    one contact, keeping the strongest. Mirrors the live DetectionLog merge, but
    in image space so it lines up with the labels."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["score"])
    used = [False] * len(dets)
    out = []
    for i in order:
        if used[i]:
            continue
        used[i] = True
        cx, cp = dets[i]["x"], dets[i]["ping"]
        group = [dets[i]]
        for j in order:
            if not used[j] and abs(dets[j]["x"] - cx) <= dpx \
                    and abs(dets[j]["ping"] - cp) <= dping:
                used[j] = True
                group.append(dets[j])
        best = max(group, key=lambda g: g["score"])
        out.append({**best, "n": len(group)})
    return out


# ---- match contacts to labels ---------------------------------------------

def _match(label, contact, tol):
    """A contact matches a label if its centre is within tolerance of the
    label's centre — widened by half the label's own box so a big target counts
    a detection anywhere on it."""
    tolx = max(tol, label["w"] / 2 + 10)
    tolp = max(tol, label["h"] / 2 + 10)
    return (abs(contact["x"] - label["x"]) <= tolx
            and abs(contact["ping"] - label["y"]) <= tolp)


def score(labels, contacts, tol=40):
    lab_hit = [any(_match(L, C, tol) for C in contacts) for L in labels]
    con_tp = [any(_match(L, C, tol) for L in labels) for C in contacts]
    tp = sum(lab_hit)
    fn = len(labels) - tp
    fp = sum(1 for t in con_tp if not t)
    recall = tp / len(labels) if labels else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "n_labels": len(labels), "n_contacts": len(contacts),
        "missed": [labels[i] for i, h in enumerate(lab_hit) if not h],
        "false": [contacts[i] for i, t in enumerate(con_tp) if not t],
    }


# ---- shadow calibration ----------------------------------------------------

def calibrate_shadow(labels, contacts, tol=40):
    """Split contacts into true / false using the labels, then compare the
    shadow strength of each group. With labels in hand we can pick shadow_db
    empirically — the value that keeps the most real targets while cutting the
    most false positives — instead of guessing. Prints the two distributions
    and a threshold sweep with the F1 each cut would yield."""
    con_tp = [any(_match(L, C, tol) for L in labels) for C in contacts]
    tp_sh = sorted(c["shadow"] for c, t in zip(contacts, con_tp) if t)
    fp_sh = sorted(c["shadow"] for c, t in zip(contacts, con_tp) if not t)
    # Per-label best shadow: a label is recovered at threshold thr if ANY of its
    # matching contacts clears thr. Counting labels (not contacts) keeps recall <=1.
    lab_best = []
    for L in labels:
        ms = [C["shadow"] for C in contacts if _match(L, C, tol)]
        if ms:
            lab_best.append(max(ms))

    def pct(xs, q):
        if not xs:
            return 0.0
        i = max(0, min(len(xs) - 1, int(round(q / 100.0 * (len(xs) - 1)))))
        return xs[i]

    print(f"\nshadow calibration  (tol={tol})")
    print(f"  true contacts:  {len(tp_sh)}   "
          f"shadow min/med/max = {pct(tp_sh,0):.1f} / {pct(tp_sh,50):.1f} / "
          f"{pct(tp_sh,100):.1f}")
    print(f"  false contacts: {len(fp_sh)}   "
          f"shadow min/med/max = {pct(fp_sh,0):.1f} / {pct(fp_sh,50):.1f} / "
          f"{pct(fp_sh,100):.1f}")

    # A contact is kept when its shadow strength >= thr (hard-gate semantics).
    n_lab = len(labels)
    print(f"\n  {'shadow_db':>9}  {'TP kept':>7}  {'FP kept':>7}  "
          f"{'prec':>5}  {'recall':>6}  {'F1':>5}")
    print("  " + "-" * 50)
    best = None
    for thr in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        tp_c = sum(1 for s in tp_sh if s >= thr)      # true contacts kept
        fp = sum(1 for s in fp_sh if s >= thr)        # false contacts kept
        labs_kept = sum(1 for s in lab_best if s >= thr)  # distinct labels recovered
        prec = tp_c / (tp_c + fp) if (tp_c + fp) else 0.0
        rec = labs_kept / n_lab if n_lab else 0.0
        tp = labs_kept
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {thr:>9.1f}  {tp:>7}  {fp:>7}  {prec:>5.2f}  {rec:>6.2f}  {f1:>5.2f}")
        if best is None or f1 > best[1]:
            best = (thr, f1)
    print(f"\n[calib] best F1 {best[1]:.2f} at shadow_db={best[0]:g}  "
          f"(use: --shadow hard with shadow_db≈{best[0]:g})")


# ---- annotated image dump --------------------------------------------------

def dump_image(path, pings, labels, contacts, tol=40, max_width=1100):
    """Render the whole scan as a colour waterfall with truth/TP/FP overlaid so
    the false positives can be SEEN, not guessed at. Cyan = hand label (truth),
    green = true-positive contact, red = false-positive contact. Written to a PNG
    so it can be inspected outside the GUI."""
    import numpy as np
    import cv2
    from label_xtf import waterfall_from_pings
    from sonar_display import colorize

    gray, ping_numbers, nadir_col = waterfall_from_pings(pings)
    color = colorize(gray, palette="blue", scale=1)
    H, W = color.shape[:2]
    pn_row = {pn: i for i, pn in enumerate(ping_numbers)}

    con_tp = [any(_match(L, C, tol) for L in labels) for C in contacts]
    scale = min(1.0, max_width / W)
    disp = cv2.resize(color, (int(round(W * scale)), int(round(H * scale))),
                      interpolation=cv2.INTER_AREA)

    def pt(col, pingnum):
        row = pn_row.get(pingnum)
        if row is None:
            return None
        return int(col * scale), int(row * scale)

    # false positives first (red), then true positives (green) on top
    for C, t in zip(contacts, con_tp):
        if t:
            continue
        p = pt(C["x"], C["ping"])
        if p:
            cv2.circle(disp, p, 4, (0, 0, 255), 1, cv2.LINE_AA)
    for C, t in zip(contacts, con_tp):
        if not t:
            continue
        p = pt(C["x"], C["ping"])
        if p:
            cv2.circle(disp, p, 5, (0, 230, 0), 2, cv2.LINE_AA)
    # truth labels as cyan crosses, sized by their box
    for L in labels:
        p = pt(L["x"], L["y"])
        if p:
            cv2.drawMarker(disp, p, (255, 255, 0), cv2.MARKER_CROSS, 14, 1,
                           cv2.LINE_AA)
    if nadir_col > 0:
        nx = int(nadir_col * scale)
        cv2.line(disp, (nx, 0), (nx, disp.shape[0]), (90, 90, 90), 1)

    cv2.imwrite(path, disp)
    print(f"[dump] {path}  {disp.shape[1]}x{disp.shape[0]}  "
          f"(cyan=label  green=TP  red=FP)")


# ---- report ----------------------------------------------------------------

def report(r, contacts=None, width=None, nadir=None):
    from collections import Counter
    print(f"\nlabels: {r['n_labels']}   detector contacts: {r['n_contacts']}")
    print(f"  TP {r['tp']}   FP {r['fp']}   FN {r['fn']}")
    print(f"  precision {r['precision']:.2f}   recall {r['recall']:.2f}   "
          f"F1 {r['f1']:.2f}")

    # Where is the flood coming from? Break the false positives down by which
    # detector raised them and where they sit, so an over-firing run points
    # straight at the cause instead of a wall of boxes.
    if contacts is not None:
        by_src = Counter(c["source"] for c in contacts)
        print(f"  contacts by source: {dict(by_src)}")
    if r["false"]:
        fsrc = Counter(c["source"] for c in r["false"])
        print(f"  false positives by source: {dict(fsrc)}")
        if width:
            edge = sum(1 for c in r["false"]
                       if c["x"] < 120 or c["x"] > width - 120)
            print(f"  FPs in the outer-120px far-range band: {edge}/{len(r['false'])}"
                  + (f"   (raise edge_guard)" if edge > len(r["false"]) * 0.3 else ""))
        if nadir:
            nad = sum(1 for c in r["false"] if abs(c["x"] - nadir) < 120)
            print(f"  FPs within 120px of nadir: {nad}/{len(r['false'])}"
                  + (f"   (raise nadir_guard)" if nad > len(r["false"]) * 0.3 else ""))

    if r["missed"]:
        print(f"\nmissed labels ({len(r['missed'])})  — lower cfar_k / min_area "
              f"or check the band guards:")
        for L in r["missed"]:
            print(f"    ch{L['channel']}  x={L['x']:5d}  ping={L['y']:6d}  "
                  f"box={L['w']}x{L['h']}")
    if r["false"]:
        fa = sorted(r["false"], key=lambda c: -c["score"])
        print(f"\nfalse positives ({len(r['false'])})  — raise cfar_k / min_area "
              f"or enable shadow gating.  strongest:")
        for C in fa[:12]:
            print(f"    x={C['x']:5d}  ping={C['ping']:6d}  src={C['source']:5s}  "
                  f"score={C['score']:.1f}  seen {C.get('n', 1)}x")
        if len(fa) > 12:
            print(f"    ... and {len(fa) - 12} more")


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Score the detector against a labeled .hits scan.")
    ap.add_argument("xtf", help="Path to the labeled .xtf scan")
    ap.add_argument("--hits", default=None, help="Labels (default: <xtf>.hits)")
    ap.add_argument("--channel", default="both", choices=["both", "0", "1"],
                    help="Must match how you labeled (default: both)")
    ap.add_argument("--detector", default="both",
                    choices=["both", "cfar", "classical", "roi", "blob"])
    ap.add_argument("--cfar-k", type=float, default=None)
    ap.add_argument("--min-area", type=int, default=None, dest="min_area")
    ap.add_argument("--fast-threshold", type=int, default=None, dest="fast_threshold")
    ap.add_argument("--dbscan-eps", type=float, default=None, dest="dbscan_eps",
                    help="ROI mode: DBSCAN neighbour radius in px (default 20)")
    ap.add_argument("--dbscan-min", type=int, default=None, dest="dbscan_min",
                    help="ROI mode: min keypoints to form a cluster (default 5)")
    ap.add_argument("--cfar-confirm", action="store_true", dest="cfar_confirm",
                    help="Keep a CFAR box only if its centre sits in a dense DBSCAN "
                         "feature cluster (precision gate for --detector cfar).")
    ap.add_argument("--confirm-min-feat", type=int, default=None, dest="confirm_min_feat",
                    help="Min features a confirming cluster must hold (default 3*dbscan-min).")
    ap.add_argument("--kp-cap", type=int, default=None, dest="kp_cap",
                    help="Max features in the cloud fed to DBSCAN (default 100).")
    ap.add_argument("--cfar-train-y", type=int, default=None, dest="cfar_train_y",
                    help="Along-track train half-height. Set > guard-y for a 2-D "
                         "ring that suppresses ripple fields, e.g. 55")
    ap.add_argument("--cfar-guard-y", type=int, default=None, dest="cfar_guard_y",
                    help="Along-track guard half-height (must exceed a target's "
                         "along-track half-height), e.g. 25")
    ap.add_argument("--nadir-guard", type=int, default=None, dest="nadir_guard",
                    help="Half-width (px) of the nadir/near-range band to exclude")
    ap.add_argument("--edge-guard", type=int, default=None, dest="edge_guard",
                    help="Margin (px) trimmed off the far-range left/right edges")
    # ---- blob / contour shape detector (--detector blob) ----
    ap.add_argument("--detect-gamma", type=float, default=None, dest="detect_gamma",
                    help="Global gamma on the detection image (>1 suppresses small "
                         "bright speckle; replaces CLAHE). Recommended for blob mode, e.g. 1.8")
    ap.add_argument("--blob-pct", type=float, default=None, dest="blob_pct",
                    help="Keep pixels above this percentile of the feature image (default 98)")
    ap.add_argument("--blob-min-area", type=int, default=None, dest="blob_min_area",
                    help="Smallest blob in px^2 (default 60)")
    ap.add_argument("--blob-max-area", type=int, default=None, dest="blob_max_area",
                    help="Largest blob in px^2 (default 60000)")
    ap.add_argument("--blob-min-aspect", type=float, default=None, dest="blob_min_aspect",
                    help="Min w/h aspect (default 0.12)")
    ap.add_argument("--blob-max-aspect", type=float, default=None, dest="blob_max_aspect",
                    help="Max w/h aspect (default 8.0)")
    ap.add_argument("--blob-min-solidity", type=float, default=None, dest="blob_min_solidity",
                    help="Min blob_area/hull_area (default 0.45)")
    ap.add_argument("--blob-min-extent", type=float, default=None, dest="blob_min_extent",
                    help="Min blob_area/bbox_area (default 0.30)")
    ap.add_argument("--blob-min-contrast", type=float, default=None, dest="blob_min_contrast",
                    help="Min blob-minus-ring contrast in feature-image units (default 8)")
    ap.add_argument("--shadow", default=None, choices=["off", "soft", "hard"],
                    help="Shadow mode override")
    ap.add_argument("--dump", default=None,
                    help="Write an annotated PNG (truth/TP/FP overlaid on the "
                         "waterfall) to this path, then exit. e.g. --dump fp.png")
    ap.add_argument("--shadow-calib", action="store_true", dest="shadow_calib",
                    help="Fit shadow_db from the labels: run once, split contacts "
                         "into true/false, print the shadow distribution of each "
                         "and the threshold that best separates them.")
    ap.add_argument("--sweep-k", default=None, dest="sweep_k",
                    help="Comma list of cfar_k to try in one run, e.g. 3,4,5,6. "
                         "Prints a P/R/F1 table per k instead of a full report.")
    ap.add_argument("--tol", type=int, default=40,
                    help="Match tolerance in px (default 40)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only the first N pings (quick check)")
    args = ap.parse_args()

    from label_xtf import load_hits
    from replay_xtf import iter_pings

    hits_path = args.hits or (os.path.splitext(args.xtf)[0] + ".hits")
    labels = load_hits(hits_path)
    if not labels:
        print(f"[score] no labels in {hits_path}")
        return

    det_kwargs = {"detector": args.detector}
    if args.cfar_k is not None:
        det_kwargs["cfar_k"] = args.cfar_k
    if args.min_area is not None:
        det_kwargs["cfar_min_area"] = args.min_area
    if args.fast_threshold is not None:
        det_kwargs["fast_threshold"] = args.fast_threshold
    if args.cfar_train_y is not None:
        det_kwargs["cfar_train_y"] = args.cfar_train_y
    if args.cfar_guard_y is not None:
        det_kwargs["cfar_guard_y"] = args.cfar_guard_y
    if args.nadir_guard is not None:
        det_kwargs["nadir_guard"] = args.nadir_guard
    if args.edge_guard is not None:
        det_kwargs["edge_guard"] = args.edge_guard
    if args.dbscan_eps is not None:
        det_kwargs["dbscan_epsilon"] = args.dbscan_eps
    if args.dbscan_min is not None:
        det_kwargs["dbscan_min_points"] = args.dbscan_min
    if args.cfar_confirm:
        det_kwargs["cfar_confirm"] = True
    if args.confirm_min_feat is not None:
        det_kwargs["cfar_confirm_min_feat"] = args.confirm_min_feat
    if args.kp_cap is not None:
        det_kwargs["kp_cap"] = args.kp_cap
    if args.detect_gamma is not None:
        det_kwargs["detect_gamma"] = args.detect_gamma
    if args.blob_pct is not None:
        det_kwargs["blob_pct"] = args.blob_pct
    if args.blob_min_area is not None:
        det_kwargs["blob_min_area"] = args.blob_min_area
    if args.blob_max_area is not None:
        det_kwargs["blob_max_area"] = args.blob_max_area
    if args.blob_min_aspect is not None:
        det_kwargs["blob_min_aspect"] = args.blob_min_aspect
    if args.blob_max_aspect is not None:
        det_kwargs["blob_max_aspect"] = args.blob_max_aspect
    if args.blob_min_solidity is not None:
        det_kwargs["blob_min_solidity"] = args.blob_min_solidity
    if args.blob_min_extent is not None:
        det_kwargs["blob_min_extent"] = args.blob_min_extent
    if args.blob_min_contrast is not None:
        det_kwargs["blob_min_contrast"] = args.blob_min_contrast
    if args.shadow is not None:
        det_kwargs["shadow_mode"] = args.shadow

    print(f"[score] {args.xtf}")
    print(f"[score] labels={len(labels)} from {os.path.basename(hits_path)}  "
          f"detector={args.detector}  params={ {k: v for k, v in det_kwargs.items() if k != 'detector'} }")

    pings = [p for p, _i, _la, _lo in iter_pings(args.xtf, args.channel)]
    if args.limit:
        pings = pings[:args.limit]
    width = len(pings[0]["samples_db"]) if pings else None
    nadir = pings[0].get("nadir_col", 0) if pings else None
    print(f"[score] streaming {len(pings)} pings ...")

    # Render an annotated waterfall so the false positives can be inspected.
    if args.dump:
        dets = run_detector(pings, **det_kwargs)
        contacts = dedup(dets)
        dump_image(args.dump, pings, labels, contacts, tol=args.tol)
        return

    # Calibrate shadow_db against the labels. Force soft mode so every detection
    # is annotated with its shadow strength and NONE are dropped — we want the
    # full true/false split to fit the threshold from.
    if args.shadow_calib:
        ck = {**det_kwargs, "shadow_mode": "soft"}
        dets = run_detector(pings, **ck)
        contacts = dedup(dets)
        calibrate_shadow(labels, contacts, tol=args.tol)
        return

    # Sweep: run the detector once per cfar_k value over the same pings and
    # print a compact table. Lets you find the k that maximises F1 in one
    # command instead of re-running by hand (each pass is the slow part).
    if args.sweep_k:
        ks = [float(s) for s in args.sweep_k.split(",") if s.strip()]
        print(f"[score] sweeping cfar_k over {ks}\n")
        print(f"  {'k':>5}  {'contacts':>8}  {'TP':>3}  {'FP':>3}  {'FN':>3}  "
              f"{'prec':>5}  {'recall':>6}  {'F1':>5}")
        print("  " + "-" * 52)
        best = None
        for k in ks:
            dets = run_detector(pings, quiet=True, **{**det_kwargs, "cfar_k": k})
            contacts = dedup(dets)
            r = score(labels, contacts, tol=args.tol)
            print(f"  {k:>5.1f}  {r['n_contacts']:>8}  {r['tp']:>3}  {r['fp']:>3}  "
                  f"{r['fn']:>3}  {r['precision']:>5.2f}  {r['recall']:>6.2f}  "
                  f"{r['f1']:>5.2f}")
            if best is None or r["f1"] > best[1]:
                best = (k, r["f1"])
        print(f"\n[score] best F1 {best[1]:.2f} at cfar_k={best[0]:g}")
        return

    dets = run_detector(pings, **det_kwargs)
    contacts = dedup(dets)
    r = score(labels, contacts, tol=args.tol)
    report(r, contacts=contacts, width=width, nadir=nadir or None)


if __name__ == "__main__":
    main()
