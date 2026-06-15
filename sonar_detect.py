"""
sonar_detect.py
Builds a rolling waterfall image from pings and runs anomaly detection.

Two detector families are available, selected with `detector=`:

  "cfar"  (default) — CFAR + shadow gating
  ----------------------------------------
      A sliding-window adaptive-threshold detector. For every pixel it
      estimates the local seabed background from a band of *reference cells*
      (a guard band around the pixel is excluded so a target can't bias its
      own background), and flags the pixel when it rises more than k local
      standard deviations above that background:

          threshold(pixel) = local_mean + k · local_std

      This is the z-score / Gaussian form of Constant-False-Alarm-Rate
      detection. It needs no labels and no pre-recorded "normal seabed"
      baseline — the background is estimated from the data itself, per pixel,
      so it adapts to whatever bottom the boat is currently over. The window
      sums are computed in O(1) per pixel with an integral image (summed-area
      table), which is also why the GPU backend is trivially parallel.

      CFAR runs in the dB domain (the samples' native, log-compressed scale):
      each column is flat-fielded by *subtracting* its along-track median —
      TVG/range falloff is additive in dB — so seabed sits near 0 dB and a
      target is a positive dB excess. Working in dB rather than linear keeps a
      few hot clutter pixels from blowing up the local variance and pinning the
      threshold out of reach.

      The reference band is horizontal: background comes from RANGE-direction
      neighbours only. A target running along-track (a log at roughly constant
      range) always has clean seabed left and right whatever its length, so it
      fires along its whole extent instead of self-suppressing its interior.

      Shadow gating: a real object proud of the seabed throws an acoustic
      shadow (a dark patch) on its far-range side. Seabed texture does not.
      Each CFAR hit is kept only if a region several dB below background sits
      just beyond it in range — this rejects bright clutter with no labels.

  "classical" — Hough Circle + ROI (FAST → MSER)
  ----------------------------------------------
      The original two-stage CV detector. Kept for comparison; needs manual
      threshold tuning and has no model of the background.

  "both" runs both families and merges with NMS.

Waterfall image layout
----------------------
  Rows (y-axis)    = pings, newest at the bottom
  Columns (x-axis) = sample index = range from transducer

Detection dict keys
-------------------
  x, y   — top-left corner of bounding box (pixels)
  w, h   — width × height (pixels)
  source — "cfar" | "hough" | "roi"
  score  — peak CFAR z-score (cfar only)
"""

from collections import deque

import numpy as np
import cv2


class WaterfallDetector:
    def __init__(
        self,
        max_rows=500,
        detect_every=50,
        display_every=5,
        # which detector family to run: "cfar" | "classical" | "both"
        detector="cfar",
        # ---- CFAR params ----
        # The reference window is a horizontal band: background is estimated
        # from RANGE-direction (left/right) neighbours only, never along-track.
        # A target running along-track (e.g. a log at roughly constant range)
        # always has clean seabed to its left and right whatever its length, so
        # a horizontal-band reference fires along the whole target — a square
        # window would let the target fill its own along-track reference cells
        # and self-suppress its interior.
        cfar_guard_x=10,       # range guard half-width (px); must exceed the
                               #   target's range half-width or the target biases
                               #   its own background
        cfar_train_x=45,       # range reference half-width (px); ref = the wings
                               #   from guard_x..train_x on each side
        cfar_band_y=1,         # along-track half-height averaged for a stable
                               #   background estimate (rows = 2*band_y+1)
        cfar_k=4.5,            # threshold = local_mean + k * local_std (higher → less sensitive)
        cfar_min_area=60,      # minimum connected-component area (px) to keep
        cfar_close_ksize=7,    # morphological-close kernel to knit a target's pixels together
        cfar_merge_gap=12,     # merge CFAR boxes whose gap is below this (px); 0 disables
        cfar_backend="numpy",  # "numpy" (CPU) or "torch" (GPU on the edge server)
        # ---- shadow gating ----
        # A proud object (mine, rock) throws an acoustic shadow on its far-range
        # side; requiring one rejects bright clutter. OFF by default because a
        # flat-lying target (e.g. a waterlogged log half-buried in sediment)
        # casts no usable shadow and a hard gate would drop it. Enable for
        # proud-object searches. The threshold is in dB below local background.
        shadow_gate=False,
        shadow_db=5.0,         # shadow region must average this many dB below background
        shadow_min_len=6,      # minimum shadow probe length (px)
        # ---- classical (Hough) params ----
        hough_min_dist=30,
        hough_param1=50,       # Canny high threshold
        hough_param2=80,       # accumulator threshold (higher → fewer, surer circles)
        hough_min_r=8,
        hough_max_r=80,
        # ---- classical FAST params ----
        fast_threshold=30,
        # ---- classical MSER params ----
        mser_delta=5,
        mser_min_area=200,
        mser_max_area=3000,
        # NMS
        nms_iou=0.3,
        # column band to suppress before detection, e.g. the side-scan nadir /
        # water-column gap: (center_col, half_width). None = no masking.
        mask_band=None,
        # minimum local contrast (region mean minus surrounding-ring mean, in
        # 0-255 units) for a *classical* detection to be kept. CFAR hits are
        # gated by their own z-score and shadow test instead. 0 disables.
        min_contrast=30,
        on_detection=None,
        on_frame=None,
    ):
        self.max_rows = max_rows
        self.detect_every = detect_every
        self.display_every = display_every
        self.detector = detector
        self.nms_iou = nms_iou
        self.mask_band = mask_band
        self.min_contrast = min_contrast
        self.on_detection = on_detection
        self.on_frame = on_frame

        # CFAR
        self._cfar_guard_x = cfar_guard_x
        self._cfar_train_x = cfar_train_x
        self._cfar_band_y = cfar_band_y
        self._cfar_k = cfar_k
        self._cfar_min_area = cfar_min_area
        self._cfar_close_ksize = cfar_close_ksize
        self._cfar_merge_gap = cfar_merge_gap
        self._cfar_backend = cfar_backend
        self._cfar_device = None   # lazily resolved torch device

        # shadow
        self._shadow_gate = shadow_gate
        self._shadow_db = shadow_db
        self._shadow_min_len = shadow_min_len

        # classical
        self._hough_min_dist = hough_min_dist
        self._hough_p1 = hough_param1
        self._hough_p2 = hough_param2
        self._hough_min_r = hough_min_r
        self._hough_max_r = hough_max_r

        self._fast = cv2.FastFeatureDetector_create(
            threshold=fast_threshold, nonmaxSuppression=True
        )
        self._mser = cv2.MSER_create(
            delta=mser_delta,
            min_area=mser_min_area,
            max_area=mser_max_area,
        )

        self._rows = deque(maxlen=max_rows)
        self._count = 0
        self._last_objects = []
        # Rows that had scrolled off the top when the last detection ran.
        # Detection only runs every `detect_every` pings, but the display
        # refreshes every `display_every` pings on a freshly-scrolled image —
        # so between detections we shift the stale boxes up by however many rows
        # have since dropped off the top, keeping each box on its feature
        # instead of letting it drift (the "flicker").
        self._last_detect_dropped = 0

    # ---- ingest ------------------------------------------------------------
    def _rows_dropped(self):
        """Total rows that have scrolled off the top of the rolling window."""
        return self._count - len(self._rows)

    def add_ping(self, ping):
        self._rows.append(np.asarray(ping["samples_db"], dtype=np.float64))
        self._count += 1

        if self._count % self.detect_every == 0:
            objects, image = self._detect(ping)
            self._last_objects = objects if objects else []
            self._last_detect_dropped = self._rows_dropped()
            if self.on_detection:
                self.on_detection(objects, ping, image)
            return objects, image

        if self.on_frame and self._count % self.display_every == 0 and len(self._rows) >= 3:
            img = self._build_display_image(list(self._rows))
            if img is not None:
                shift = self._rows_dropped() - self._last_detect_dropped
                tracked = [dict(o, y=o["y"] - shift) for o in self._last_objects]
                self.on_frame(tracked, ping, img)

        return None, None

    # ---- imager ------------------------------------------------------------
    # Real sonar backscatter has high dynamic range (strong specular returns
    # saturate while seafloor sits low) and a range-dependent brightness
    # profile (TVG). A naive equalizeHist + global min-max amplifies seafloor
    # speckle into thousands of false detections and crushes real targets.
    #
    # Instead we:
    #   1. column flat-field — divide each column by its along-track median,
    #      removing the static range/TVG pattern so targets stand out
    #   2. percentile stretch — clip to [2, 99.5]% so saturation spikes don't
    #      dominate the contrast
    #   3. median despeckle — remove single-pixel backscatter noise

    @staticmethod
    def _stack(rows):
        width = int(np.median([r.shape[0] for r in rows]))
        fixed = []
        for r in rows:
            if r.shape[0] >= width:
                fixed.append(r[:width])
            else:
                fixed.append(np.pad(r, (0, width - r.shape[0])))
        return np.vstack(fixed).astype(np.float64)

    @staticmethod
    def _flatfield_stretch(img, lo=2.0, hi=99.5):
        col_med = np.median(img, axis=0)
        col_med[col_med == 0] = 1e-6
        f = img / col_med
        plo, phi = np.percentile(f, [lo, hi])
        if phi <= plo:
            phi = plo + 1e-6
        f = np.clip((f - plo) / (phi - plo), 0.0, 1.0)
        return (f * 255).astype(np.uint8)

    def _apply_mask(self, img):
        if self.mask_band is not None:
            center, half = self.mask_band
            a = max(0, center - half)
            b = min(img.shape[1], center + half)
            img[:, a:b] = 0
        return img

    def _build_image(self, rows):
        img = self._flatfield_stretch(self._stack(rows))
        img = cv2.medianBlur(img, 5)
        return self._apply_mask(img)

    def _build_cfar_image(self, rows):
        """dB-excess image for CFAR: each column flat-fielded by *subtracting*
        its along-track median. Samples are in dB (log-compressed), where the
        TVG/range falloff is additive, so subtraction removes it and leaves
        seabed near 0 dB and a target as a positive dB excess. Unlike the
        display image this is NOT percentile-stretched/clipped.

        Working in dB (rather than converting to linear and dividing) keeps the
        dynamic range bounded: a few hot clutter pixels would otherwise dominate
        the local variance and push the CFAR threshold out of reach of a
        genuine but moderate target."""
        arr = self._stack(rows)
        col_med = np.median(arr, axis=0)
        f = (arr - col_med).astype(np.float32)   # dB above local column background
        f = cv2.medianBlur(f, 5)
        return self._apply_mask(f)

    def _build_display_image(self, rows):
        """Lighter path for live display — flat-field without the median pass."""
        return self._flatfield_stretch(self._stack(rows))

    # ---- detect dispatch ---------------------------------------------------
    def _detect(self, ping=None):
        if len(self._rows) < 3:
            return [], None
        image = self._build_image(list(self._rows))
        if image.shape[0] < 3 or image.shape[1] < 3:
            return [], image

        objects = []
        if self.detector in ("cfar", "both"):
            cfar_img = self._build_cfar_image(list(self._rows))
            objects += self._cfar_detect(cfar_img, ping)
        if self.detector in ("classical", "both"):
            classical = self._hough_detect(image) + self._roi_detect(image)
            if self.min_contrast > 0:
                classical = [o for o in classical
                             if self._contrast(image, o) >= self.min_contrast]
            objects += classical

        return self._nms(objects, self.nms_iou), image

    # ---- CFAR detector -----------------------------------------------------
    @staticmethod
    def _box_sums(integral, hy, hx):
        """Windowed sums of a (2*hy+1)×(2*hx+1) box centred on every pixel,
        computed from an integral image (summed-area table) in O(1) per pixel.
        Window bounds are clamped at the image border; the matching cell count
        is returned so means stay correct near edges."""
        H = integral.shape[0] - 1
        W = integral.shape[1] - 1
        ys = np.arange(H)
        xs = np.arange(W)
        y0 = np.clip(ys - hy, 0, H)[:, None]
        y1 = np.clip(ys + hy + 1, 0, H)[:, None]
        x0 = np.clip(xs - hx, 0, W)[None, :]
        x1 = np.clip(xs + hx + 1, 0, W)[None, :]
        total = (integral[y1, x1] - integral[y0, x1]
                 - integral[y1, x0] + integral[y0, x0])
        count = (y1 - y0) * (x1 - x0)
        return total, count

    def _cfar_mask_numpy(self, image):
        """CA-CFAR (z-score form) detection mask on the CPU via integral images.

        The reference band shares the guard's along-track height (hy) but spans
        the range wings from guard_x..train_x — background is estimated from
        range-direction neighbours only, so along-track targets don't suppress
        their own interior."""
        img = image.astype(np.float64)
        s, sq = cv2.integral2(img)            # both (H+1, W+1) float64

        hy = self._cfar_band_y
        gx = self._cfar_guard_x
        tx = self._cfar_train_x

        train_sum, train_n = self._box_sums(s, hy, tx)
        guard_sum, guard_n = self._box_sums(s, hy, gx)
        train_sq, _        = self._box_sums(sq, hy, tx)
        guard_sq, _        = self._box_sums(sq, hy, gx)

        ref_sum = train_sum - guard_sum
        ref_sq  = train_sq - guard_sq
        ref_n   = (train_n - guard_n).astype(np.float64)
        ref_n[ref_n < 1] = 1.0

        mean = ref_sum / ref_n
        var  = ref_sq / ref_n - mean * mean
        var[var < 0] = 0.0
        std  = np.sqrt(var)

        threshold = mean + self._cfar_k * std
        mask = (img > threshold) & (img > 0)
        return mask, threshold

    def _cfar_mask_torch(self, image):
        """Same CFAR computation on the GPU (or torch CPU) — used on the edge
        server. Falls back to numpy if torch is unavailable."""
        try:
            import torch
        except ImportError:
            return self._cfar_mask_numpy(image)

        if self._cfar_device is None:
            self._cfar_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        dev = self._cfar_device

        img = torch.as_tensor(image, dtype=torch.float64, device=dev)
        H, W = img.shape

        def integral(a):
            sat = torch.zeros((H + 1, W + 1), dtype=torch.float64, device=dev)
            sat[1:, 1:] = a.cumsum(0).cumsum(1)
            return sat

        s  = integral(img)
        sq = integral(img * img)

        def box_sums(sat, hy, hx):
            ys = torch.arange(H, device=dev)
            xs = torch.arange(W, device=dev)
            y0 = (ys - hy).clamp(0, H)[:, None]
            y1 = (ys + hy + 1).clamp(0, H)[:, None]
            x0 = (xs - hx).clamp(0, W)[None, :]
            x1 = (xs + hx + 1).clamp(0, W)[None, :]
            total = sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]
            count = (y1 - y0) * (x1 - x0)
            return total, count.to(torch.float64)

        hy = self._cfar_band_y
        gx = self._cfar_guard_x
        tx = self._cfar_train_x
        train_sum, train_n = box_sums(s, hy, tx)
        guard_sum, guard_n = box_sums(s, hy, gx)
        train_sq, _        = box_sums(sq, hy, tx)
        guard_sq, _        = box_sums(sq, hy, gx)

        ref_n = (train_n - guard_n).clamp(min=1.0)
        mean  = (train_sum - guard_sum) / ref_n
        var   = ((train_sq - guard_sq) / ref_n - mean * mean).clamp(min=0.0)
        std   = var.sqrt()

        threshold = mean + self._cfar_k * std
        mask = (img > threshold) & (img > 0)
        return mask.cpu().numpy(), threshold.cpu().numpy()

    def _cfar_detect(self, image, ping=None):
        if self._cfar_backend == "torch":
            mask, _ = self._cfar_mask_torch(image)
        else:
            mask, _ = self._cfar_mask_numpy(image)

        mask_u8 = mask.astype(np.uint8)
        # Knit a target's scattered bright pixels into one coherent blob.
        if self._cfar_close_ksize > 1:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self._cfar_close_ksize, self._cfar_close_ksize))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k)

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        nadir_col = (ping or {}).get("nadir_col", 0)

        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self._cfar_min_area:
                continue
            obj = {"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                   "source": "cfar"}
            if self._shadow_gate and not self._has_shadow(image, obj, nadir_col):
                continue
            obj["score"] = float(self._contrast(image, obj))
            out.append(obj)

        if self._cfar_merge_gap > 0:
            out = self._merge_nearby(out, self._cfar_merge_gap)
        return out

    @staticmethod
    def _merge_nearby(boxes, gap):
        """Union boxes whose bounding rects are within `gap` px of each other —
        knits an elongated target's fragments into a single contact. Union-find
        over the gap-expanded rectangles."""
        if len(boxes) < 2:
            return boxes

        parent = list(range(len(boxes)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            parent[find(a)] = find(b)

        def near(i, j):
            a, b = boxes[i], boxes[j]
            return not (
                a["x"] - gap > b["x"] + b["w"] or
                b["x"] - gap > a["x"] + a["w"] or
                a["y"] - gap > b["y"] + b["h"] or
                b["y"] - gap > a["y"] + a["h"]
            )

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if near(i, j):
                    union(i, j)

        groups = {}
        for i in range(len(boxes)):
            groups.setdefault(find(i), []).append(boxes[i])

        merged = []
        for g in groups.values():
            x0 = min(b["x"] for b in g)
            y0 = min(b["y"] for b in g)
            x1 = max(b["x"] + b["w"] for b in g)
            y1 = max(b["y"] + b["h"] for b in g)
            merged.append({
                "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                "source": "cfar",
                "score": max(b.get("score", 0.0) for b in g),
            })
        return merged

    def _has_shadow(self, image, obj, nadir_col):
        """True if a region several dB below background sits just beyond the hit
        on its far-range side. `image` is the dB-excess CFAR image (seabed ≈ 0).

        Range increases away from the transducer. For a single channel the
        transducer is at column 0, so far-range is +x. For a combined swath the
        nadir is at nadir_col and far-range points away from it on each side.
        """
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        cx = x + w / 2.0
        far_right = True if nadir_col <= 0 else (cx >= nadir_col)

        probe = max(self._shadow_min_len, w)
        if far_right:
            sx0 = x + w
            sx1 = min(image.shape[1], sx0 + probe)
        else:
            sx1 = x
            sx0 = max(0, sx1 - probe)

        shadow = image[y:y + h, sx0:sx1]
        if shadow.size == 0:
            return False
        return float(shadow.mean()) < -self._shadow_db

    # ---- classical detectors -----------------------------------------------
    @staticmethod
    def _contrast(image, obj, pad=6):
        """How much brighter the detection is than its surrounding ring (0-255)."""
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        inner = image[y:y + h, x:x + w]
        if inner.size == 0:
            return 0.0
        Y0, Y1 = max(0, y - pad), min(image.shape[0], y + h + pad)
        X0, X1 = max(0, x - pad), min(image.shape[1], x + w + pad)
        outer = image[Y0:Y1, X0:X1]
        ring_n = outer.size - inner.size
        if ring_n <= 0:
            return 0.0
        ring_mean = (float(outer.sum()) - float(inner.sum())) / ring_n
        return float(inner.mean()) - ring_mean

    def _hough_detect(self, image):
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self._hough_min_dist,
            param1=self._hough_p1,
            param2=self._hough_p2,
            minRadius=self._hough_min_r,
            maxRadius=self._hough_max_r,
        )
        if circles is None:
            return []
        out = []
        for (cx, cy, r) in np.round(circles[0]).astype(int):
            out.append({
                "x": int(cx - r), "y": int(cy - r),
                "w": int(2 * r),  "h": int(2 * r),
                "source": "hough",
            })
        return out

    def _roi_detect(self, image):
        keypoints = self._fast.detect(image)
        if not keypoints:
            return []

        kp_xy = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        _, bboxes = self._mser.detectRegions(image)

        out = []
        for (x, y, w, h) in bboxes:
            # Reject regions with no FAST keypoint inside — likely noise
            in_box = (
                (kp_xy[:, 0] >= x) & (kp_xy[:, 0] <= x + w) &
                (kp_xy[:, 1] >= y) & (kp_xy[:, 1] <= y + h)
            ).any()
            if not in_box:
                continue
            # Reject unrealistically elongated blobs
            aspect = w / h if h > 0 else 0
            if not (0.2 <= aspect <= 5.0):
                continue
            out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                        "source": "roi"})
        return out

    # ---- NMS ---------------------------------------------------------------
    @staticmethod
    def _nms(boxes, iou_threshold=0.3):
        """Greedy IoU NMS — keeps larger boxes, suppresses overlapping smaller ones."""
        if not boxes:
            return []
        sorted_boxes = sorted(boxes, key=lambda b: b["w"] * b["h"], reverse=True)
        kept = []
        for box in sorted_boxes:
            x1, y1 = box["x"], box["y"]
            x2, y2 = x1 + box["w"], y1 + box["h"]
            overlaps = False
            for k in kept:
                kx2, ky2 = k["x"] + k["w"], k["y"] + k["h"]
                ix1, iy1 = max(x1, k["x"]), max(y1, k["y"])
                ix2, iy2 = min(x2, kx2),    min(y2, ky2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = box["w"]*box["h"] + k["w"]*k["h"] - inter
                    if union > 0 and inter / union > iou_threshold:
                        overlaps = True
                        break
            if not overlaps:
                kept.append(box)
        return kept
