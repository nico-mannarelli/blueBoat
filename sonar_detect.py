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
        # ---- optional 2-D reference ring -----------------------------------
        # Range-only reference (the default) avoids self-suppressing a long
        # along-track target, but it floods on textured seabed: a sand-ripple
        # crest has dark troughs to its left/right, so it reads as a target.
        # Setting train_y > guard_y turns the reference into a 2-D ring — a
        # guard box that excludes the target's own body, with training cells
        # *beyond it in both axes*. A ripple field still has crests beyond the
        # guard, so the background rises and the crest self-suppresses; an
        # isolated target sits on clean seabed and still fires. guard_y must
        # exceed the target's along-track half-height or it self-masks.
        cfar_guard_y=None,     # None = cfar_band_y (range-only). e.g. 25 for 2-D
        cfar_train_y=None,     # None = cfar_band_y (range-only). e.g. 55 for 2-D
        cfar_k=6.0,            # threshold = local_mean + k * local_std (higher → less sensitive)
        cfar_min_area=80,      # minimum connected-component area (px) to keep
        cfar_close_ksize=7,    # morphological-close kernel to knit a target's pixels together
        cfar_merge_gap=12,     # merge CFAR boxes whose gap is below this (px); 0 disables
        cfar_backend="numpy",  # "numpy" (CPU) or "torch" (GPU on the edge server)
        # Cluster-confirm: keep a CFAR box only if it overlaps a dense DBSCAN
        # cluster of FAST+MSER keypoints (OpenSidescan's precision mechanism).
        # A real structured object lights up CFAR AND throws a knot of texture
        # features; flat seabed / ripple / speckle fires CFAR but has no dense
        # feature knot, so it gets dropped. Uses the dbscan_* params below.
        cfar_confirm=False,
        # A confirming cluster must hold at least this many features, else it's
        # treated as loose speckle, not a real knot. None = 3 * dbscan_min.
        cfar_confirm_min_feat=None,
        # ---- shadow scoring / gating ----
        # A proud object (mine, rock) throws an acoustic shadow on its far-range
        # side; flat seabed, sand ripple and most bright clutter do not. This is
        # the main *label-free* way to tell a real target from bright noise.
        #   "soft" (default) — measure shadow strength and ADD it to the contact
        #                      score, so proud objects rank above clutter while a
        #                      shadowless-but-real target (a flat waterlogged log)
        #                      is never dropped. Recall-safe.
        #   "hard"           — drop any detection without a shadow. Use only for
        #                      proud-object searches; it will reject flat targets.
        #   "off"            — ignore shadows entirely.
        shadow_mode="soft",
        shadow_weight=2.0,     # soft mode: score += shadow_weight * shadow_dB
        shadow_gate=None,      # back-compat: shadow_gate=True forces "hard"
        shadow_db=5.0,         # hard mode: shadow must average this many dB below background
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
        # ---- ROI (OpenSidescan-style) DBSCAN clustering --------------------
        # The faithful OpenSidescan detector clusters the FAST+MSER keypoint
        # cloud with DBSCAN and keeps only dense clusters: a real structured
        # object throws a knot of corners/blobs, while speckle and a lone
        # ripple crest are sparse and get dropped as noise. This density test
        # — not intensity — is OpenSidescan's precision mechanism.
        dbscan_epsilon=20.0,     # neighbourhood radius (px)
        dbscan_min_points=5,     # min features within epsilon to seed a cluster
        roi_padding=20,          # px added around a cluster's bounding box
        kp_cap=100,              # max features in the combined cloud fed to DBSCAN
        # ---- blob / contour shape detector ---------------------------------
        # A pure-shape detector: threshold the equalized feature image to the
        # brightest pixels, knit them into connected blobs, and keep only blobs
        # whose *shape* looks like an object — right size, not absurdly
        # elongated, reasonably solid/filled, and brighter than the surrounding
        # ring. No per-pixel CFAR and no feature-density: this keys on the
        # outline of a coherent bright return, so seabed speckle (which never
        # knits into a solid blob of the right size) is rejected by morphology.
        # Detection-image contrast curve. None = CLAHE (local equalize, the old
        # behaviour) — good for the dead-end roi/confirm modes but it amplifies
        # small speckle into full-contrast features. Set a value (e.g. 1.8) to
        # use a GLOBAL gamma stretch instead: g = g**detect_gamma on the [0,1]
        # normalized image, which pushes dim speckle toward black while keeping
        # strong returns bright, without pumping up local noise. This is the
        # knob that makes the blob detector ignore small bright specks.
        detect_gamma=None,
        blob_pct=98.0,           # keep pixels above this percentile of the feature image
        blob_min_area=300,       # smallest blob (px^2) to keep — sized for logs /
                                 # large rocks, not pebbles. Raise to be stricter.
        blob_max_area=60000,     # largest blob (px^2) to keep
        blob_min_aspect=0.12,    # w/h must be >= this (rejects 1-px lines)
        blob_max_aspect=8.0,     # w/h must be <= this (rejects long ripple streaks)
        blob_min_solidity=0.45,  # blob area / convex-hull area (rejects ragged speckle)
        blob_min_extent=0.30,    # blob area / bbox area (rejects sparse scatter)
        blob_min_contrast=8.0,   # blob mean minus surrounding ring (feature-image units)
        blob_close_ksize=5,      # morphological-close kernel to knit a blob together
        blob_open_ksize=5,       # morphological-open kernel: erases sub-target
                                 # structure and severs speckle bridges. Bigger
                                 # = ignores larger small stuff. Pebble killer.
        # NMS
        nms_iou=0.3,
        # column band to suppress before detection, e.g. the side-scan nadir /
        # water-column gap: (center_col, half_width). None = no masking.
        mask_band=None,
        # half-width (px) of the blurry nadir / water-column band to exclude
        # from CFAR. The near-transducer water column is a bright, smeared zone
        # that has no real seabed and otherwise produces constant false hits;
        # this drops any detection inside it. Applied automatically per-ping
        # using each ping's nadir_col (for a single channel nadir is column 0,
        # so this masks the near-range columns). 0 disables.
        nadir_guard=50,
        # margin (px) trimmed off the far-range LEFT/RIGHT edges of the swath
        # before CFAR. At the outer boundary the reference window is truncated,
        # so the background estimate is unreliable and CFAR over-fires on the
        # low-SNR far-range fuzz (the cluster of false hits at the swath edge).
        # None = auto = cfar_guard_x + cfar_train_x (the reference half-width,
        # the principled margin). 0 disables.
        edge_guard=None,
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
        self._nadir_guard = nadir_guard
        self._edge_guard = (edge_guard if edge_guard is not None
                            else cfar_guard_x + cfar_train_x)
        self.min_contrast = min_contrast
        self.on_detection = on_detection
        self.on_frame = on_frame

        # CFAR
        self._cfar_guard_x = cfar_guard_x
        self._cfar_train_x = cfar_train_x
        self._cfar_band_y = cfar_band_y
        # along-track guard/train half-heights. Default to band_y so the
        # reference stays the range-only horizontal band (back-compat). Set
        # train_y > guard_y to switch to a 2-D ring that suppresses ripple.
        self._cfar_guard_y = cfar_guard_y if cfar_guard_y is not None else cfar_band_y
        self._cfar_train_y = cfar_train_y if cfar_train_y is not None else cfar_band_y
        self._cfar_k = cfar_k
        self._cfar_min_area = cfar_min_area
        self._cfar_close_ksize = cfar_close_ksize
        self._cfar_merge_gap = cfar_merge_gap
        self._cfar_confirm = cfar_confirm
        self._confirm_min_feat = (cfar_confirm_min_feat
                                  if cfar_confirm_min_feat is not None
                                  else 3 * dbscan_min_points)
        self._cfar_backend = cfar_backend
        self._cfar_device = None   # lazily resolved torch device

        # shadow
        if shadow_gate is True:          # back-compat with the old boolean gate
            shadow_mode = "hard"
        self._shadow_mode = shadow_mode
        self._shadow_weight = shadow_weight
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
        self._dbscan_eps = dbscan_epsilon
        self._dbscan_min = dbscan_min_points
        self._roi_padding = roi_padding
        self._kp_cap = kp_cap
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._detect_gamma = detect_gamma

        # blob / contour shape detector
        self._blob_pct = blob_pct
        self._blob_min_area = blob_min_area
        self._blob_max_area = blob_max_area
        self._blob_min_aspect = blob_min_aspect
        self._blob_max_aspect = blob_max_aspect
        self._blob_min_solidity = blob_min_solidity
        self._blob_min_extent = blob_min_extent
        self._blob_min_contrast = blob_min_contrast
        self._blob_close_ksize = blob_close_ksize
        self._blob_open_ksize = blob_open_ksize

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
        # NB: do NOT zero the nadir band here. Zeroing creates a flat,
        # zero-variance region whose edge depresses the CFAR threshold for
        # neighbouring real-seabed pixels (they over-fire in a fringe just
        # outside the band). The nadir is excluded the right way — by zeroing
        # the post-threshold mask inside the guard bands (_cfar_detect) — which
        # leaves the threshold estimate uncorrupted.
        return f

    def _build_feature_image(self, rows):
        """Grayscale image for FAST/MSER feature detection, built like
        OpenSidescan's: a contrast-stretched, histogram-equalized greyscale
        (strong return = bright). Crucially this is NOT the dB-excess CFAR
        image — that amplifies per-pixel speckle and floods the feature
        detectors. Two deviations from the paper's plain global equalize, both
        to stop seabed noise from generating a corner at every pixel:
          * a median despeckle first, and
          * CLAHE (clip-limited, local) instead of global equalizeHist, which
            would otherwise stretch the seabed noise floor into full contrast.
        The result: smooth seabed stays feature-poor, only structured objects
        throw the dense feature knots DBSCAN needs."""
        arr = self._stack(rows)
        f = arr - np.median(arr, axis=0)               # dB excess, strong = +
        lo, hi = np.percentile(f, [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1e-6
        g = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
        if self._detect_gamma is not None:
            # Global gamma curve on the normalized [0,1] image. gamma > 1 bends
            # the curve so dim seabed speckle is pushed toward black while strong
            # returns stay bright — a monotonic, image-wide contrast that cannot
            # amplify local noise the way CLAHE's per-tile equalize does. This is
            # what makes the blob detector ignore small bright specks.
            g = np.power(g, self._detect_gamma)
            g = (g * 255).astype(np.uint8)
            g = cv2.medianBlur(g, 5)
        else:
            # Legacy path (roi / cfar-confirm): CLAHE local equalize.
            g = (g * 255).astype(np.uint8)
            g = cv2.medianBlur(g, 5)
            g = self._clahe.apply(g)
        # Do NOT mask the nadir band here: a hard 0-edge would itself throw a
        # line of FAST corners. The nadir is handled by the guard bands, which
        # drop any detection whose centre lands in the band.
        return g

    def _build_display_image(self, rows):
        """Grayscale image for the UI, oriented strong-return = bright (SonarView
        convention), the opposite of the detection images. Flat-field by
        subtracting the column median in dB so seabed sits mid-low and hard
        returns are bright; shadows go dark. Cosmetic only — sonar_display.py
        adds the colormap/gamma on top.

        SonarView-style quality comes from three things done here on the dB
        image (before any colour/gamma): a wide robust stretch so the seabed
        fills the mid-tones instead of crushing to black, a gentle S-curve that
        deepens shadows and lifts returns without clipping, and an
        edge-preserving despeckle that knocks down per-ping salt-and-pepper
        without smearing real targets or the shadow edges the eye reads as
        relief."""
        arr = self._stack(rows)
        f = arr - np.median(arr, axis=0)          # dB excess: strong = positive = bright

        # Robust stretch. A wider low/high window than a hard 2/99.5 clip keeps
        # the seabed texture in the mid greys (SonarView never crushes the
        # bottom to pure black) while still pinning the brightest returns white.
        lo, hi = np.percentile(f, [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1e-6
        g = np.clip((f - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

        # Edge-preserving despeckle. Bilateral smooths the within-seabed speckle
        # but leaves target/shadow boundaries crisp, the single biggest reason
        # SonarView imagery looks "clean" next to a raw waterfall. Cheap at
        # display sizes; skip on tiny images where it is pointless.
        if g.shape[0] >= 8 and g.shape[1] >= 8:
            g = cv2.bilateralFilter(g, d=5, sigmaColor=0.10, sigmaSpace=3)

        # Gentle S-curve (smoothstep blended): darken the low seabed, lift the
        # upper returns, no hard clip. Weighted toward the curve so the bottom
        # sits dark and targets/shadows keep their contrast — the SonarView look.
        s = g * g * (3.0 - 2.0 * g)
        g = 0.2 * g + 0.8 * s

        return (np.clip(g, 0.0, 1.0) * 255).astype(np.uint8)

    # ---- detect dispatch ---------------------------------------------------
    def _detect(self, ping=None):
        if len(self._rows) < 3:
            return [], None
        rows = list(self._rows)
        # Detection disabled: just render the waterfall, never mark anything.
        if self.detector in (None, "off", "none"):
            return [], self._build_display_image(rows)
        image = self._build_image(rows)
        if image.shape[0] < 3 or image.shape[1] < 3:
            return [], self._build_display_image(rows)

        cfar_img = None
        feat_img = None
        objects = []
        if self.detector in ("cfar", "both"):
            cfar_img = self._build_cfar_image(rows)
            if self._cfar_confirm:
                feat_img = self._build_feature_image(rows)
            objects += self._cfar_detect(cfar_img, ping, cluster_image=feat_img)
        if self.detector in ("classical", "both"):
            classical = self._hough_detect(image) + self._roi_detect(image)
            if self.min_contrast > 0:
                classical = [o for o in classical
                             if self._contrast(image, o) >= self.min_contrast]
            for o in classical:
                o["score"] = float(self._contrast(image, o))
            # The CFAR mask is already blanked inside the nadir/edge guard
            # bands; the classical detectors run on the raw display image and
            # would re-introduce the fuzzy-band false hits we suppress for CFAR.
            # Drop any classical box centred in a guard band so "both" keeps the
            # same clean swath, just with the extra textured/shadow recall.
            classical = self._drop_in_guards(classical, image.shape[1], ping)
            objects += classical
        if self.detector == "roi":
            # Faithful OpenSidescan: FAST+MSER keypoint cloud on an equalized
            # greyscale -> DBSCAN density clustering -> one padded bbox per
            # cluster -> drop guard bands.
            feat_img = self._build_feature_image(rows)
            roi = self._roi_dbscan_detect(feat_img)
            for o in roi:
                o["score"] = float(self._contrast(image, o))
            objects += self._drop_in_guards(roi, image.shape[1], ping)
        if self.detector == "blob":
            # Pure-shape: threshold the equalized feature image, knit the bright
            # pixels into blobs, keep only blobs whose outline looks like an
            # object (size / aspect / solidity / extent / contrast). Speckle
            # never forms a solid blob of the right size, so morphology rejects
            # it without any per-pixel anomaly test.
            feat_img = self._build_feature_image(rows)
            blobs = self._blob_detect(feat_img)
            objects += self._drop_in_guards(blobs, image.shape[1], ping)

        # Shadow scoring (label-free precision lever): boost the score of
        # detections that cast an acoustic shadow on their far-range side, or in
        # "hard" mode drop the ones that don't. Needs the dB-excess CFAR image.
        if self._shadow_mode != "off" and objects:
            if cfar_img is None:
                cfar_img = self._build_cfar_image(rows)
            objects = self._apply_shadow(objects, cfar_img, ping)

        objects = self._nms(objects, self.nms_iou)
        # Hand the callback the display-oriented image (strong = bright), not the
        # inverted detection image, so the drawn window matches SonarView.
        return objects, self._build_display_image(rows)

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

        The reference is train_box(train_y, train_x) minus guard_box(guard_y,
        guard_x). With train_y == guard_y == band_y this is the range-only
        horizontal band (wings from guard_x..train_x) — background from range
        neighbours only, so along-track targets don't suppress their interior.
        With train_y > guard_y it becomes a 2-D ring that also samples
        along-track, suppressing ripple fields."""
        img = image.astype(np.float64)
        s, sq = cv2.integral2(img)            # both (H+1, W+1) float64

        gy = self._cfar_guard_y
        ty = self._cfar_train_y
        gx = self._cfar_guard_x
        tx = self._cfar_train_x

        train_sum, train_n = self._box_sums(s, ty, tx)
        guard_sum, guard_n = self._box_sums(s, gy, gx)
        train_sq, _        = self._box_sums(sq, ty, tx)
        guard_sq, _        = self._box_sums(sq, gy, gx)

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

        gy = self._cfar_guard_y
        ty = self._cfar_train_y
        gx = self._cfar_guard_x
        tx = self._cfar_train_x
        train_sum, train_n = box_sums(s, ty, tx)
        guard_sum, guard_n = box_sums(s, gy, gx)
        train_sq, _        = box_sums(sq, ty, tx)
        guard_sq, _        = box_sums(sq, gy, gx)

        ref_n = (train_n - guard_n).clamp(min=1.0)
        mean  = (train_sum - guard_sum) / ref_n
        var   = ((train_sq - guard_sq) / ref_n - mean * mean).clamp(min=0.0)
        std   = var.sqrt()

        threshold = mean + self._cfar_k * std
        mask = (img > threshold) & (img > 0)
        return mask.cpu().numpy(), threshold.cpu().numpy()

    def _guard_bands(self, width, ping=None):
        """Column ranges [(a, b), ...] to exclude from detection: the blurry
        nadir / water-column band and the truncated far-range left/right edges.

        nadir_col comes from the ping; for a single channel nadir sits at
        column 0, so the nadir guard clears the near-range smear. These bands
        are blanked on the CFAR mask AND used to drop classical detections, so
        every detector treats the fuzzy zones the same way."""
        bands = []
        guard = self._nadir_guard
        if guard > 0:
            ncol = (ping or {}).get("nadir_col", 0)
            if ncol and ncol > 0:
                bands.append((max(0, ncol - guard), min(width, ncol + guard)))
            else:
                bands.append((0, min(width, guard)))
        eg = self._edge_guard
        if eg > 0:
            bands.append((0, min(width, eg)))
            bands.append((max(0, width - eg), width))
        return bands

    def _drop_in_guards(self, objects, width, ping=None):
        """Remove detections whose horizontal centre falls inside any guard band
        (nadir / far-range edge). Used for classical hits, whose detectors run
        on the un-masked display image."""
        bands = self._guard_bands(width, ping)
        if not bands:
            return objects
        kept = []
        for o in objects:
            cx = o["x"] + o["w"] / 2.0
            if any(a <= cx < b for a, b in bands):
                continue
            kept.append(o)
        return kept

    def _cfar_detect(self, image, ping=None, cluster_image=None):
        if self._cfar_backend == "torch":
            mask, _ = self._cfar_mask_torch(image)
        else:
            mask, _ = self._cfar_mask_numpy(image)

        mask_u8 = mask.astype(np.uint8)

        # Blank the blurry nadir / water-column band and the truncated far-range
        # edges so CFAR never fires there. Done on the mask (before components
        # form) so a target can't bridge across a band either. The same bands
        # are reused to filter classical detections in "both" mode.
        W = mask_u8.shape[1]
        for a, b in self._guard_bands(W, ping):
            mask_u8[:, a:b] = 0

        # Knit a target's scattered bright pixels into one coherent blob.
        if self._cfar_close_ksize > 1:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self._cfar_close_ksize, self._cfar_close_ksize))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k)

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self._cfar_min_area:
                continue
            obj = {"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                   "source": "cfar"}
            obj["score"] = float(self._contrast(image, obj))
            out.append(obj)

        if self._cfar_merge_gap > 0:
            out = self._merge_nearby(out, self._cfar_merge_gap)

        if self._cfar_confirm and out:
            out = self._confirm_with_clusters(
                out, cluster_image if cluster_image is not None else image)
        return out

    def _confirm_with_clusters(self, boxes, image):
        """Keep a CFAR box only if its centre sits inside a *substantial* DBSCAN
        cluster of FAST+MSER keypoints. Real structured objects fire CFAR *and*
        throw a dense knot of texture features; flat seabed / ripple / speckle
        fire CFAR but only form small, loose clusters, so they are dropped here.

        Two strictness levers beyond DBSCAN's own eps/min_points:
          * the confirming cluster must hold >= confirm_min_feat features (kills
            the small loose clusters diffuse speckle throws), and
          * the CFAR box *centre* must fall inside the cluster box, not merely
            touch its edge (kills boxes that just graze a real knot)."""
        clusters = self._roi_dbscan_detect(image)
        floor = self._confirm_min_feat
        clusters = [c for c in clusters if c.get("n_feat", 0) >= floor]
        if not clusters:
            return []

        def centre_inside(b, c):
            bx = b["x"] + b["w"] / 2.0
            by = b["y"] + b["h"] / 2.0
            return (c["x"] <= bx <= c["x"] + c["w"] and
                    c["y"] <= by <= c["y"] + c["h"])

        return [b for b in boxes if any(centre_inside(b, c) for c in clusters)]

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

    def _apply_shadow(self, objects, cfar_image, ping=None):
        """Score (soft) or gate (hard) detections by the acoustic shadow on their
        far-range side. A proud object casts a dark shadow there; flat clutter
        and sand ripple do not, so this separates real targets from bright noise
        without any labels. In soft mode the shadow strength is added to the
        score so proud objects rank higher but nothing is dropped; in hard mode
        a detection without a sufficient shadow is removed."""
        nadir_col = (ping or {}).get("nadir_col", 0)
        out = []
        for o in objects:
            s = self._shadow_strength(cfar_image, o, nadir_col)
            o["shadow"] = round(s, 1)
            if self._shadow_mode == "hard" and s < self._shadow_db:
                continue
            if self._shadow_mode == "soft":
                o["score"] = float(o.get("score", 0.0)) + self._shadow_weight * s
            out.append(o)
        return out

    def _shadow_strength(self, image, obj, nadir_col):
        """How many dB below local seabed the region just beyond the detection's
        far-range side sits (0 if none, or if it's brighter than seabed).
        `image` is the dB-excess CFAR image where seabed ≈ 0, so a shadow reads
        as a negative-dB patch.

        Range increases away from the transducer. For a single channel the
        transducer is at column 0, so far-range is +x. For a combined swath the
        nadir is at nadir_col and far-range points away from it on each side."""
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
            return 0.0
        return max(0.0, -float(shadow.mean()))

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

    def _blob_detect(self, image):
        """Pure-shape detector. `image` is the equalized feature image (strong
        return = bright). Threshold to the brightest pixels, knit them into
        connected blobs with a close/open, then keep only blobs whose *shape*
        resembles a real object:

          * area within [min_area, max_area] — drops specks and swath-wide smears
          * aspect w/h within [min_aspect, max_aspect] — drops 1-px lines and
            long ripple streaks
          * solidity (blob area / convex-hull area) >= min_solidity — a real
            target fills its hull; ragged speckle clusters do not
          * extent (blob area / bbox area) >= min_extent — drops sparse scatter
            that only loosely fills its bounding box
          * contrast (blob mean - surrounding ring) >= min_contrast — the blob
            must actually be brighter than its neighbourhood

        Morphology + shape, no per-pixel CFAR and no feature density."""
        if image is None or image.size == 0:
            return []

        thr = float(np.percentile(image, self._blob_pct))
        # Guard against a flat/empty window where the percentile collapses onto
        # the noise floor and would threshold half the image.
        if thr <= 0:
            thr = 1.0
        binary = (image >= thr).astype(np.uint8) * 255

        if self._blob_open_ksize > 1:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self._blob_open_ksize, self._blob_open_ksize))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        if self._blob_close_ksize > 1:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self._blob_close_ksize, self._blob_close_ksize))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self._blob_min_area or area > self._blob_max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue
            aspect = w / float(h)
            if not (self._blob_min_aspect <= aspect <= self._blob_max_aspect):
                continue
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < self._blob_min_solidity:
                continue
            extent = area / float(w * h)
            if extent < self._blob_min_extent:
                continue
            obj = {"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                   "source": "blob"}
            con = self._contrast(image, obj)
            if con < self._blob_min_contrast:
                continue
            obj["score"] = float(con)
            out.append(obj)
        return out

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

    @staticmethod
    def _to_kp_input(image):
        """FAST/MSER on some OpenCV builds (e.g. 4.13) run an internal
        BGR->GRAY and reject a 1-channel image. Hand them a 3-channel copy;
        keypoint coordinates are identical either way."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _roi_detect(self, image):
        kpimg = self._to_kp_input(image)
        keypoints = self._fast.detect(kpimg)
        if not keypoints:
            return []

        kp_xy = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        _, bboxes = self._mser.detectRegions(kpimg)

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

    @staticmethod
    def _dbscan(points, eps, min_pts):
        """Minimal DBSCAN (no sklearn dependency). Returns a cluster label per
        point; -1 = noise. Density-based: a point with >= min_pts neighbours
        within eps seeds a cluster that grows through its dense neighbours, so
        only knots of features survive — the OpenSidescan precision step."""
        n = len(points)
        labels = np.full(n, -1, dtype=int)
        if n == 0:
            return labels
        pts = np.asarray(points, dtype=np.float64)
        visited = np.zeros(n, dtype=bool)
        eps2 = float(eps) * float(eps)

        # Spatial grid (cell = eps) so a neighbour query only scans the 3x3
        # block of cells around a point instead of all n points. Turns DBSCAN
        # from O(n^2) into ~O(n) for the dense keypoint clouds a busy waterfall
        # produces — without this the replay stalls once the image fills up.
        cell = max(float(eps), 1e-6)
        gxy = np.floor(pts / cell).astype(np.int64)
        grid = {}
        for idx in range(n):
            grid.setdefault((gxy[idx, 0], gxy[idx, 1]), []).append(idx)

        def neighbours(i):
            cx, cy = gxy[i, 0], gxy[i, 1]
            cand = []
            for ax in (cx - 1, cx, cx + 1):
                for ay in (cy - 1, cy, cy + 1):
                    bucket = grid.get((ax, ay))
                    if bucket:
                        cand.extend(bucket)
            if not cand:
                return np.empty(0, dtype=int)
            cand = np.asarray(cand)
            d2 = (pts[cand, 0] - pts[i, 0]) ** 2 + (pts[cand, 1] - pts[i, 1]) ** 2
            return cand[d2 <= eps2]

        cluster = 0
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            nb = neighbours(i)
            if len(nb) < min_pts:
                continue                       # noise (may be claimed as border)
            labels[i] = cluster
            seeds = list(nb)
            k = 0
            while k < len(seeds):
                j = seeds[k]
                k += 1
                if not visited[j]:
                    visited[j] = True
                    nbj = neighbours(j)
                    if len(nbj) >= min_pts:
                        seeds.extend(nbj.tolist())
                if labels[j] == -1:
                    labels[j] = cluster
            cluster += 1
        return labels

    def _roi_dbscan_detect(self, image):
        """OpenSidescan's RoiDetector, ported: gather FAST corners + MSER blobs
        into one keypoint cloud, drop edge keypoints, DBSCAN-cluster the cloud,
        and emit one padded bounding box per dense cluster. Density — not
        intensity — is what separates a structured target from speckle."""
        kpimg = self._to_kp_input(image)
        fast_kps = list(self._fast.detect(kpimg))
        try:
            mser_kps = list(self._mser.detect(kpimg))
        except cv2.error:
            mser_kps = []
        # Cap the COMBINED cloud to the kp_cap most salient features (default
        # 100). The strongest FAST corners by response come first — they
        # concentrate tightly on structured objects — then MSER blobs fill any
        # remaining slots. (MSER fires abundantly on bland seabed and must not
        # be allowed to crowd out the object's corners.) Forcing the detector
        # to see only the top ~100 features is what makes it
        # OpenSidescan-insensitive: diffuse seabed speckle can't out-vote a real
        # object's dense knot once only the most salient features survive.
        cap = self._kp_cap
        fast_kps.sort(key=lambda kp: kp.response, reverse=True)
        if len(fast_kps) >= cap:
            kps = fast_kps[:cap]
        else:
            kps = fast_kps + mser_kps[:cap - len(fast_kps)]
        if not kps:
            return []

        H, W = image.shape[:2]
        pts = np.array([kp.pt for kp in kps], dtype=np.float64)
        sizes = np.array([kp.size for kp in kps], dtype=np.float64)
        # drop keypoints whose extent leaves the image (OpenSidescan edge guard)
        keep = ((pts[:, 0] - sizes >= 0) & (pts[:, 0] + sizes <= W) &
                (pts[:, 1] - sizes >= 0) & (pts[:, 1] + sizes <= H))
        pts, sizes = pts[keep], sizes[keep]
        if len(pts) == 0:
            return []

        labels = self._dbscan(pts, self._dbscan_eps, self._dbscan_min)
        if labels.size == 0 or labels.max() < 0:
            return []

        pad = self._roi_padding
        out = []
        for c in range(labels.max() + 1):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                continue
            cx, cy, sz = pts[idx, 0], pts[idx, 1], sizes[idx]
            x0 = max(0, int((cx - sz).min() - pad))
            y0 = max(0, int((cy - sz).min() - pad))
            x1 = min(W - 1, int((cx + sz).max() + pad))
            y1 = min(H - 1, int((cy + sz).max() + pad))
            if x1 <= x0 or y1 <= y0:
                continue
            out.append({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                        "source": "roi", "n_feat": int(len(idx))})
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
