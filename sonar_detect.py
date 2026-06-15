"""
sonar_detect.py
Builds a rolling waterfall image from pings and runs two-stage detection:

  Stage 1 — Hough Circle detector
      Finds compact, circular acoustic returns (mines, rocks, debris).
      High precision, low false-positive rate.

  Stage 2 — ROI detector  (FAST keypoints → MSER regions)
      FAST locates candidate edge/corner points cheaply; MSER regions are
      only kept when they contain at least one FAST keypoint and pass an
      aspect-ratio filter.  This rejects MSER blobs that grow in featureless
      noise regions while still catching non-circular anomalies.

  NMS — greedy IoU-based non-maximum suppression removes duplicates where
      both detectors fired on the same object.

Waterfall image layout
----------------------
  Rows (y-axis)    = pings, newest at the bottom
  Columns (x-axis) = sample index = range from transducer

Detection dict keys
-------------------
  x, y  — top-left corner of bounding box (pixels)
  w, h  — width × height (pixels)
  source — "hough" | "roi"
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
        # Hough circle params
        hough_min_dist=30,
        hough_param1=50,       # Canny high threshold
        hough_param2=80,       # accumulator threshold (higher → fewer, surer circles)
        hough_min_r=8,
        hough_max_r=80,
        # FAST params
        fast_threshold=30,
        # MSER params (ROI stage)
        mser_delta=5,
        mser_min_area=200,
        mser_max_area=3000,
        # NMS
        nms_iou=0.3,
        # column band to suppress before detection, e.g. the side-scan nadir /
        # water-column gap: (center_col, half_width). None = no masking.
        mask_band=None,
        # minimum local contrast (region mean minus surrounding-ring mean, in
        # 0-255 units) for a detection to be kept. Trims amplified noise; set
        # to 0 to disable.
        min_contrast=30,
        on_detection=None,
        on_frame=None,
    ):
        self.max_rows = max_rows
        self.detect_every = detect_every
        self.display_every = display_every
        self.nms_iou = nms_iou
        self.mask_band = mask_band
        self.min_contrast = min_contrast
        self.on_detection = on_detection
        self.on_frame = on_frame

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

    # ---- ingest ------------------------------------------------------------
    def add_ping(self, ping):
        self._rows.append(np.asarray(ping["samples_db"], dtype=np.float64))
        self._count += 1

        if self._count % self.detect_every == 0:
            objects, image = self._detect()
            self._last_objects = objects if objects else []
            if self.on_detection:
                self.on_detection(objects, ping, image)
            return objects, image

        if self.on_frame and self._count % self.display_every == 0 and len(self._rows) >= 3:
            img = self._build_display_image(list(self._rows))
            if img is not None:
                self.on_frame(self._last_objects, ping, img)

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

    def _build_display_image(self, rows):
        """Lighter path for live display — flat-field without the median pass."""
        return self._flatfield_stretch(self._stack(rows))

    # ---- detectors ---------------------------------------------------------
    def _detect(self):
        if len(self._rows) < 3:
            return [], None
        image = self._build_image(list(self._rows))
        if image.shape[0] < 3 or image.shape[1] < 3:
            return [], image

        objects = []
        objects += self._hough_detect(image)
        objects += self._roi_detect(image)
        objects = self._nms(objects, self.nms_iou)
        if self.min_contrast > 0:
            objects = [o for o in objects
                       if self._contrast(image, o) >= self.min_contrast]
        return objects, image

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
