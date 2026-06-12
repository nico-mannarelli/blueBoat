"""
sonar_detect.py
Builds a rolling waterfall image from pings and runs OpenSidescan-style
detection on it (image preprocessing + MSER region detection).

This mirrors what OpenSidescan's SidescanImager + RoiDetector do, but in Python
with OpenCV in real time:

  Imager pipeline  (sidescanimager.h):
      stack pings into a Mat -> normalize -> 8-bit -> equalizeHist
      -> fastNlMeansDenoising -> blur(2x2)

  Detector pipeline (roidetector.h):
      FAST keypoints + MSER regions (+ DBSCAN clustering in the original;
      MSER alone is enough to get bounding boxes, DBSCAN can be added later)

Waterfall image layout
----------------------
  Rows (y-axis)    = pings, newest at the bottom
  Columns (x-axis) = sample index = range from transducer

MSER bboxes are (x, y, w, h) in OpenCV convention:
  x = column = sample index  →  range_mm = start_mm + x * mm_per_sample
  y = row    = ping index    →  along-track position in the waterfall buffer
"""

from collections import deque

import numpy as np
import cv2


class WaterfallDetector:
    def __init__(
        self,
        max_rows=500,          # how many pings the waterfall holds
        detect_every=50,       # run full MSER detection every N pings
        display_every=5,       # refresh the live display every N pings
        mser_delta=2,
        mser_min_area=50,
        mser_max_area=5000,
        on_detection=None,     # callback(objects, latest_ping, image) -> None
        on_frame=None,         # callback(last_objects, latest_ping, image) -> None
    ):
        self.max_rows = max_rows
        self.detect_every = detect_every
        self.display_every = display_every
        self.on_detection = on_detection
        self.on_frame = on_frame

        self._rows = deque(maxlen=max_rows)
        self._count = 0
        self._last_objects = []   # bboxes from the most recent detection run

        self._mser = cv2.MSER_create(
            delta=mser_delta,
            min_area=mser_min_area,
            max_area=mser_max_area,
        )

    # ---- ingest ------------------------------------------------------------
    def add_ping(self, ping):
        """Add one decoded ping (dict from OmniScanParser) to the waterfall."""
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

    # ---- imager (matches sidescanimager.h) ---------------------------------
    @staticmethod
    def _build_image(rows):
        # Rows may have slightly different lengths if range settings changed
        # mid-run; pad/truncate to the most common width so the Mat is rectangular.
        width = int(np.median([r.shape[0] for r in rows]))
        fixed = []
        for r in rows:
            if r.shape[0] >= width:
                fixed.append(r[:width])
            else:
                fixed.append(np.pad(r, (0, width - r.shape[0])))
        img = np.vstack(fixed).astype(np.float64)

        # normalize -> 8-bit
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

        # post-process greyscale (same order as OpenSidescan)
        img = cv2.equalizeHist(img)
        img = cv2.fastNlMeansDenoising(img, h=10)
        img = cv2.blur(img, (2, 2))
        return img

    @staticmethod
    def _build_display_image(rows):
        """Lightweight version for live display — skips denoising for speed."""
        width = int(np.median([r.shape[0] for r in rows]))
        fixed = []
        for r in rows:
            if r.shape[0] >= width:
                fixed.append(r[:width])
            else:
                fixed.append(np.pad(r, (0, width - r.shape[0])))
        img = np.vstack(fixed).astype(np.float64)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
        return cv2.equalizeHist(img)

    # ---- detector (matches roidetector.h, MSER stage) ----------------------
    def _detect(self):
        if len(self._rows) < 3:
            return [], None

        image = self._build_image(list(self._rows))
        if image.shape[0] < 3 or image.shape[1] < 3:
            return [], image
        _regions, bboxes = self._mser.detectRegions(image)

        objects = []
        for (x, y, w, h) in bboxes:
            objects.append(
                {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            )
        return objects, image
