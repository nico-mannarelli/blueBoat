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
"""

from collections import deque

import numpy as np
import cv2


class WaterfallDetector:
    def __init__(
        self,
        max_rows=500,          # how many pings the waterfall holds
        detect_every=50,       # run detection once per this many pings
        mser_delta=2,
        mser_min_area=50,
        mser_max_area=5000,
        on_detection=None,     # callback(list_of_objects, latest_ping) -> None
    ):
        self.max_rows = max_rows
        self.detect_every = detect_every
        self.on_detection = on_detection

        self._rows = deque(maxlen=max_rows)
        self._count = 0

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
            if self.on_detection:
                self.on_detection(objects, ping, image)
            return objects, image
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
