"""
contact_export.py
Bridges the live detection pipeline to the web map dashboard.

Call `export_contact(...)` once per new contact (e.g. right after DetectionLog
appends one in your detect loop, or inside SonarDashboard.render once a new
cid shows up). It does two things:

  1. Saves a colorized snapshot of the waterfall around the contact as a PNG
     under <out_dir>/images/
  2. Appends/updates that contact's record in <out_dir>/contacts.json

The web page (index.html) polls contacts.json and reads images/ over a plain
static file server — no backend framework needed.

Usage in your pipeline:

    from contact_export import ContactExporter
    exporter = ContactExporter("sonar_web_data")   # or wherever you serve from

    # ... inside your detect loop, whenever DetectionLog gets a new contact:
    exporter.export(contact, gray, palette="blue")
"""

import json
import os
import threading
import time

import cv2

from sonar_display import colorize

import shared_states


class ContactExporter:
    def __init__(self, out_dir="sonar_web_data"):
        self.out_dir = out_dir
        self.img_dir = os.path.join(out_dir, "images")
        self.json_path = os.path.join(out_dir, "contacts.json")
        os.makedirs(self.img_dir, exist_ok=True)
        self._lock = threading.Lock()
        if not os.path.exists(self.json_path):
            self._write({"updated": time.time(), "contacts": []})

    def _read(self):
        try:
            with open(self.json_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"updated": time.time(), "contacts": []}

    def _write(self, data):
        tmp = self.json_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.json_path)   # atomic, avoids partial reads

    def export(self, contact, gray, palette="blue", crop_margin=40,
               contrast=1.12, gamma=0.85, brightness=0.0):
        """
        contact: dict with at least id, lat, lon, range_m, source (matches
                 DetectionLog.contacts records already used by the dashboard).
                 May also include x, y, w, h (waterfall pixel box) for a
                 cropped snapshot; falls back to the full frame if absent.
        gray:    the raw waterfall ndarray at the moment of detection.
        """
        cid = contact["id"]
        if shared_states.mission_1_png:
            img_name = f"{cid:06d}.png"
        else:
            img_name = f"2{cid:05d}.png"
        img_path = os.path.join(self.img_dir, img_name)

        disp = colorize(gray, palette=palette, contrast=contrast,
                         gamma=gamma, brightness=brightness)

        if all(k in contact for k in ("x", "y", "w", "h")):
            h_img, w_img = disp.shape[:2]
            x0 = max(0, int(contact["x"] - crop_margin))
            y0 = max(0, int(contact["y"] - crop_margin))
            x1 = min(w_img, int(contact["x"] + contact["w"] + crop_margin))
            y1 = min(h_img, int(contact["y"] + contact["h"] + crop_margin))
            if x1 > x0 and y1 > y0:
                disp = disp[y0:y1, x0:x1]

        cv2.imwrite(img_path, disp)

        record = {
            "id": cid,
            "lat": contact["lat"],
            "lon": contact["lon"],
            "range_m": contact.get("range_m"),
            "source": contact.get("source", "unknown"),
            "hits": contact.get("hits"),
            "image": f"images/{img_name}",
            "timestamp": time.time(),
        }

        with self._lock:
            data = self._read()
            data["contacts"] = [c for c in data["contacts"] if c["id"] != cid]
            data["contacts"].append(record)
            data["updated"] = time.time()
            self._write(data)

        return record
