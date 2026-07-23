"""
contact_export.py — per-contact export + real-time map upload.

Two independent jobs:
  export()          local record: snapshot PNG under <out_dir>/images/ +
                    contacts.json. Called by the dashboard per contact.
  upload_contact()  the mission-2 MAP UPLOAD. Called at DETECTION time (from
                    main.handle_detection) so the crop is cut from the exact
                    frame the contact was detected in. Deduped per contact id.
                    POSTs low_res_img (224x224 classifier crop) + high_res_img
                    (colorized snapshot) + bbox + lat/lon to the CommandCenter
                    map endpoint (CONTACT_URL), which classifies the low-res
                    crop and draws the label + box on the high-res image.

Uploads run on a background thread so the sonar pipeline NEVER waits on the
network; a dead map server costs markers, not the mission.
"""

import json
import os
import queue
import threading
import time

import cv2
import requests

from sonar_display import colorize
from crop_saver import make_crop

import shared_states

# ---- map upload config ------------------------------------------------------
CONTACT_URL        = "http://10.107.30.63:30932/marker/boat"  # CommandCenter map endpoint
CONTACT_TIMEOUT_S  = 60.0     # classify + forward on the server side
CROP_MARGIN        = 40       # px of context around the contact in the snapshot


class ContactExporter:
    def __init__(self, out_dir="sonar_web_data"):
        self.out_dir = out_dir
        self.img_dir = os.path.join(out_dir, "images")
        self.json_path = os.path.join(out_dir, "contacts.json")
        os.makedirs(self.img_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._clf_q = queue.Queue()
        self._clf_thread = None
        self._clf_failures = 0
        self._uploaded = set()   # contact ids already uploaded (dedup)
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

    def export(self, contact, gray, palette="blue", crop_margin=CROP_MARGIN,
               contrast=1.12, gamma=0.85, brightness=0.0):
        """Local record only: save a colorized snapshot PNG under images/ and
        upsert the contact into contacts.json. Does NOT upload — the map
        upload is upload_contact(), fired at detection time. Called by the
        dashboard once per contact."""
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

    # ---- real-time map upload (mission 2, at detection time) ----------------

    def upload_contact(self, gray, cid, lat, lon, box):
        """Cut the classifier crop + a colorized snapshot from `gray` (the
        exact frame the contact was detected in) and queue a map upload.
        `box` = (x, y, w, h) in `gray` pixel coords. One upload per cid.

        Called from main.handle_detection during mission 2, NOT from the
        dashboard — so the crop always matches the detection frame."""
        if cid in self._uploaded:
            return
        x, y, w, h = (int(v) for v in box)

        # low_res_img: the 224x224 gray classifier crop (library format)
        crop = make_crop(gray, (x, y, w, h))
        if crop is None:
            return
        ok, crop_buf = cv2.imencode(".png", crop)
        if not ok:
            return

        # high_res_img: colorized display snapshot cropped around the contact
        disp = colorize(gray, palette="blue")
        h_img, w_img = disp.shape[:2]
        x0, y0 = max(0, x - CROP_MARGIN), max(0, y - CROP_MARGIN)
        x1, y1 = min(w_img, x + w + CROP_MARGIN), min(h_img, y + h + CROP_MARGIN)
        sx0, sy0 = 0, 0
        if x1 > x0 and y1 > y0:
            disp = disp[y0:y1, x0:x1]
            sx0, sy0 = x0, y0
        ok2, snap_buf = cv2.imencode(".png", disp)
        if not ok2:
            return

        bbox = (x - sx0, y - sy0, w, h)   # detection box in snapshot coords
        self._uploaded.add(cid)
        self._enqueue(crop_buf.tobytes(), snap_buf.tobytes(),
                      cid, lat, lon, bbox)

    def _enqueue(self, crop_png, snap_png, cid, lat, lon, bbox):
        """Queue one contact for upload; starts the worker thread on first use."""
        if self._clf_thread is None:
            self._clf_thread = threading.Thread(
                target=self._worker, daemon=True, name="map-upload")
            self._clf_thread.start()
        self._clf_q.put((crop_png, snap_png, cid, lat, lon, bbox))

    def _worker(self):
        """Forever: pop a contact off the queue and POST it to the
        CommandCenter map endpoint. The server classifies low_res_img and
        draws the label + bbox ("[x,y,w,h]") on high_res_img for the map."""
        while True:
            crop_png, snap_png, cid, lat, lon, bbox = self._clf_q.get()
            bx, by, bw, bh = bbox
            try:
                r = requests.post(
                    CONTACT_URL,
                    data={"lat": lat, "lon": lon,
                          "bbox": f"[{bx},{by},{bw},{bh}]"},
                    files=[("low_res_img", (f"crop_{cid}.png", crop_png,
                                            "image/png")),
                           ("high_res_img", (f"snap_{cid}.png", snap_png,
                                             "image/png"))],
                    timeout=CONTACT_TIMEOUT_S)
                self._clf_failures = 0
                print(f"[export] contact #{cid} sent -> HTTP {r.status_code}")
            except Exception:
                self._clf_failures += 1
                if self._clf_failures == 3:
                    print("[export] map endpoint unreachable — real-time "
                          "contact uploads off")
