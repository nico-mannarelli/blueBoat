"""
contact_export.py — per-contact export: local web-map files + edge upload.

DATA IN:  one contact + the waterfall frame per new detection
          (sonar_dashboard.render calls export() once per new cid)
DATA OUT: local disk — snapshot PNG under <out_dir>/images/
          network (mission 2 only) — low_res_img (224x224 classifier crop) +
          high_res_img (colorized snapshot) + bbox + lat/lon POSTed to the
          CommandCenter map endpoint (CONTACT_URL), which classifies the
          low-res crop and draws the label + box on the high-res image
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

# ---- real-time contact upload (mission 2) -----------------------------------
# Uploads run on a background thread so the sonar pipeline NEVER waits on
# the network; a dead edge server costs markers, not the mission (api.py
# batch upload afterwards is the manual backup).
CONTACT_URL        = "http://10.107.30.63:30932/marker/boat"  # CommandCenter map endpoint
CONTACT_TIMEOUT_S  = 15.0     # classify + forward on the server side
CLASSIFY_MISSION_2_ONLY = True   # False = also send mission-1 survey contacts


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
        """IN: one contact dict + the waterfall frame it was detected in.
        OUT: snapshot PNG + contacts.json record on local disk; in mission 2
        also queues crop + snapshot + bbox for upload to the edge server."""
        cid = contact["id"]
        if shared_states.mission_1_png:
            img_name = f"{cid:06d}.png"
        else:
            img_name = f"2{cid:05d}.png"
        img_path = os.path.join(self.img_dir, img_name)

        disp = colorize(gray, palette=palette, contrast=contrast,
                         gamma=gamma, brightness=brightness)

        sx0 = sy0 = 0    # snapshot crop origin, for bbox-in-snapshot coords
        if all(k in contact for k in ("x", "y", "w", "h")):
            h_img, w_img = disp.shape[:2]
            x0 = max(0, int(contact["x"] - crop_margin))
            y0 = max(0, int(contact["y"] - crop_margin))
            x1 = min(w_img, int(contact["x"] + contact["w"] + crop_margin))
            y1 = min(h_img, int(contact["y"] + contact["h"] + crop_margin))
            if x1 > x0 and y1 > y0:
                disp = disp[y0:y1, x0:x1]
                sx0, sy0 = x0, y0

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

        # mission 2: hand the uploader the classifier-format crop (raw gray,
        # same prep the library was built from) + the snapshot path + the
        # detection box in snapshot coords (for box+label drawing server-side)
        in_mission_2 = not shared_states.mission_1_png
        if (in_mission_2 or not CLASSIFY_MISSION_2_ONLY) \
                and all(k in contact for k in ("x", "y", "w", "h")):
            crop = make_crop(gray, (contact["x"], contact["y"],
                                    contact["w"], contact["h"]))
            if crop is not None:
                ok, buf = cv2.imencode(".png", crop)
                if ok:
                    bbox = (int(contact["x"]) - sx0, int(contact["y"]) - sy0,
                            int(contact["w"]), int(contact["h"]))
                    self._enqueue(buf.tobytes(), cid, contact["lat"],
                                  contact["lon"], img_path,
                                  "2" if in_mission_2 else "1", bbox)

        return record

    # ---- background classify + marker uploader ------------------------------

    def _enqueue(self, crop_png, cid, lat, lon, img_path, mission, bbox):
        """Queue one contact for upload; starts the worker thread on first use."""
        if self._clf_thread is None:
            self._clf_thread = threading.Thread(
                target=self._worker, daemon=True, name="classify-upload")
            self._clf_thread.start()
        self._clf_q.put((crop_png, cid, lat, lon, img_path, mission, bbox))

    def _worker(self):
        """Forever: pop a contact off the queue, POST it to the CommandCenter
        map endpoint (CONTACT_URL). The server classifies low_res_img and
        draws the label + bbox on high_res_img for the map.
            low_res_img  = 224x224 gray classifier crop
            high_res_img = colorized display snapshot
            bbox         = "[x,y,w,h]" in high_res_img pixel coords
        """
        while True:
            crop_png, cid, lat, lon, img_path, mission, bbox = self._clf_q.get()
            bx, by, bw, bh = bbox
            try:
                with open(img_path, "rb") as f:
                    r = requests.post(
                        CONTACT_URL,
                        data={"lat": lat, "lon": lon,
                              "bbox": f"[{bx},{by},{bw},{bh}]"},
                        files=[("low_res_img", (f"crop_{cid}.png", crop_png,
                                                "image/png")),
                               ("high_res_img", (os.path.basename(img_path), f,
                                                 "image/png"))],
                        timeout=CONTACT_TIMEOUT_S)
                self._clf_failures = 0
                print(f"[export] contact #{cid} sent -> HTTP {r.status_code}")
            except Exception:
                self._clf_failures += 1
                if self._clf_failures == 3:
                    print("[export] map endpoint unreachable — real-time "
                          "contact uploads off (use api.py after the mission)")
