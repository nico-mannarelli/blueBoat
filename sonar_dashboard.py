"""
sonar_dashboard.py
A polished, single-window operator dashboard for the live sonar pipeline.

This replaces the bare `cv2.imshow(colorize(image))` view with a composited,
dark-themed console that surrounds the waterfall with the context an operator
actually needs:

    +--------------------------------------------------------------+
    |  OmniScan 450 — Side-Scan            ● LIVE   ping #1234      |  header
    +-----------------------------------------+--------------------+
    |                                         |  TELEMETRY         |
    |              waterfall                  |   fix / lat / lon  |
    |          (colorized, scaled)            |   heading ⌖        |
    |     nadir line · detection boxes        +--------------------+
    |                                         |  CONTACTS (n)      |
    |                                         |   #3  12.4 m  cfar |
    |  weak ▁▂▃▄▅▆▇█ strong  (colour bar)     |   #2  ...          |
    |  |---5---10---15 m  (range ruler)       +--------------------+
    |                                         |  TRACK  (map)      |
    +-----------------------------------------+--------------------+
    |  source: scan.xtf   palette: blue   4x       Q to quit       |  footer
    +--------------------------------------------------------------+

It is pure numpy + OpenCV (no new dependencies) and is purely cosmetic: it reads
the same `gray` waterfall, `objects`, `ping`, `VehicleState` and `DetectionLog`
the pipeline already produces and draws a frame. Detection logic is untouched.

Usage:
    dash = SonarDashboard(title="OmniScan 450 — XTF Replay", palette="blue",
                          source_label="scan.xtf")
    frame = dash.render(gray, objects, ping, vehicle, log)
    cv2.imshow(dash.window, frame); cv2.waitKey(1)
"""

import math
import time

import numpy as np
import cv2

from sonar_display import colorize

from contact_export import ContactExporter ####

# ---- theme (all colours BGR) ----------------------------------------------
BG       = (24, 20, 17)
PANEL    = (40, 34, 28)
PANEL_HI = (52, 45, 37)
BORDER   = (74, 64, 54)
GRID     = (60, 52, 44)
TEXT     = (240, 238, 234)
MUTED    = (170, 162, 152)
FAINT    = (120, 114, 106)
ACCENT   = (235, 188, 92)    # warm cyan-blue — headers, highlights
GOOD     = (120, 210, 120)   # green — valid fix
WARN     = (70, 180, 245)    # orange — degraded
BAD      = (80, 80, 235)     # red — no fix
DETECT   = (90, 235, 255)    # yellow — detection boxes
HOUGH    = (235, 220, 90)    # cyan — classical/hough detections

FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTD = cv2.FONT_HERSHEY_DUPLEX

# ---- layout ----------------------------------------------------------------
W, H       = 1280, 760
PAD        = 16
HEADER_H   = 56
FOOTER_H   = 30
SIDEBAR_W  = 360


# ---- low-level drawing helpers --------------------------------------------

def _text(img, s, x, y, scale=0.46, color=TEXT, thick=1, font=FONT,
          shadow=True, right=False):
    """Anti-aliased text with a subtle shadow. `y` is the baseline.
    Returns the text width in px."""
    (tw, th), _ = cv2.getTextSize(s, font, scale, thick)
    if right:
        x = x - tw
    if shadow:
        cv2.putText(img, s, (x + 1, y + 1), font, scale, (0, 0, 0), thick,
                    cv2.LINE_AA)
    cv2.putText(img, s, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return tw


def _round_rect(img, x0, y0, x1, y1, color, thick=-1, r=10):
    """Filled or stroked rounded rectangle."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    r = int(min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    if thick < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r),
                       (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        cv2.ellipse(img, (x0 + r, y0 + r), (r, r), 180, 0, 90, color, thick, cv2.LINE_AA)
        cv2.ellipse(img, (x1 - r, y0 + r), (r, r), 270, 0, 90, color, thick, cv2.LINE_AA)
        cv2.ellipse(img, (x1 - r, y1 - r), (r, r), 0,   0, 90, color, thick, cv2.LINE_AA)
        cv2.ellipse(img, (x0 + r, y1 - r), (r, r), 90,  0, 90, color, thick, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thick, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thick, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thick, cv2.LINE_AA)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thick, cv2.LINE_AA)


def _card(img, x0, y0, x1, y1, title=None):
    """Panel with rounded border and an optional small-caps title bar."""
    _round_rect(img, x0, y0, x1, y1, PANEL, -1, r=10)
    _round_rect(img, x0, y0, x1, y1, BORDER, 1, r=10)
    if title:
        _text(img, title.upper(), x0 + 14, y0 + 22, 0.42, ACCENT, 1, FONTD)
        cv2.line(img, (x0 + 14, y0 + 30), (x1 - 14, y0 + 30), GRID, 1, cv2.LINE_AA)
        return y0 + 30
    return y0


def _dashed_v(img, x, y0, y1, color, gap=6, thick=1):
    y = y0
    while y < y1:
        cv2.line(img, (x, y), (x, min(y + gap // 2, y1)), color, thick, cv2.LINE_AA)
        y += gap


def _nice_step(span, target_ticks=6):
    """A human-friendly tick step (1/2/5 × 10ⁿ) near span/target_ticks."""
    if span <= 0:
        return 1.0
    raw = span / target_ticks
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


# ---- dashboard -------------------------------------------------------------

class SonarDashboard:
    def __init__(self, title="OmniScan 450", palette="blue", source_label="",
                 mode="LIVE", contrast=1.12, gamma=0.85, brightness=0.0, web_data_dir="sonar_web/data"):
        self.window = title
        self.title = title
        self.palette = palette
        self.source_label = source_label
        self.mode = mode
        # Display tone controls (cosmetic only — do not affect detection):
        # contrast is a multiplicative gain around mid-grey, gamma<1 lifts the
        # faint seabed, brightness is an additive shift.
        self.contrast = contrast
        self.gamma = gamma
        self.brightness = brightness
        self._t0 = time.time()
        self._last_t = time.time()
        self._fps = None
        self.exporter = ContactExporter(web_data_dir) ####
        self._exported_ids = set()  ####

    # -- helpers ------------------------------------------------------------

    def _tick_fps(self):
        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = inst if self._fps is None else 0.85 * self._fps + 0.15 * inst

    # -- public -------------------------------------------------------------

    def render(self, gray, objects, ping, vehicle, log):
        self._tick_fps()
        canvas = np.empty((H, W, 3), np.uint8)
        canvas[:] = BG

        ping = ping or {}
        objects = objects or []
        fix = vehicle.fix if vehicle is not None else None


        # creates images of all coords
        if log is not None:
            for c in log.contacts:
                if c["id"] not in self._exported_ids:
                    self.exporter.export(c, gray, palette=self.palette,
                                        contrast=self.contrast, gamma=self.gamma,
                                        brightness=self.brightness)
                    self._exported_ids.add(c["id"])



        self._header(canvas, ping, len(objects))
        wf_x1 = W - SIDEBAR_W - PAD * 2
        self._waterfall(canvas, PAD, HEADER_H + PAD, wf_x1, H - FOOTER_H - PAD,
                        gray, objects, ping)
        sx0 = wf_x1 + PAD
        self._sidebar(canvas, sx0, HEADER_H + PAD, W - PAD, H - FOOTER_H - PAD,
                      fix, ping, log)
        self._footer(canvas)
        return canvas

    # -- header -------------------------------------------------------------

    def _header(self, img, ping, n_obj):
        cv2.rectangle(img, (0, 0), (W, HEADER_H), PANEL, -1)
        cv2.line(img, (0, HEADER_H), (W, HEADER_H), BORDER, 1)

        # sonar glyph (concentric arcs) + title
        cx, cy = 26, HEADER_H // 2
        for r in (7, 12, 17):
            cv2.ellipse(img, (cx, cy), (r, r), 0, -55, 55, ACCENT, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy + 1), 2, ACCENT, -1, cv2.LINE_AA)
        _text(img, self.title, 50, HEADER_H // 2 + 6, 0.62, TEXT, 1, FONTD)

        # status pill (right)
        live = self.mode.upper().startswith("LIVE")
        dot = GOOD if live else WARN
        pill = f"{self.mode.upper()}"
        pw = cv2.getTextSize(pill, FONTD, 0.46, 1)[0][0]
        px1 = W - PAD
        px0 = px1 - pw - 44
        _round_rect(img, px0, 14, px1, HEADER_H - 14, PANEL_HI, -1, r=13)
        cv2.circle(img, (px0 + 16, HEADER_H // 2), 5, dot, -1, cv2.LINE_AA)
        _text(img, pill, px0 + 28, HEADER_H // 2 + 5, 0.46, TEXT, 1, FONTD)

        # ping + detections counter (centre-right of header)
        info = f"ping #{ping.get('ping_number', '-')}     {n_obj} detection(s)"
        _text(img, info, px0 - 24, HEADER_H // 2 + 5, 0.48, MUTED, 1, FONT,
              right=True)

    # -- waterfall ----------------------------------------------------------

    def _waterfall(self, img, x0, y0, x1, y1, gray, objects, ping):
        _round_rect(img, x0, y0, x1, y1, (16, 13, 11), -1, r=10)
        _round_rect(img, x0, y0, x1, y1, BORDER, 1, r=10)

        ruler_h = 30
        cbar_w  = 54
        ix0, iy0 = x0 + 12, y0 + 12
        ix1 = x1 - cbar_w - 8
        iy1 = y1 - ruler_h - 6

        if gray is None or gray.size == 0 or min(gray.shape[:2]) < 2:
            _text(img, "WAITING FOR PINGS...", (ix0 + ix1) // 2 - 80,
                  (iy0 + iy1) // 2, 0.6, FAINT, 1, FONTD)
            self._colorbar(img, ix1 + 16, iy0, iy1)
            return

        disp = colorize(gray, palette=self.palette, contrast=self.contrast,
                        gamma=self.gamma, brightness=self.brightness)
        ih, iw = disp.shape[:2]
        avail_w, avail_h = ix1 - ix0, iy1 - iy0
        scale = min(avail_w / iw, avail_h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        ox = ix0 + (avail_w - nw) // 2
        oy = iy0 + (avail_h - nh) // 2
        interp = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA
        resized = cv2.resize(disp, (nw, nh), interpolation=interp)
        img[oy:oy + nh, ox:ox + nw] = resized
        cv2.rectangle(img, (ox - 1, oy - 1), (ox + nw, oy + nh), GRID, 1)

        def X(c):
            return int(round(ox + c * scale))

        def Y(r):
            return int(round(oy + r * scale))

        nadir = ping.get("nadir_col", 0)
        if nadir and nadir > 0:
            _dashed_v(img, X(nadir), oy, oy + nh, (110, 110, 120), gap=8)
            _text(img, "NADIR", X(nadir) + 4, oy + 14, 0.36, (150, 150, 160))

        # detection markers — deliberately minimal so they never cover the
        # imagery the operator is trying to read. Just four thin corner brackets
        # standing a few px off the target, plus a small unobtrusive id.
        for obj in objects:
            bx0, by0 = X(obj["x"]), Y(obj["y"])
            bx1, by1 = X(obj["x"] + obj["w"]), Y(obj["y"] + obj["h"])
            by0 = max(oy, by0)
            by1 = min(oy + nh, by1)
            if by1 <= by0:
                continue
            col = HOUGH if obj.get("source") == "hough" else DETECT
            m = 3                                   # stand-off gap
            ax0, ay0, ax1, ay1 = bx0 - m, by0 - m, bx1 + m, by1 + m
            # bracket arm length scales with the box but stays short
            L = max(5, min(10, (ax1 - ax0) // 4, (ay1 - ay0) // 4))
            for (px, py, dx, dy) in ((ax0, ay0, 1, 1), (ax1, ay0, -1, 1),
                                     (ax0, ay1, 1, -1), (ax1, ay1, -1, -1)):
                cv2.line(img, (px, py), (px + dx * L, py), col, 1, cv2.LINE_AA)
                cv2.line(img, (px, py), (px, py + dy * L), col, 1, cv2.LINE_AA)
            cid = obj.get("cid")
            if cid:
                _text(img, f"{cid}", ax1 + 3, ay0 + 8, 0.34, col, 1, shadow=False)

        self._range_ruler(img, ix0, ix1, oy + nh + 14, ox, nw, scale, ping)
        self._colorbar(img, ix1 + 16, iy0, iy1)

    def _range_ruler(self, img, ix0, ix1, y, ox, nw, scale, ping):
        mpc = ping.get("mm_per_sample", 0) / 1000.0   # metres per column
        nadir = ping.get("nadir_col", 0)
        cv2.line(img, (ox, y), (ox + nw, y), BORDER, 1, cv2.LINE_AA)
        if mpc <= 0:
            _text(img, "range", (ox + ox + nw) // 2 - 16, y + 18, 0.38, FAINT)
            return

        iw = nw / scale                                # columns shown
        if nadir and nadir > 0:
            max_r = max(nadir, iw - nadir) * mpc
            step = _nice_step(max_r)
            r = 0.0
            while r <= max_r + 1e-6:
                for sign in ((-1, 1) if r > 0 else (1,)):
                    col = nadir + sign * r / mpc
                    if 0 <= col <= iw:
                        xpix = int(round(ox + col * scale))
                        cv2.line(img, (xpix, y), (xpix, y + 5), BORDER, 1, cv2.LINE_AA)
                        _text(img, f"{r:g}", xpix - 6, y + 18, 0.36, MUTED)
                r += step
            _text(img, "<- port    range (m)    stbd ->",
                  ox + nw // 2 - 76, y + 18, 0.36, FAINT)
        else:
            max_r = iw * mpc
            step = _nice_step(max_r)
            r = 0.0
            while r <= max_r + 1e-6:
                xpix = int(round(ox + (r / mpc) * scale))
                cv2.line(img, (xpix, y), (xpix, y + 5), BORDER, 1, cv2.LINE_AA)
                _text(img, f"{r:g}", xpix - 6, y + 18, 0.36, MUTED)
                r += step
            _text(img, "range from transducer (m)", ox + nw - 150, y + 18,
                  0.36, FAINT)

    def _colorbar(self, img, x, y0, y1):
        h = y1 - y0
        ramp = np.linspace(255, 0, h).astype(np.uint8).reshape(h, 1)
        bar = colorize(np.repeat(ramp, 16, axis=1), palette=self.palette)
        img[y0:y1, x:x + 16] = bar
        cv2.rectangle(img, (x, y0), (x + 16, y1), BORDER, 1)
        _text(img, "strong", x - 2, y0 - 5, 0.34, MUTED)
        _text(img, "weak", x - 2, y1 + 14, 0.34, MUTED)

    # -- sidebar ------------------------------------------------------------

    def _sidebar(self, img, x0, y0, x1, y1, fix, ping, log):
        # ---- telemetry card ----
        t_y1 = y0 + 150
        body = _card(img, x0, y0, x1, t_y1, "Telemetry")
        self._telemetry(img, x0, body, x1, t_y1, fix, ping)

        # ---- contacts card ----
        c_y0 = t_y1 + PAD
        c_y1 = c_y0 + 300
        n = len(log) if log is not None else 0
        body = _card(img, x0, c_y0, x1, c_y1, f"Contacts  ({n})")
        self._contacts(img, x0, body, x1, c_y1, log)

        # ---- track map card ----
        m_y0 = c_y1 + PAD
        body = _card(img, x0, m_y0, x1, y1, "Track")
        self._minimap(img, x0 + 12, body + 8, x1 - 12, y1 - 12, fix, log)

    def _telemetry(self, img, x0, y0, x1, y1, fix, ping):
        lx = x0 + 16
        vx = x1 - 14
        y = y0 + 24

        if fix:
            cv2.circle(img, (lx + 3, y - 4), 5, GOOD, -1, cv2.LINE_AA)
            _text(img, "GPS FIX", lx + 16, y, 0.44, GOOD, 1, FONTD)
        else:
            cv2.circle(img, (lx + 3, y - 4), 5, BAD, -1, cv2.LINE_AA)
            _text(img, "NO FIX", lx + 16, y, 0.44, BAD, 1, FONTD)

        def row(label, value, yy, vcol=TEXT):
            _text(img, label, lx, yy, 0.42, MUTED)
            _text(img, value, vx, yy, 0.46, vcol, 1, FONT, right=True)

        y += 26
        row("Latitude",  f"{fix['lat']:.6f}" if fix else "-", y)
        y += 22
        row("Longitude", f"{fix['lon']:.6f}" if fix else "-", y)
        y += 22
        hdg = fix["heading_deg"] if fix else None
        row("Heading", f"{hdg:.1f} deg" if hdg is not None else "-", y)
        y += 22
        ch = ping.get("channel_number")
        ch_s = {0: "0 - port", 1: "1 - stbd"}.get(ch, "combined swath")
        row("Channel", ch_s, y)

        # compass dial (top-right of card)
        self._compass(img, vx - 22, y0 + 14, 18, hdg)

    def _compass(self, img, cx, cy, r, hdg):
        cv2.circle(img, (cx, cy), r, PANEL_HI, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, BORDER, 1, cv2.LINE_AA)
        _text(img, "N", cx - 4, cy - r + 8, 0.32, MUTED, shadow=False)
        if hdg is None:
            return
        a = math.radians(hdg - 90)
        tip = (int(cx + (r - 4) * math.cos(a)), int(cy + (r - 4) * math.sin(a)))
        tail = (int(cx - (r - 9) * math.cos(a)), int(cy - (r - 9) * math.sin(a)))
        cv2.arrowedLine(img, tail, tip, ACCENT, 2, cv2.LINE_AA, tipLength=0.5)

    def _contacts(self, img, x0, y0, x1, y1, log):
        if log is None or len(log) == 0:
            _text(img, "no contacts yet", x0 + 16, y0 + 34, 0.42, FAINT)
            return
        rowh = 46
        y = y0 + 10
        for c in log.contacts:
            if y + rowh > y1 - 4:
                break
            scol = HOUGH if c["source"] == "hough" else DETECT
            cv2.rectangle(img, (x0 + 12, y + 4), (x0 + 15, y + rowh - 8), scol, -1)
            _text(img, f"#{c['id']:02d}", x0 + 24, y + 18, 0.5, TEXT, 1, FONTD)
            _text(img, f"{c['range_m']:.1f} m", x0 + 78, y + 18, 0.44, ACCENT)
            _text(img, f"{c['source']} {c['hits']}x", x1 - 14, y + 18, 0.4,
                  MUTED, 1, FONT, right=True)
            _text(img, f"{c['lat']:.5f}, {c['lon']:.5f}", x0 + 24, y + 36,
                  0.4, MUTED)
            if y + rowh < y1 - 4:
                cv2.line(img, (x0 + 14, y + rowh - 2), (x1 - 14, y + rowh - 2),
                         GRID, 1)
            y += rowh

    def _minimap(self, img, x0, y0, x1, y1, fix, log):
        cv2.rectangle(img, (x0, y0), (x1, y1), (16, 13, 11), -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), GRID, 1)
        pts = list(log.track) if log is not None else []
        contacts = log.contacts if log is not None else []
        allpts = pts + [(c["lat"], c["lon"]) for c in contacts]
        if not allpts:
            _text(img, "no track yet", (x0 + x1) // 2 - 40, (y0 + y1) // 2,
                  0.4, FAINT)
            return

        lats = [p[0] for p in allpts]
        lons = [p[1] for p in allpts]
        lat0, lat1 = min(lats), max(lats)
        lon0, lon1 = min(lons), max(lons)
        clat = (lat0 + lat1) / 2
        # equirectangular: scale lon by cos(lat) so aspect is roughly true
        coslat = max(0.1, math.cos(math.radians(clat)))
        spanx = max((lon1 - lon0) * coslat, 1e-6)
        spany = max(lat1 - lat0, 1e-6)
        pad = 12
        mw, mh = (x1 - x0) - 2 * pad, (y1 - y0) - 2 * pad
        s = min(mw / spanx, mh / spany)
        offx = x0 + pad + (mw - spanx * s) / 2
        offy = y0 + pad + (mh - spany * s) / 2

        def P(lat, lon):
            px = offx + (lon - lon0) * coslat * s
            py = offy + (lat1 - lat) * s     # north up
            return int(px), int(py)

        if len(pts) >= 2:
            poly = np.array([P(*p) for p in pts], np.int32)
            cv2.polylines(img, [poly], False, (150, 130, 90), 1, cv2.LINE_AA)
        for c in contacts:
            cx, cy = P(c["lat"], c["lon"])
            col = HOUGH if c["source"] == "hough" else DETECT
            cv2.circle(img, (cx, cy), 4, col, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 4, (20, 20, 20), 1, cv2.LINE_AA)
        if fix:
            vx, vy = P(fix["lat"], fix["lon"])
            a = math.radians(fix["heading_deg"] - 90)
            tip = (int(vx + 9 * math.cos(a)), int(vy + 9 * math.sin(a)))
            l = (int(vx + 6 * math.cos(a + 2.5)), int(vy + 6 * math.sin(a + 2.5)))
            r = (int(vx + 6 * math.cos(a - 2.5)), int(vy + 6 * math.sin(a - 2.5)))
            cv2.fillConvexPoly(img, np.array([tip, l, r]), GOOD, cv2.LINE_AA)

    # -- footer -------------------------------------------------------------

    def _footer(self, img):
        y = H - FOOTER_H
        cv2.rectangle(img, (0, y), (W, H), PANEL, -1)
        cv2.line(img, (0, y), (W, y), BORDER, 1)
        bits = []
        if self.source_label:
            bits.append(f"source: {self.source_label}")
        bits.append(f"palette: {self.palette}")
        if self._fps:
            bits.append(f"{self._fps:4.1f} fps")
        bits.append(f"elapsed {int(time.time() - self._t0)}s")
        _text(img, "      |      ".join(bits), 16, y + 20, 0.42, MUTED)
        _text(img, "Q to quit", W - 16, y + 20, 0.42, FAINT, 1, FONT, right=True)
