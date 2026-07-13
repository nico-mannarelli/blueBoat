"""
detection_log.py
Accumulates georeferenced sonar detections into a de-duplicated list of
*contacts* held in memory for the duration of a mission.

Why de-duplicate?
-----------------
The detector runs on a rolling waterfall every `detect_every` pings, so the
same physical object is re-reported many times as the boat passes over it (and
again on overlapping survey lines). A raw detection stream is therefore not a
list of objects — it's a list of *sightings*. `DetectionLog` collapses sightings
that fall within `merge_radius_m` metres of each other into a single contact,
tracking how many times it was seen (a cheap confidence proxy) and keeping the
strongest sighting's range/size/score.

The result is `log.contacts`: a clean, stable list of unique lat/lon positions
that is what you actually hand off to another program (a mission planner, a
revisit-waypoint uploader, a GeoJSON map layer).

Nothing here writes to disk or talks to the vehicle — it is a pure in-memory
collector. The `to_*` helpers serialise the list in one call (used by the
end-of-survey planner hand-off), but are never invoked automatically.
"""

import math
import time

from shapely.geometry import Point, Polygon

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in metres."""
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


class DetectionLog:
    """In-memory, de-duplicated store of georeferenced contacts."""

    def __init__(self, merge_radius_m=3.0, max_track=4000):
        self.merge_radius_m = merge_radius_m
        self._contacts = []          # list[dict]
        self._next_id = 1
        self.track = []              # [(lat, lon), ...] vehicle path for the map
        self._max_track = max_track
        self._started = time.time()

    # ---- ingest ------------------------------------------------------------

    def add_fix(self, lat, lon):
        """Record a vehicle position for the track map (call once per ping)."""
        if lat is None or lon is None:
            return
        if not self.track or self.track[-1] != (lat, lon):
            self.track.append((lat, lon))
            if len(self.track) > self._max_track:
                self.track = self.track[-self._max_track:]

    def add(self, lat, lon, range_m=None, size=None, source="cfar", score=0.0,
            ping_number=None):
        """Register one georeferenced sighting; returns the (new or merged)
        contact dict. Sightings within `merge_radius_m` of an existing contact
        are merged into it rather than creating a duplicate."""
        if lat is None or lon is None:
            return None

        for c in self._contacts:
            if haversine_m(c["lat"], c["lon"], lat, lon) <= self.merge_radius_m:
                # running mean position keeps the contact stable as it is
                # re-sighted from slightly different fixes
                n = c["hits"]
                c["lat"] = (c["lat"] * n + lat) / (n + 1)
                c["lon"] = (c["lon"] * n + lon) / (n + 1)
                c["hits"] = n + 1
                c["last_ping"] = ping_number
                c["last_seen"] = time.time()
                if score >= c["score"]:           # keep the strongest sighting
                    c["score"] = score
                    c["range_m"] = range_m if range_m is not None else c["range_m"]
                    c["size"] = size if size is not None else c["size"]
                    c["source"] = source
                return c

        contact = {
            "id": self._next_id,
            "lat": lat,
            "lon": lon,
            "range_m": range_m if range_m is not None else 0.0,
            "size": size or (0, 0),
            "source": source,
            "score": score,
            "hits": 1,
            "first_ping": ping_number,
            "last_ping": ping_number,
            "first_seen": time.time(),
            "last_seen": time.time(),
        }
        self._contacts.append(contact)
        self._next_id += 1
        return contact

    # ---- read --------------------------------------------------------------

    @property
    def contacts(self):
        """All unique contacts, newest first."""
        return sorted(self._contacts, key=lambda c: c["id"], reverse=True)

    def __len__(self):
        return len(self._contacts)

    # ---- serialisation helpers (used by the planner hand-off) --------------

    def to_records(self):
        """Plain list[dict] of contacts — the simplest thing to hand to another
        program (json.dumps, a REST POST body, a DataFrame, etc.)."""
        out = []
        for c in sorted(self._contacts, key=lambda c: c["id"]):
            out.append({
                "id": c["id"],
                "lat": round(c["lat"], 7),
                "lon": round(c["lon"], 7),
                "range_m": round(c["range_m"], 1),
                "size_px": list(c["size"]),
                "source": c["source"],
                "score": round(c["score"], 2),
                "hits": c["hits"],
            })
        return out

    def to_coords(self, third=0, min_hits=1, largest=None):
        """Plain list of (lat, lon, third) tuples — the exact shape a script that
        consumes `coords = [(lat, lon, z), ...]` wants.

        third    : value placed in the 3rd slot of every tuple (altitude, a
                   flag, or just 0 — whatever the other script expects there).
        min_hits : drop contacts seen on fewer than this many pings (1 keeps
                   everything; 2 drops one-ping flickers before a revisit).
        largest  : if set, keep only the N biggest contacts by detection area
                   (size_px width*height) — e.g. largest=50 returns the 50
                   largest shapes, the ones most likely to be real objects
                   (logs, big rocks) rather than small clutter.

        The list is always returned in **detection order** (the sequence the
        contacts were first seen as the boat scanned), which follows the survey
        track — so a mission visiting them sweeps the area instead of jumping
        around. `largest` only affects *selection*, not order."""
        points = [(38.143789, -76.524991), 
        (38.143895, -76.524179), (38.143750, -76.523869), (38.141055, -76.526649), (38.141553, -76.527268)]
        fence = Polygon(points)

        points2 = [(38.143660, -76.523602), (38.143441, -76.523113), (38.140821, -76.525738), (38.140964, -76.526389)]
        fence2 = Polygon(points2)

        rows = [r for r in self.to_records() if r["hits"] >= min_hits]

        ### added to use points inside fence for pipeline
        def inside_fence(r):
            p = Point(r["lat"], r["lon"])
            return fence.contains(p) or fence2.contains(p)
        rows = [r for r in rows if inside_fence(r)]

        if largest is not None:
            def area(r):
                w, h = (list(r["size_px"]) + [0, 0])[:2]
                return float(w) * float(h)
            # pick the N largest, then restore detection (track) order
            rows = sorted(rows, key=area, reverse=True)[:largest]
            rows = sorted(rows, key=lambda r: r["id"])
        return [(r["lat"], r["lon"], third) for r in rows]

    def to_coords_literal(self, var="coords", third=0, min_hits=1, largest=None):
        """The list above rendered as a paste-ready Python assignment, e.g.
        `coords = [(37.421031, -122.17589, 0), (37.4212, -122.1761, 0)]`."""
        pts = ", ".join(f"({lat}, {lon}, {third})"
                        for lat, lon, third in
                        self.to_coords(third, min_hits, largest))
        return f"{var} = [{pts}]"

    def to_geojson(self):
        """GeoJSON FeatureCollection — drops straight into QGIS, Mission
        Planner's KML import, Leaflet, etc."""
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                 "coordinates": [r["lon"], r["lat"]]},
                    "properties": {k: v for k, v in r.items()
                                   if k not in ("lat", "lon")},
                }
                for r in self.to_records()
            ],
        }

    def to_csv(self):
        """CSV text: id,lat,lon,range_m,source,score,hits."""
        lines = ["id,lat,lon,range_m,source,score,hits"]
        for r in self.to_records():
            lines.append(f"{r['id']},{r['lat']},{r['lon']},{r['range_m']},"
                         f"{r['source']},{r['score']},{r['hits']}")
        return "\n".join(lines)
