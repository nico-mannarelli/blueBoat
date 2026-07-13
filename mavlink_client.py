"""
mavlink_client.py
Connects to the mavlink2rest WebSocket on BlueOS and maintains a
thread-safe snapshot of the vehicle's position and heading.

mavlink2rest endpoint:
    ws://192.168.2.2:6040/v1/ws/mavlink

Messages arrive as JSON with this structure (mavlink2rest v1):
    {
        "header": {"system_id": 1, "component_id": 1, "sequence": N},
        "message": {"type": "GLOBAL_POSITION_INT", "lat": ..., ...}
    }

We use:
    ATTITUDE (ID 30)            — yaw in radians -> heading_deg
    GLOBAL_POSITION_INT (ID 33) — lat/lon/alt

Heading comes from ATTITUDE not GLOBAL_POSITION_INT.hdg because
hdg is optional in MAVLink (UINT16_MAX = unknown) and less accurate.
"""

import json
import math
import threading
import time

import websocket

#from mavcon import connection


# MISSION_CURRENT.mission_state enum (MAVLink MISSION_STATE):
#   0 UNKNOWN  1 NO_MISSION  2 NOT_STARTED  3 ACTIVE  4 PAUSED  5 COMPLETE
MISSION_STATE_COMPLETE = 5


class VehicleState:
    """Thread-safe snapshot of the vehicle's latest position and attitude.

    Updated by MAVLinkClient from the background thread; read by the main
    thread inside handle_detection() for georeferencing.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._lat     = None
        self._lon     = None
        self._alt_m   = None
        self._heading = None   # degrees, 0-360, from ATTITUDE.yaw
        self._ts      = None
        self._mission_state = None   # MISSION_CURRENT.mission_state (5 = COMPLETE)
        self._mission_total = None   # MISSION_CURRENT.total (waypoint count)

    def update_gps(self, lat, lon, alt_m, timestamp_ms=None):
        with self._lock:
            self._lat   = lat
            self._lon   = lon
            self._alt_m = alt_m
            self._ts    = timestamp_ms

    def update_heading(self, heading_deg):
        with self._lock:
            self._heading = heading_deg % 360.0

    def update_mission_state(self, mission_state):
        with self._lock:
            self._mission_state = mission_state

    def update_mission_total(self, total):
        with self._lock:
            self._mission_total = total

    @property
    def mission_state(self):
        with self._lock:
            return self._mission_state

    @property
    def mission_total(self):
        with self._lock:
            return self._mission_total

    @property
    def fix(self):
        """Return a position+heading dict, or None if either is missing."""
        with self._lock:
            if self._lat is None or self._heading is None:
                return None
            return {
                "lat":         self._lat,
                "lon":         self._lon,
                "alt_m":       self._alt_m,
                "heading_deg": self._heading,
                "timestamp_ms": self._ts,
            }

    @property
    def ready(self):
        with self._lock:
            return self._lat is not None and self._heading is not None


def _unwrap(v):
    """mavlink2rest sometimes wraps scalars: {"type":"Bounded","value":N} -> N."""
    if isinstance(v, dict):
        return v.get("value", 0)
    return v


# MAV_MISSION_STATE variant names, with and without the enum's common prefix
# (Rust serde often serializes an enum by its variant NAME rather than the
# numeric value the .xml/dialect assigns it, unlike "Bounded" scalar wrappers
# such as {"type":"Bounded","value":N}). We accept int, the Bounded wrapper,
# and a bare name string so we don't silently fail to match "COMPLETE".
_MISSION_STATE_NAMES = {
    "MISSION_STATE_UNKNOWN": 0,     "UNKNOWN": 0,
    "MISSION_STATE_NO_MISSION": 1,  "NO_MISSION": 1,
    "MISSION_STATE_NOT_STARTED": 2, "NOT_STARTED": 2,
    "MISSION_STATE_ACTIVE": 3,      "ACTIVE": 3,
    "MISSION_STATE_PAUSED": 4,      "PAUSED": 4,
    "MISSION_STATE_COMPLETE": 5,    "COMPLETE": 5,
}


def _mission_state_value(v):
    """Normalize MISSION_CURRENT.mission_state to an int (or None), across
    mavlink2rest's possible encodings: a plain int, {"type":"Bounded",
    "value":N}, or a bare/enum-tagged variant-name string."""
    if v is None:
        return None
    if isinstance(v, dict):
        if "value" in v:
            v = v["value"]
        else:
            v = v.get("type", v)   # some serde enums tag by name only
    if isinstance(v, str):
        name = v.strip().upper()
        if name in _MISSION_STATE_NAMES:
            return _MISSION_STATE_NAMES[name]
        try:
            return int(name)
        except ValueError:
            return None
    if isinstance(v, (int, float)):
        return int(v)
    return None


class MAVLinkClient:
    """WebSocket client for mavlink2rest.

    Runs run_forever() in a daemon thread started by start().
    All state updates go through VehicleState which is safe to read
    from any thread.
    """

    def __init__(self, host="192.168.2.2", port=6040, state=None,
                 on_mission_complete=None):
        self.host  = host
        self.port  = port
        self.state = state or VehicleState()
        self._ws   = None
        # Called once (from this thread) when the autopilot reports the mission
        # is COMPLETE. main.py wires this to sonar.stop() to end the survey.
        self.on_mission_complete = on_mission_complete
        self._mission_complete_fired = False
        #self._logged_mission_current = False   # one-shot raw-payload debug log

    # ---- message handling --------------------------------------------------

    def _fire_mission_complete(self, reason):
        """Fire on_mission_complete exactly once, from whichever signal got
        there first (mission_state or MISSION_ITEM_REACHED)."""
        if self._mission_complete_fired:
            return
        self._mission_complete_fired = True
        print(f"[mavlink] mission COMPLETE ({reason})")
        if self.on_mission_complete:
            try:
                self.on_mission_complete()
            except Exception as e:
                print(f"[mavlink] on_mission_complete error: {e}")

    def _handle(self, packet):
        if not isinstance(packet, dict):
            return
        # mavlink2rest wraps the MAVLink message under a "message" key
        msg = packet.get("message", packet)
        msg_type = str(msg.get("type", "")).upper()

        if msg_type == "MISSION_CURRENT":
            # One-time diagnostic: print the raw field so we can see exactly
            # how mavlink2rest encodes mission_state on this autopilot
            # (int? {"type":"Bounded","value":N}? a bare variant-name string?).
            if not self._logged_mission_current:
                self._logged_mission_current = True
                print(f"[mavlink] MISSION_CURRENT sample: {msg}")

            # `total` (waypoint count) is a plain, always-populated field —
            # keep it so MISSION_ITEM_REACHED below can tell "last waypoint".
            total = _unwrap(msg.get("total", 0))
            if total:
                self.state.update_mission_total(total)

            # mission_state may be absent on some autopilots, or present in a
            # shape _unwrap() alone can't read (e.g. serialized by variant
            # name rather than a numeric wrapper) — _mission_state_value()
            # normalizes all of those. It must not be the ONLY completion
            # signal though — see MISSION_ITEM_REACHED below.
            ms = _mission_state_value(msg.get("mission_state", None))
            if ms is not None:
                self.state.update_mission_state(ms)
                if ms == MISSION_STATE_COMPLETE:
                    self._fire_mission_complete("mission_state=5")
            return

        if msg_type == "MISSION_ITEM_REACHED":
            # Firmware-independent completion signal: seq/total have been in
            # MAVLink far longer than mission_state, so this fires even on
            # autopilots that never populate MISSION_CURRENT.mission_state
            # (the bug this was added for: the survey never ended on its own
            # because mission_state was always absent).
            seq   = _unwrap(msg.get("seq", -1))
            total = self.state.mission_total
            if total and seq >= total - 1:
                self._fire_mission_complete(
                    f"reached final waypoint {seq}/{total}")
            return

        if msg_type == "GLOBAL_POSITION_INT":
            lat = _unwrap(msg.get("lat", 0))
            lon = _unwrap(msg.get("lon", 0))
            alt = _unwrap(msg.get("alt", 0))
            try:
                if abs(lat) <= 900_000_000:   # sanity: ±90° in 1e-7 deg
                    self.state.update_gps(
                        lat=lat / 1e7,
                        lon=lon / 1e7,
                        alt_m=alt / 1000.0,
                        timestamp_ms=_unwrap(msg.get("time_boot_ms", 0)),
                    )
            except TypeError:
                pass

        elif msg_type == "ATTITUDE":
            yaw = _unwrap(msg.get("yaw", 0))   # radians
            try:
                self.state.update_heading(math.degrees(yaw))
            except TypeError:
                pass

    # ---- websocket callbacks -----------------------------------------------

    def _on_open(self, ws):
        print(f"[mavlink] connected  ws://{self.host}:{self.port}/v1/ws/mavlink")

    def _on_message(self, ws, message):
        try:
            self._handle(json.loads(message))
        except (json.JSONDecodeError, Exception):
            pass

    def _on_error(self, ws, error):
        print(f"[mavlink] error: {error}")

    def _on_close(self, ws, code, msg):
        print(f"[mavlink] closed (code={code})")

    # ---- run loop ----------------------------------------------------------

    def run_forever(self, auto_reconnect=True):
        while True:
            try:
                url = f"ws://{self.host}:{self.port}/v1/ws/mavlink"
                self._ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever()

            except Exception as e:
                print(f"[mavlink] connection failed: {e}")

            if not auto_reconnect:
                break
            print("[mavlink] reconnecting in 3s...")
            time.sleep(3)

    def start(self):
        """Launch run_forever() in a background daemon thread."""
        t = threading.Thread(target=self.run_forever, daemon=True, name="mavlink")
        t.start()
        return t