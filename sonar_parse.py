"""
sonar_parse.py
Feeds raw bytes into brping's PingParser and yields decoded packets.

Two packet types are produced:

  {"type": "ping",  "ping_number": ..., "samples_db": [...], ...}
  {"type": "gps",   "lat": ..., "lon": ..., "alt_m": ...,
                    "heading_deg": ..., "timestamp_ms": ...}

os_mono_profile (ID 2198) — brping decodes this natively.

MAVLINK_WRAPPER (ID 150) — not in brping's payload_dict, so brping
  returns ERROR and prints "Unknown message: 150". We intercept ERROR
  returns, manually verify the checksum, and JSON-decode the payload.
  SonarLink wraps whatever MAVLink the vehicle is broadcasting; we
  extract GLOBAL_POSITION_INT to get GPS + heading.

Note: brping will print "Unknown message:  150" to stdout for every
MAVLINK_WRAPPER packet (~GPS rate). That is expected and harmless.
"""

import json
import struct

from brping.pingmessage import PingParser, PingMessage
from brping import definitions

OS_MONO_PROFILE    = definitions.OMNISCAN450_OS_MONO_PROFILE  # 2198
MAVLINK_WRAPPER_ID = 150
_HDR               = PingMessage.headerLength                 # 8


class OmniScanParser:
    def __init__(self):
        self._parser = PingParser()

    def feed(self, raw_bytes):
        """Generator: yield a dict for every decoded packet.

        Ping keys:  type, ping_number, start_mm, length_mm, num_results,
                    vehicle_heading_deg, min_pwr_db, max_pwr_db,
                    samples_db, mm_per_sample, timestamp_ms
        GPS keys:   type, lat, lon, alt_m, heading_deg, timestamp_ms
        """
        for byte in raw_bytes:
            result = self._parser.parse_byte(byte)
            msg    = self._parser.rx_msg

            if result == PingParser.NEW_MESSAGE:
                if msg.message_id != OS_MONO_PROFILE:
                    continue
                # Guard: brping leaves pwr_results as int 0 when the fixed
                # payload fails to unpack (e.g. other handshake messages).
                if not isinstance(msg.pwr_results, (bytes, bytearray)):
                    continue
                decoded = self._decode_profile(msg)
                if decoded is not None:
                    yield decoded

            elif result == PingParser.ERROR:
                # brping returns ERROR for unknown IDs (checksum field is never
                # set so verify_checksum() always fails). Intercept MAVLINK_WRAPPER.
                if (msg is not None
                        and msg.message_id == MAVLINK_WRAPPER_ID
                        and msg.msg_data is not None):
                    gps = self._decode_mavlink_wrapper(msg)
                    if gps is not None:
                        yield gps

    # ---- os_mono_profile ---------------------------------------------------

    @staticmethod
    def _decode_profile(msg):
        """Convert a parsed os_mono_profile into a usable dict.

        pwr_results is a bytearray of u16 values scaled between
        min_pwr_db and max_pwr_db. We map them to actual dB.
        """
        raw = bytes(msg.pwr_results)
        n   = len(raw) // 2
        if n == 0:
            return None

        samples_u16 = struct.unpack(f"<{n}H", raw)
        min_db      = float(msg.min_pwr_db)
        max_db      = float(msg.max_pwr_db)
        span        = max_db - min_db
        samples_db  = [min_db + (v / 65535.0) * span for v in samples_u16]

        return {
            "type":                "ping",
            "ping_number":         msg.ping_number,
            "start_mm":            msg.start_mm,
            "length_mm":           msg.length_mm,
            "num_results":         n,
            "timestamp_ms":        msg.timestamp_ms,
            "vehicle_heading_deg": float(msg.vehicle_heading_deg),
            "transducer_heading_deg": float(msg.transducer_heading_deg),
            "min_pwr_db":          min_db,
            "max_pwr_db":          max_db,
            "samples_db":          samples_db,
            "mm_per_sample":       msg.length_mm / n,
        }

    # ---- MAVLINK_WRAPPER ---------------------------------------------------

    @staticmethod
    def _decode_mavlink_wrapper(msg):
        """Extract a GPS fix from a MAVLINK_WRAPPER packet, or return None.

        brping never verifies the checksum for unknown IDs so we do it here.
        The payload is a variable-length JSON string. We only act on
        GLOBAL_POSITION_INT (MAVLink ID 33) which carries lat/lon/alt/heading.
        """
        raw = bytes(msg.msg_data)

        # Manual checksum verification (bytes 0..8+N-1, stored at 8+N..8+N+1)
        end     = _HDR + msg.payload_length
        expected = sum(raw[:end]) & 0xFFFF
        stored   = struct.unpack("<H", raw[end:end + 2])[0]
        if expected != stored:
            return None

        try:
            data = json.loads(raw[_HDR:end].decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, ValueError):
            return None

        # pymavlink serialises the type under "mavpackettype"; other
        # implementations use "type" or "message_type"
        msg_type = str(
            data.get("mavpackettype")
            or data.get("type")
            or data.get("message_type")
            or ""
        ).upper()

        if "GLOBAL_POSITION_INT" not in msg_type:
            return None

        lat = data.get("lat") or data.get("latitude")
        lon = data.get("lon") or data.get("longitude")
        if lat is None or lon is None:
            return None

        # MAVLink GLOBAL_POSITION_INT: lat/lon in 1e-7 deg, alt in mm, hdg in cdeg
        return {
            "type":        "gps",
            "lat":         lat / 1e7,
            "lon":         lon / 1e7,
            "alt_m":       data.get("alt", 0) / 1000.0,
            "heading_deg": data.get("hdg", data.get("heading", 0)) / 100.0,
            "timestamp_ms": data.get("time_boot_ms", 0),
        }
