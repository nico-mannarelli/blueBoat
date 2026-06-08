"""
sonar_parse.py
Feeds raw bytes into brping's PingParser and yields decoded OmniScan profiles.

brping does the hard part (finding BR magic bytes, reading the header,
verifying the checksum). We only add: feed it bytes one at a time, and when a
complete os_mono_profile (id 2198) arrives, pull out the fields we care about
and convert pwr_results into actual dB values.
"""

import struct

from brping.pingmessage import PingParser
from brping import definitions

OS_MONO_PROFILE = definitions.OMNISCAN450_OS_MONO_PROFILE  # 2198


class OmniScanParser:
    def __init__(self):
        self._parser = PingParser()

    def feed(self, raw_bytes):
        """Generator: yield a dict for every complete os_mono_profile packet.

        Usage:
            for ping in parser.feed(chunk):
                ...
        """
        for byte in raw_bytes:
            if self._parser.parse_byte(byte) == PingParser.NEW_MESSAGE:
                msg = self._parser.rx_msg
                if msg.message_id == OS_MONO_PROFILE:
                    decoded = self._decode_profile(msg)
                    if decoded is not None:
                        yield decoded

    @staticmethod
    def _decode_profile(msg):
        """Pull the useful fields off the parsed message.

        pwr_results comes back as a raw bytearray of u16 values. Each value is a
        scaled intensity; we map it from [0, 65535] onto the packet's own
        [min_pwr_db, max_pwr_db] range so downstream code works in real dB.
        """
        raw = bytes(msg.pwr_results)
        n = len(raw) // 2
        if n == 0:
            return None

        samples_u16 = struct.unpack(f"<{n}H", raw)

        min_db = float(msg.min_pwr_db)
        max_db = float(msg.max_pwr_db)
        span = max_db - min_db
        samples_db = [min_db + (v / 65535.0) * span for v in samples_u16]

        return {
            "ping_number": msg.ping_number,
            "start_mm": msg.start_mm,
            "length_mm": msg.length_mm,
            "num_results": n,
            "vehicle_heading_deg": float(msg.vehicle_heading),
            "min_pwr_db": min_db,
            "max_pwr_db": max_db,
            "samples_db": samples_db,        # list[float], length n
            "mm_per_sample": msg.length_mm / n,
        }
