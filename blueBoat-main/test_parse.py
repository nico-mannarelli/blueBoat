"""
test_parse.py
Builds real os_mono_profile binary packets via brping, feeds them through
OmniScanParser, and asserts the decoded dicts are correct.

Tests:
  1. Single packet — field values round-trip correctly
  2. Two packets concatenated in one feed() call — both decoded
  3. Packet split across two feed() calls — reassembled correctly
  4. dB conversion math — spot-check a known u16 value
"""

import struct
import sys

from brping.pingmessage import PingMessage
import brping.definitions as defs

from sonar_parse import OmniScanParser

MSG_ID = defs.OMNISCAN450_OS_MONO_PROFILE


def build_packet(
    ping_number=1,
    start_mm=0,
    length_mm=10_000,
    min_pwr_db=-90.0,
    max_pwr_db=-40.0,
    samples_u16=None,
):
    n = len(samples_u16) if samples_u16 is not None else 100
    if samples_u16 is None:
        samples_u16 = [32767] * n

    msg = PingMessage(MSG_ID)
    msg.ping_number = ping_number
    msg.start_mm = start_mm
    msg.length_mm = length_mm
    msg.timestamp_ms = 0
    msg.ping_hz = 800_000
    msg.gain_index = 6
    msg.num_results = n
    msg.sos_dmps = 15_000
    msg.channel_number = 0
    msg.reserved = 0
    msg.pulse_duration_sec = 0.000125
    msg.analog_gain = 1.0
    msg.max_pwr_db = max_pwr_db
    msg.min_pwr_db = min_pwr_db
    msg.transducer_heading_deg = 0.0
    msg.vehicle_heading_deg = 0.0
    msg.pwr_results = bytearray(struct.pack(f"<{n}H", *samples_u16))
    return bytes(msg.pack_msg_data())


def check(condition, label):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        sys.exit(1)


# ---- test 1: single packet -------------------------------------------------
print("Test 1: single packet round-trip")
parser = OmniScanParser()
pkt = build_packet(ping_number=1, start_mm=500, length_mm=8000, samples_u16=[32767] * 80)
results = list(parser.feed(pkt))
check(len(results) == 1, f"decoded 1 ping (got {len(results)})")
p = results[0]
check(p["ping_number"] == 1, f"ping_number=1 (got {p['ping_number']})")
check(p["start_mm"] == 500, f"start_mm=500 (got {p['start_mm']})")
check(p["length_mm"] == 8000, f"length_mm=8000 (got {p['length_mm']})")
check(p["num_results"] == 80, f"num_results=80 (got {p['num_results']})")
check(len(p["samples_db"]) == 80, f"samples_db length 80 (got {len(p['samples_db'])})")

# ---- test 2: dB conversion math --------------------------------------------
print("\nTest 2: dB conversion accuracy")
# u16=0 → min_pwr_db, u16=65535 → max_pwr_db, u16=32767 → midpoint
parser2 = OmniScanParser()
pkt2 = build_packet(
    ping_number=2,
    min_pwr_db=-90.0,
    max_pwr_db=-40.0,
    samples_u16=[0, 65535, 32767],
)
results2 = list(parser2.feed(pkt2))
check(len(results2) == 1, "decoded 1 ping")
s = results2[0]["samples_db"]
check(abs(s[0] - (-90.0)) < 0.01, f"u16=0 → -90.0 dB (got {s[0]:.4f})")
check(abs(s[1] - (-40.0)) < 0.01, f"u16=65535 → -40.0 dB (got {s[1]:.4f})")
expected_mid = -90.0 + (32767 / 65535.0) * 50.0
check(abs(s[2] - expected_mid) < 0.01, f"u16=32767 → {expected_mid:.4f} dB (got {s[2]:.4f})")

# ---- test 3: two packets concatenated -------------------------------------
print("\nTest 3: two packets in one feed() call")
parser3 = OmniScanParser()
buf = build_packet(ping_number=10, samples_u16=[1000] * 60) + \
      build_packet(ping_number=11, samples_u16=[2000] * 60)
results3 = list(parser3.feed(buf))
check(len(results3) == 2, f"decoded 2 pings (got {len(results3)})")
check(results3[0]["ping_number"] == 10, "first ping_number=10")
check(results3[1]["ping_number"] == 11, "second ping_number=11")

# ---- test 4: packet split across two feed() calls -------------------------
print("\nTest 4: packet split across two feed() calls")
parser4 = OmniScanParser()
pkt4 = build_packet(ping_number=20, samples_u16=[9999] * 50)
mid = len(pkt4) // 2
r1 = list(parser4.feed(pkt4[:mid]))
r2 = list(parser4.feed(pkt4[mid:]))
check(len(r1) == 0, "no decode on first half-packet")
check(len(r2) == 1, "decoded on second half-packet")
check(r2[0]["ping_number"] == 20, f"ping_number=20 (got {r2[0]['ping_number']})")

# ---- test 5: non-profile message followed by a real profile ---------------
print("\nTest 5: non-profile message (protocol_version) before a real profile")
# Build a valid COMMON_PROTOCOL_VERSION packet — SonarLink sends this on
# connect. brping knows the message id but its payload format produces 0
# expected bytes while payload_length > 0, which triggers the silent
# "error unpacking payload" print. Our parser must swallow it and still
# yield the profile that follows.
proto_ver_buf = bytearray([
    0x42, 0x52,          # BR
    0x04, 0x00,          # payload_length = 4
    0x01, 0x12,          # message_id = 0x1201 = COMMON_PROTOCOL_VERSION
    0x00, 0x00,          # src/dst device id
    0x01, 0x00, 0x00, 0x01,  # payload (version fields)
    0x00, 0x00,          # placeholder checksum (parser verifies; may flag ERROR but should not raise)
])
parser5 = OmniScanParser()
import io, contextlib
# brping prints on unpack failure — suppress so test output stays clean
with contextlib.redirect_stdout(io.StringIO()):
    noise = list(parser5.feed(bytes(proto_ver_buf)))
    profile_bytes = build_packet(ping_number=30, samples_u16=[1234] * 40)
    results5 = list(parser5.feed(proto_ver_buf + profile_bytes))
check(results5[-1]["ping_number"] == 30, f"profile after bad packet decoded (got {[r['ping_number'] for r in results5]})")

# ---- test 6: MAVLINK_WRAPPER GPS decode ------------------------------------
print("\nTest 6: MAVLINK_WRAPPER containing GLOBAL_POSITION_INT")
import json as _json

def build_mavlink_wrapper(payload_dict):
    """Build a valid MAVLINK_WRAPPER (ID 150) packet from a JSON-serialisable dict."""
    payload = _json.dumps(payload_dict).encode("utf-8")
    n = len(payload)
    header = struct.pack("<BBHHBB", 0x42, 0x52, n, 150, 0, 0)
    body   = header + payload
    ck     = sum(body) & 0xFFFF
    return body + struct.pack("<H", ck)

gps_payload = {
    "mavpackettype": "GLOBAL_POSITION_INT",
    "time_boot_ms":  12345,
    "lat":           407127000,    # 40.7127° N  (New York)
    "lon":          -740059000,    # -74.0059° W
    "alt":           5000,         # 5 m
    "hdg":           9000,         # 90.00°
}
parser6 = OmniScanParser()
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    results6 = list(parser6.feed(build_mavlink_wrapper(gps_payload)))
check(len(results6) == 1, f"decoded 1 GPS packet (got {len(results6)})")
gps = results6[0]
check(gps["type"] == "gps", f"type='gps' (got {gps['type']})")
check(abs(gps["lat"]  -  40.7127) < 0.0001, f"lat≈40.7127 (got {gps['lat']:.6f})")
check(abs(gps["lon"]  - -74.0059) < 0.0001, f"lon≈-74.0059 (got {gps['lon']:.6f})")
check(abs(gps["alt_m"] - 5.0)     < 0.001,  f"alt_m=5.0 (got {gps['alt_m']})")
check(abs(gps["heading_deg"] - 90.0) < 0.01, f"heading=90.0° (got {gps['heading_deg']})")

print("\nTest 7: MAVLINK_WRAPPER mixed with ping packets")
parser7 = OmniScanParser()
buf7 = (
    build_mavlink_wrapper(gps_payload)
    + build_packet(ping_number=99, samples_u16=[32767] * 30)
    + build_mavlink_wrapper(gps_payload)
)
with contextlib.redirect_stdout(io.StringIO()):
    results7 = list(parser7.feed(buf7))
types7 = [r["type"] for r in results7]
check(types7 == ["gps", "ping", "gps"], f"gps/ping/gps order (got {types7})")
check(results7[1]["ping_number"] == 99, "ping_number=99")

print("\nAll parser tests passed.")
