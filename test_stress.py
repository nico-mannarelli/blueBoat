"""
test_stress.py
Adversarial and stress tests for the full pipeline.
Tries to crash or corrupt: OmniScanParser, VehicleState, MAVLinkClient._handle,
georeference, and WaterfallDetector.
"""

import contextlib
import io
import json
import math
import os
import random
import struct
import sys
import threading
import time

from brping.pingmessage import PingMessage
import brping.definitions as defs

from sonar_parse import OmniScanParser
from sonar_detect import WaterfallDetector
from mavlink_client import VehicleState, MAVLinkClient

PASS = 0
FAIL = 0

def check(cond, label, note=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {label}")
        PASS += 1
    else:
        extra = f"  ({note})" if note else ""
        print(f"  FAIL  {label}{extra}")
        FAIL += 1

def silent(fn, *a, **kw):
    """Call fn, suppressing stdout (brping prints on bad packets)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ── helpers ──────────────────────────────────────────────────────────────────

def build_ping(ping_number=1, num_results=100, start_mm=0, length_mm=10_000,
               min_pwr=-90.0, max_pwr=-40.0, samples=None):
    if samples is None:
        samples = [32767] * num_results
    msg = PingMessage(defs.OMNISCAN450_OS_MONO_PROFILE)
    msg.ping_number          = ping_number
    msg.start_mm             = start_mm
    msg.length_mm            = length_mm
    msg.timestamp_ms         = 0
    msg.ping_hz              = 450_000
    msg.gain_index           = 6
    msg.num_results          = num_results
    msg.sos_dmps             = 15_000
    msg.channel_number       = 0
    msg.reserved             = 0
    msg.pulse_duration_sec   = 0.000125
    msg.analog_gain          = 1.0
    msg.max_pwr_db           = max_pwr
    msg.min_pwr_db           = min_pwr
    msg.transducer_heading_deg = 0.0
    msg.vehicle_heading_deg  = 0.0
    msg.pwr_results          = bytearray(struct.pack(f"<{num_results}H", *samples))
    return bytes(msg.pack_msg_data())

def build_wrapper(payload_dict):
    payload = json.dumps(payload_dict).encode()
    n = len(payload)
    header = struct.pack("<BBHHBB", 0x42, 0x52, n, 150, 0, 0)
    body   = header + payload
    ck     = sum(body) & 0xFFFF
    return body + struct.pack("<H", ck)


# ═══════════════════════════════════════════════════════════════════
# 1. PARSER — random / garbage bytes
# ═══════════════════════════════════════════════════════════════════
print("\n── 1. Parser: garbage bytes ─────────────────────────────────")

p = OmniScanParser()
try:
    results = silent(list, p.feed(bytes(range(256))))
    check(True, "256 ascending bytes — no crash")
except Exception as e:
    check(False, "256 ascending bytes — no crash", str(e))

p2 = OmniScanParser()
rng = random.Random(42)
try:
    garbage = bytes(rng.randint(0, 255) for _ in range(10_000))
    results = silent(list, p2.feed(garbage))
    check(True, f"10 000 random bytes — no crash ({len(results)} packets decoded)")
except Exception as e:
    check(False, "10 000 random bytes — no crash", str(e))

# Only 0x00 bytes
p3 = OmniScanParser()
try:
    silent(list, p3.feed(bytes(4096)))
    check(True, "4096 null bytes — no crash")
except Exception as e:
    check(False, "4096 null bytes — no crash", str(e))

# Only 0xFF bytes
p4 = OmniScanParser()
try:
    silent(list, p4.feed(bytes([0xFF] * 4096)))
    check(True, "4096 0xFF bytes — no crash")
except Exception as e:
    check(False, "4096 0xFF bytes — no crash", str(e))


# ═══════════════════════════════════════════════════════════════════
# 2. PARSER — truncated / partial packets
# ═══════════════════════════════════════════════════════════════════
print("\n── 2. Parser: truncated packets ─────────────────────────────")

# Feed only the header, never finish the packet
p5 = OmniScanParser()
try:
    pkt = build_ping(ping_number=5)
    silent(list, p5.feed(pkt[:4]))   # half the header
    r = silent(list, p5.feed(pkt[4:8]))  # rest of header, no payload
    check(True, "header-only feed — no crash")
except Exception as e:
    check(False, "header-only feed — no crash", str(e))

# 1 byte at a time
p6 = OmniScanParser()
try:
    pkt = build_ping(ping_number=6)
    results = []
    for b in pkt:
        results += silent(list, p6.feed(bytes([b])))
    check(len(results) == 1, f"byte-by-byte feed yields exactly 1 ping (got {len(results)})")
except Exception as e:
    check(False, "byte-by-byte feed — no crash", str(e))

# Valid packet followed immediately by truncated garbage with BR magic
p7 = OmniScanParser()
try:
    pkt = build_ping(ping_number=7)
    tail = bytes([0x42, 0x52, 0xFF, 0xFF])  # BR magic + huge length, no payload
    r = silent(list, p7.feed(pkt + tail))
    check(len(r) == 1, f"valid + truncated tail — still gets 1 ping (got {len(r)})")
except Exception as e:
    check(False, "valid + truncated tail — no crash", str(e))


# ═══════════════════════════════════════════════════════════════════
# 3. PARSER — zero / edge-case sample counts
# ═══════════════════════════════════════════════════════════════════
print("\n── 3. Parser: zero / edge-case samples ──────────────────────")

# num_results=0: brping can't build a 0-result packet (fixed payload size mismatch),
# so we can't test via wire format. Verify _decode_profile handles it gracefully instead.
from sonar_parse import OmniScanParser as _P
import sonar_parse as _sp
class _FakeMsg:
    message_id = _sp.OS_MONO_PROFILE
    ping_number = 8; start_mm = 0; length_mm = 1000; timestamp_ms = 0
    ping_hz = 450_000; gain_index = 6; num_results = 0; sos_dmps = 15_000
    channel_number = 0; reserved = 0; pulse_duration_sec = 0.000125
    analog_gain = 1.0; max_pwr_db = -40.0; min_pwr_db = -90.0
    transducer_heading_deg = 0.0; vehicle_heading_deg = 0.0
    pwr_results = bytearray()
try:
    r8 = _sp.OmniScanParser._decode_profile(_FakeMsg())
    check(r8 is None, "num_results=0 → _decode_profile returns None")
except Exception as e:
    check(False, "num_results=0 — no crash", str(e))

# min_pwr_db == max_pwr_db (zero span → all samples map to min_pwr_db)
p9 = OmniScanParser()
try:
    pkt = build_ping(ping_number=9, num_results=3, min_pwr=-60.0, max_pwr=-60.0,
                     samples=[0, 32767, 65535])
    r = silent(list, p9.feed(pkt))
    check(len(r) == 1, "min==max_pwr_db — decoded (no div-by-zero crash)")
    if r:
        check(all(abs(v - (-60.0)) < 0.01 for v in r[0]["samples_db"]),
              "min==max_pwr_db — all samples equal min_pwr")
except Exception as e:
    check(False, "min==max_pwr_db — no crash", str(e))

# 1 sample
p10 = OmniScanParser()
try:
    pkt = build_ping(ping_number=10, num_results=1, samples=[65535])
    r = silent(list, p10.feed(pkt))
    check(len(r) == 1 and len(r[0]["samples_db"]) == 1, "1-sample ping decoded")
except Exception as e:
    check(False, "1-sample ping — no crash", str(e))

# Large batch of packets
p11 = OmniScanParser()
try:
    buf = b"".join(build_ping(ping_number=i) for i in range(500))
    r = silent(list, p11.feed(buf))
    check(len(r) == 500, f"500 concatenated pings decoded (got {len(r)})")
except Exception as e:
    check(False, "500 concatenated pings — no crash", str(e))


# ═══════════════════════════════════════════════════════════════════
# 4. PARSER — MAVLINK_WRAPPER bad inputs
# ═══════════════════════════════════════════════════════════════════
print("\n── 4. Parser: MAVLINK_WRAPPER bad inputs ────────────────────")

def feed_wrapper_raw(raw_bytes):
    p = OmniScanParser()
    return silent(list, p.feed(raw_bytes))

# Bad checksum
gps_dict = {"mavpackettype": "GLOBAL_POSITION_INT", "lat": 407127000,
            "lon": -740059000, "alt": 5000, "hdg": 9000, "time_boot_ms": 0}
good_pkt = build_wrapper(gps_dict)
bad_ck = good_pkt[:-2] + struct.pack("<H", 0xDEAD)
try:
    r = feed_wrapper_raw(bad_ck)
    check(len(r) == 0, "MAVLINK_WRAPPER bad checksum → silently dropped")
except Exception as e:
    check(False, "MAVLINK_WRAPPER bad checksum — no crash", str(e))

# Non-JSON payload with valid checksum
raw_payload = b"not json at all!!!"
hdr = struct.pack("<BBHHBB", 0x42, 0x52, len(raw_payload), 150, 0, 0)
body = hdr + raw_payload
ck = sum(body) & 0xFFFF
bad_json_pkt = body + struct.pack("<H", ck)
try:
    r = feed_wrapper_raw(bad_json_pkt)
    check(len(r) == 0, "MAVLINK_WRAPPER non-JSON payload → silently dropped")
except Exception as e:
    check(False, "MAVLINK_WRAPPER non-JSON payload — no crash", str(e))

# JSON but wrong message type
try:
    r = feed_wrapper_raw(build_wrapper({"mavpackettype": "HEARTBEAT", "type": 0}))
    check(len(r) == 0, "MAVLINK_WRAPPER HEARTBEAT type → silently dropped")
except Exception as e:
    check(False, "MAVLINK_WRAPPER HEARTBEAT — no crash", str(e))

# GPS with missing lat
try:
    r = feed_wrapper_raw(build_wrapper({"mavpackettype": "GLOBAL_POSITION_INT", "lon": -740059000}))
    check(len(r) == 0, "MAVLINK_WRAPPER missing lat → silently dropped")
except Exception as e:
    check(False, "MAVLINK_WRAPPER missing lat — no crash", str(e))

# GPS with lat=0, lon=0 (falsy but valid)
try:
    r = feed_wrapper_raw(build_wrapper({"mavpackettype": "GLOBAL_POSITION_INT",
                                        "lat": 0, "lon": 0, "alt": 0, "hdg": 0}))
    # lat=0, lon=0 is falsy in Python — check we handle it
    # Note: sonar_parse uses `data.get("lat") or data.get("latitude")` which treats 0 as None!
    check(True, "MAVLINK_WRAPPER lat=0/lon=0 — no crash (behavior noted)")
except Exception as e:
    check(False, "MAVLINK_WRAPPER lat=0/lon=0 — no crash", str(e))

# Empty JSON object
try:
    r = feed_wrapper_raw(build_wrapper({}))
    check(len(r) == 0, "MAVLINK_WRAPPER empty JSON → silently dropped")
except Exception as e:
    check(False, "MAVLINK_WRAPPER empty JSON — no crash", str(e))

# Garbage inside valid packet stream: valid GPS, then 100 garbage bytes, then valid ping
try:
    garbage = bytes(rng.randint(0, 255) for _ in range(100))
    buf = build_wrapper(gps_dict) + garbage + build_ping(ping_number=42)
    r = silent(list, OmniScanParser().feed(buf))
    types = [x["type"] for x in r]
    check("ping" in types, f"garbage between packets — ping still decoded (types={types})")
except Exception as e:
    check(False, "garbage between packets — no crash", str(e))


# ═══════════════════════════════════════════════════════════════════
# 5. VehicleState — thread safety
# ═══════════════════════════════════════════════════════════════════
print("\n── 5. VehicleState: thread safety ───────────────────────────")

vs = VehicleState()
errors = []

def writer_gps(n):
    for i in range(n):
        vs.update_gps(lat=float(i)/1e6, lon=float(-i)/1e6, alt_m=float(i)/100)

def writer_hdg(n):
    for i in range(n):
        vs.update_heading(float(i % 360))

def reader(n):
    for _ in range(n):
        try:
            _ = vs.fix
            _ = vs.ready
        except Exception as e:
            errors.append(str(e))

threads = (
    [threading.Thread(target=writer_gps, args=(5000,)) for _ in range(4)]
    + [threading.Thread(target=writer_hdg, args=(5000,)) for _ in range(4)]
    + [threading.Thread(target=reader,     args=(5000,)) for _ in range(8)]
)
for t in threads: t.start()
for t in threads: t.join()
check(len(errors) == 0, f"16-thread concurrent read/write — no exceptions (errs={errors})")

# fix before any data
vs2 = VehicleState()
check(vs2.fix is None, "fix() before any data → None")
check(vs2.ready is False, "ready before any data → False")

# fix with GPS but no heading
vs3 = VehicleState()
vs3.update_gps(lat=40.0, lon=-70.0, alt_m=0)
check(vs3.fix is None, "fix() with GPS but no heading → None")

# fix with heading but no GPS
vs4 = VehicleState()
vs4.update_heading(90.0)
check(vs4.fix is None, "fix() with heading but no GPS → None")

# heading wraps correctly
vs5 = VehicleState()
vs5.update_heading(370.0)
vs5.update_gps(lat=1.0, lon=1.0, alt_m=0)
check(abs(vs5.fix["heading_deg"] - 10.0) < 0.001, "heading 370° wraps to 10°")

vs5.update_heading(-45.0)
check(abs(vs5.fix["heading_deg"] - 315.0) < 0.001, "heading -45° wraps to 315°")


# ═══════════════════════════════════════════════════════════════════
# 6. MAVLinkClient._handle — malformed messages
# ═══════════════════════════════════════════════════════════════════
print("\n── 6. MAVLinkClient._handle: malformed messages ─────────────")

state = VehicleState()
client = MAVLinkClient(state=state)

def handle(pkt):
    try:
        client._handle(pkt)
        return True
    except Exception as e:
        return str(e)

check(handle({}) is True,                              "empty dict — no crash")
check(handle({"message": {}}) is True,                 "message with no type — no crash")
check(handle({"message": {"type": "HEARTBEAT"}}) is True, "unknown message type — no crash")
check(handle({"message": {"type": None}}) is True,     "type=None — no crash")
check(handle(None) is True,                            "None packet — no crash")  # AttributeError expected → should not raise
check(handle({"message": {"type": "GLOBAL_POSITION_INT",
                           "lat": "not_a_number", "lon": 0, "alt": 0}}) is True,
      "GLOBAL_POSITION_INT with string lat — no crash")
check(handle({"message": {"type": "GLOBAL_POSITION_INT",
                           "lat": 999_000_000, "lon": 0, "alt": 0}}) is True,
      "GLOBAL_POSITION_INT lat > ±90° sanity limit — no crash, ignored")
state_before = state.fix
check(state_before is None, "lat > sanity limit — fix not updated")

# Valid GPS message via _handle
check(handle({"message": {"type": "GLOBAL_POSITION_INT",
                           "lat": 407127000, "lon": -740059000,
                           "alt": 5000, "time_boot_ms": 0}}) is True,
      "valid GLOBAL_POSITION_INT — no crash")

# Valid ATTITUDE message
check(handle({"message": {"type": "ATTITUDE",
                           "yaw": math.radians(90)}}) is True,
      "valid ATTITUDE — no crash")

check(state.ready is True, "state.ready after valid GPS + ATTITUDE")
fix = state.fix
check(fix is not None and abs(fix["lat"] - 40.7127) < 0.001, "lat correct after _handle")
check(fix is not None and abs(fix["heading_deg"] - 90.0) < 0.01, "heading correct after _handle")

# Wrapped scalar values (mavlink2rest Bounded type)
state2 = VehicleState()
client2 = MAVLinkClient(state=state2)
check(client2._handle({"message": {"type": "GLOBAL_POSITION_INT",
                                   "lat": {"type": "Bounded", "value": 407127000},
                                   "lon": {"type": "Bounded", "value": -740059000},
                                   "alt": {"type": "Bounded", "value": 5000},
                                   "time_boot_ms": 0}}) is None,  # _handle returns None
      "Bounded-wrapped scalars — no crash")
state2.update_heading(45.0)
check(state2.ready, "Bounded-wrapped GPS: state becomes ready")

# ATTITUDE with wrapped yaw
state3 = VehicleState()
state3.update_gps(lat=1.0, lon=1.0, alt_m=0)
client3 = MAVLinkClient(state=state3)
client3._handle({"message": {"type": "ATTITUDE",
                              "yaw": {"type": "Bounded", "value": math.pi / 2}}})
check(abs(state3.fix["heading_deg"] - 90.0) < 0.01, "Bounded-wrapped yaw decoded correctly")


# ═══════════════════════════════════════════════════════════════════
# 7. georeference — edge cases
# ═══════════════════════════════════════════════════════════════════
print("\n── 7. georeference: edge cases ──────────────────────────────")

# Import the function directly
sys.path.insert(0, os.path.dirname(__file__))

# Recreate georeference locally (it references `vehicle` global in main.py,
# so we test the math directly)
def geo(detection, ping, fix):
    if fix is None:
        return None
    range_m = (ping["start_mm"] + detection["y"] * ping["mm_per_sample"]) / 1000.0
    bearing = math.radians(ping["transducer_heading_deg"])
    dlat = range_m * math.cos(bearing) / 111_111.0
    dlon = range_m * math.sin(bearing) / (111_111.0 * math.cos(math.radians(fix["lat"])))
    return fix["lat"] + dlat, fix["lon"] + dlon

fix = {"lat": 41.0, "lon": -70.0, "heading_deg": 0.0}
ping_n = {"start_mm": 0, "mm_per_sample": 100, "transducer_heading_deg": 0.0}

# No fix
check(geo({"y": 10}, ping_n, None) is None, "no fix → None")

# Range = 0 (y=0, start_mm=0)
r0 = geo({"y": 0}, ping_n, fix)
check(r0 is not None and abs(r0[0] - 41.0) < 1e-9 and abs(r0[1] - -70.0) < 1e-9,
      "range=0 → returns reference lat/lon unchanged")

# North bearing (0°) → only dlat changes
r_n = geo({"y": 100}, ping_n, fix)   # 100 * 100mm = 10m north
check(r_n is not None and r_n[0] > fix["lat"] and abs(r_n[1] - fix["lon"]) < 1e-9,
      "bearing=0° → moves north only")

# East bearing (90°) → only dlon changes
ping_e = {**ping_n, "transducer_heading_deg": 90.0}
r_e = geo({"y": 100}, ping_e, fix)
check(r_e is not None and abs(r_e[0] - fix["lat"]) < 1e-9 and r_e[1] > fix["lon"],
      "bearing=90° → moves east only")

# Equator (lat=0) — cos(0)=1, no division weirdness
fix_eq = {**fix, "lat": 0.0}
try:
    r_eq = geo({"y": 100}, ping_e, fix_eq)
    check(r_eq is not None, "bearing=90° at equator (lat=0) — no crash")
except Exception as e:
    check(False, "bearing=90° at equator — no crash", str(e))

# Very large y (beyond actual sonar range — should just give large range)
try:
    r_big = geo({"y": 999_999}, ping_n, fix)
    check(r_big is not None, "very large y — no crash")
except Exception as e:
    check(False, "very large y — no crash", str(e))

# Polar lat (near 90°) — cos(lat) approaches 0, dlon → inf
fix_pole = {**fix, "lat": 89.9999}
ping_e2 = {**ping_n, "transducer_heading_deg": 90.0}
try:
    r_pole = geo({"y": 100}, ping_e2, fix_pole)
    check(True, f"lat≈90° east bearing — no crash (lon={r_pole[1]:.1f})")
except Exception as e:
    check(False, "lat≈90° east bearing — no crash", str(e))


# ═══════════════════════════════════════════════════════════════════
# 8. WaterfallDetector — edge cases
# ═══════════════════════════════════════════════════════════════════
print("\n── 8. WaterfallDetector: edge cases ─────────────────────────")

detections_received = []

def on_det(objects, ping, image):
    detections_received.append((objects, ping, image))

def make_ping(ping_number, num_results=100, samples=None):
    if samples is None:
        samples_u16 = [32767] * num_results
    else:
        samples_u16 = samples
    return {
        "type": "ping",
        "ping_number": ping_number,
        "start_mm": 0,
        "length_mm": 10_000,
        "num_results": num_results,
        "timestamp_ms": ping_number * 50,
        "vehicle_heading_deg": 0.0,
        "transducer_heading_deg": 0.0,
        "min_pwr_db": -90.0,
        "max_pwr_db": -40.0,
        "samples_db": [(-90.0 + (v / 65535.0) * 50.0) for v in samples_u16],
        "mm_per_sample": 100.0,
    }

# detect_every=1 (fires on every ping after max_rows filled)
det1 = WaterfallDetector(max_rows=10, detect_every=1, on_detection=on_det)
try:
    for i in range(20):
        det1.add_ping(make_ping(i))
    check(True, "detect_every=1, 20 pings — no crash")
except Exception as e:
    check(False, "detect_every=1 — no crash", str(e))

# 1-sample pings
det2 = WaterfallDetector(max_rows=50, detect_every=50, on_detection=on_det)
try:
    for i in range(60):
        det2.add_ping(make_ping(i, num_results=1))
    check(True, "1-sample pings — no crash")
except Exception as e:
    check(False, "1-sample pings — no crash", str(e))

# All-max samples (saturated)
det3 = WaterfallDetector(max_rows=50, detect_every=50, on_detection=on_det)
try:
    for i in range(60):
        det3.add_ping(make_ping(i, samples=[65535] * 100))
    check(True, "all-saturated samples — no crash")
except Exception as e:
    check(False, "all-saturated samples — no crash", str(e))

# All-zero samples (silent)
det4 = WaterfallDetector(max_rows=50, detect_every=50, on_detection=on_det)
try:
    for i in range(60):
        det4.add_ping(make_ping(i, samples=[0] * 100))
    check(True, "all-zero samples — no crash")
except Exception as e:
    check(False, "all-zero samples — no crash", str(e))

# Variable num_results across pings (sonar could change range mid-mission)
det5 = WaterfallDetector(max_rows=50, detect_every=50, on_detection=on_det)
try:
    for i in range(60):
        n = 50 + (i % 3) * 25   # 50, 75, 100 cycling
        det5.add_ping(make_ping(i, num_results=n))
    check(True, "variable num_results per ping — no crash")
except Exception as e:
    check(False, "variable num_results per ping — no crash", str(e))

# Missing keys in ping dict
det6 = WaterfallDetector(max_rows=50, detect_every=50, on_detection=on_det)
bad_ping = {"type": "ping", "ping_number": 0}
try:
    det6.add_ping(bad_ping)
    check(True, "ping with missing keys — no crash (graceful)")
except (KeyError, TypeError):
    check(True, "ping with missing keys — raises KeyError/TypeError (expected)")
except Exception as e:
    check(False, "ping with missing keys — unexpected exception", str(e))


# ═══════════════════════════════════════════════════════════════════
# 9. Parser — interleaved GPS + pings stress run
# ═══════════════════════════════════════════════════════════════════
print("\n── 9. Parser: stress — interleaved GPS + 1000 pings ─────────")

p_stress = OmniScanParser()
buf = b""
gps_dict2 = {"mavpackettype": "GLOBAL_POSITION_INT",
             "lat": 407127000, "lon": -740059000, "alt": 5000, "hdg": 9000, "time_boot_ms": 0}
for i in range(1000):
    buf += build_ping(ping_number=i)
    if i % 20 == 0:
        buf += build_wrapper(gps_dict2)

try:
    results_stress = silent(list, p_stress.feed(buf))
    pings = [r for r in results_stress if r["type"] == "ping"]
    gps   = [r for r in results_stress if r["type"] == "gps"]
    check(len(pings) == 1000, f"1000 pings decoded (got {len(pings)})")
    check(len(gps) == 50,     f"50 GPS packets decoded (got {len(gps)})")   # i%20==0 for i in range(1000)
except Exception as e:
    check(False, "stress interleaved — no crash", str(e))


# ═══════════════════════════════════════════════════════════════════
# 10. Known edge-case: MAVLINK_WRAPPER lat=0 treated as falsy
# ═══════════════════════════════════════════════════════════════════
print("\n── 10. Known issue: lat=0/lon=0 treated as falsy ────────────")

p_zero = OmniScanParser()
zero_gps = {"mavpackettype": "GLOBAL_POSITION_INT",
            "lat": 0, "lon": 0, "alt": 0, "hdg": 0, "time_boot_ms": 0}
r_zero = silent(list, p_zero.feed(build_wrapper(zero_gps)))
check(len(r_zero) == 1, "lat=0/lon=0 decoded correctly (equator is a valid position)")
if r_zero:
    check(r_zero[0]["lat"] == 0.0 and r_zero[0]["lon"] == 0.0,
          "lat=0/lon=0 values preserved after decode")


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"{'═'*60}")
if FAIL > 0:
    sys.exit(1)
