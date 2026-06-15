"""
mock_sonarlink.py
Local mock of both Cerulean SonarLink and mavlink2rest, so main.py can
run end-to-end without the boat.

SonarLink (port 7077):
  GET /status       → fake session JSON
  GET /connect_ws   → WebSocket; waits for os_ping_params, then streams
                      os_mono_profile at 20 Hz

mavlink2rest (port 6040):
  GET /v1/ws/mavlink → WebSocket; streams GLOBAL_POSITION_INT + ATTITUDE
                       at 4 Hz in mavlink2rest JSON format

Usage:
  python mock_sonarlink.py          # terminal 1
  HOST=127.0.0.1 python main.py     # terminal 2
"""

import asyncio
import json
import math
import struct
import time

import numpy as np
from aiohttp import web
from brping.pingmessage import PingMessage, PingParser
import brping.definitions as defs

HOST       = "127.0.0.1"
PORT       = 7077
MAV_PORT   = 6040
SESSION_ID = "mock-0001"

# Fake position: somewhere in the Atlantic for obvious test output
_LAT_DEG =  41.123456
_LON_DEG = -70.654321

NUM_SAMPLES = 200
LENGTH_MM = 10_000


def _build_packet(ping_number, target_sample=None, num_results=NUM_SAMPLES,
                  length_mm=LENGTH_MM, channel_number=0):
    rng = np.random.default_rng(ping_number * 2 + channel_number)  # deterministic
    samples = rng.integers(0, 12_000, num_results).tolist()

    if target_sample is not None:
        lo = max(0, target_sample - 8)
        hi = min(num_results, target_sample + 8)
        for i in range(lo, hi):
            samples[i] = int(rng.integers(55_000, 65_535))

    msg = PingMessage(defs.OMNISCAN450_OS_MONO_PROFILE)
    msg.ping_number = ping_number
    msg.start_mm = 0
    msg.length_mm = length_mm
    msg.timestamp_ms = int(time.monotonic() * 1000) & 0xFFFF_FFFF
    msg.ping_hz = 450_000
    msg.gain_index = 6
    msg.num_results = num_results
    msg.sos_dmps = 15_000
    msg.channel_number = channel_number
    msg.reserved = 0
    msg.pulse_duration_sec = 0.000125
    msg.analog_gain = 1.0
    msg.max_pwr_db = -40.0
    msg.min_pwr_db = -90.0
    msg.transducer_heading_deg = 0.0
    msg.vehicle_heading_deg = float(ping_number % 360)
    msg.pwr_results = bytearray(struct.pack(f"<{num_results}H", *samples))
    return bytes(msg.pack_msg_data())


async def handle_status(_request):
    return web.json_response({"sessions": [{"session_id": SESSION_ID}]})


def _parse_ping_params(raw: bytes) -> dict | None:
    """Return os_ping_params fields if raw contains a valid packet, else None."""
    parser = PingParser()
    for byte in raw:
        if parser.parse_byte(byte) == PingParser.NEW_MESSAGE:
            msg = parser.rx_msg
            if msg.message_id == defs.OMNISCAN450_OS_PING_PARAMS:
                return {
                    "length_mm": msg.length_mm,
                    "num_results": msg.num_results,
                    "gain_index": msg.gain_index,
                    "enable": msg.enable,
                }
    return None


async def handle_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    sid = request.rel_url.query.get("session_id", "?")
    print(f"[mock] client connected  session_id={sid}")
    print(f"[mock] waiting for os_ping_params...")

    enabled = asyncio.Event()
    ping_params = {}

    async def recv_loop():
        # ws.receive() works across the full connection lifetime without
        # exhausting an iterator — safe to call repeatedly.
        while not ws.closed:
            msg = await ws.receive()
            if msg.type == web.WSMsgType.BINARY:
                params = _parse_ping_params(msg.data)
                if params:
                    if params["enable"]:
                        ping_params.update(params)
                        print(
                            f"[mock] os_ping_params received — starting stream  "
                            f"length={params['length_mm']}mm  "
                            f"num_results={params['num_results']}  "
                            f"gain={params['gain_index']}"
                        )
                        enabled.set()
                    else:
                        print("[mock] stop command received")
                        enabled.clear()
            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                break

    async def send_loop():
        await enabled.wait()
        ping_number = 0
        while not ws.closed:
            target = 100 if (ping_number % 300) in range(100, 200) else None
            # Emit both side-scan channels per ping, like the real OmniScan
            for channel in (0, 1):
                pkt = _build_packet(
                    ping_number,
                    target_sample=target if channel == 1 else None,
                    num_results=ping_params.get("num_results", NUM_SAMPLES),
                    length_mm=ping_params.get("length_mm", LENGTH_MM),
                    channel_number=channel,
                )
                await ws.send_bytes(pkt)
            ping_number += 1
            await asyncio.sleep(0.05)  # 20 Hz

    try:
        await asyncio.gather(recv_loop(), send_loop(), return_exceptions=True)
    except Exception as e:
        print(f"[mock] client disconnected: {e}")

    return ws


# ---- mavlink2rest mock (port 6040) -----------------------------------------

async def handle_mavlink_ws(_request):
    """Stream GLOBAL_POSITION_INT + ATTITUDE at 4 Hz in mavlink2rest format."""
    ws = web.WebSocketResponse()
    await ws.prepare(_request)
    print("[mock/mav] client connected")

    seq = 0
    heading_deg = 45.0   # start heading NE, slowly rotate
    try:
        while not ws.closed:
            ts = int(time.monotonic() * 1000) & 0xFFFF_FFFF

            gps_msg = {
                "header": {"system_id": 1, "component_id": 1, "sequence": seq},
                "message": {
                    "type":         "GLOBAL_POSITION_INT",
                    "time_boot_ms": ts,
                    "lat":          int(_LAT_DEG * 1e7),
                    "lon":          int(_LON_DEG * 1e7),
                    "alt":          500,          # 0.5 m (mm units)
                    "relative_alt": 500,
                    "vx":           10,
                    "vy":           0,
                    "vz":           0,
                    "hdg":          int(heading_deg * 100),
                },
            }
            att_msg = {
                "header": {"system_id": 1, "component_id": 1, "sequence": seq + 1},
                "message": {
                    "type":        "ATTITUDE",
                    "time_boot_ms": ts,
                    "roll":        0.0,
                    "pitch":       0.0,
                    "yaw":         math.radians(heading_deg),
                    "rollspeed":   0.0,
                    "pitchspeed":  0.0,
                    "yawspeed":    0.0,
                },
            }

            await ws.send_str(json.dumps(gps_msg))
            await ws.send_str(json.dumps(att_msg))

            heading_deg = (heading_deg + 0.5) % 360
            seq += 2
            await asyncio.sleep(0.25)   # 4 Hz
    except Exception:
        pass

    print("[mock/mav] client disconnected")
    return ws


# ---- server startup --------------------------------------------------------

sonar_app = web.Application()
sonar_app.router.add_get("/status",     handle_status)
sonar_app.router.add_get("/connect_ws", handle_ws)

mav_app = web.Application()
mav_app.router.add_get("/v1/ws/mavlink", handle_mavlink_ws)


async def _run_both():
    sonar_runner = web.AppRunner(sonar_app)
    mav_runner   = web.AppRunner(mav_app)
    await sonar_runner.setup()
    await mav_runner.setup()
    await web.TCPSite(sonar_runner, HOST, PORT).start()
    await web.TCPSite(mav_runner,   HOST, MAV_PORT).start()
    print(f"[mock] SonarLink    →  ws://{HOST}:{PORT}/connect_ws")
    print(f"[mock] mavlink2rest →  ws://{HOST}:{MAV_PORT}/v1/ws/mavlink")
    print(f'[mock] Run:  HOST=127.0.0.1 python main.py')
    await asyncio.Event().wait()   # run forever


if __name__ == "__main__":
    asyncio.run(_run_both())
