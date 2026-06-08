"""
mock_sonarlink.py
A local HTTP + WebSocket server that mimics Cerulean SonarLink, so main.py
can run end-to-end without the boat.

Endpoints:
  GET /status              → JSON with one fake session
  GET /connect_ws          → WebSocket, streams os_mono_profile packets at 20 Hz

Usage:
  # Terminal 1 — start the mock server
  python mock_sonarlink.py

  # Terminal 2 — run the pipeline pointed at localhost
  HOST=127.0.0.1 python main.py
  # (or edit HOST in main.py to "127.0.0.1" temporarily)

The mock emits a bright target near sample 100 for pings 100-200, then
goes quiet, repeating every 300 pings — enough to trigger detection runs.
"""

import asyncio
import struct
import time

import numpy as np
from aiohttp import web
from brping.pingmessage import PingMessage
import brping.definitions as defs

HOST = "127.0.0.1"
PORT = 7077
SESSION_ID = "mock-0001"

NUM_SAMPLES = 200
LENGTH_MM = 10_000


def _build_packet(ping_number, target_sample=None):
    rng = np.random.default_rng(ping_number)  # deterministic per ping
    samples = rng.integers(0, 12_000, NUM_SAMPLES).tolist()

    if target_sample is not None:
        lo = max(0, target_sample - 8)
        hi = min(NUM_SAMPLES, target_sample + 8)
        for i in range(lo, hi):
            samples[i] = int(rng.integers(55_000, 65_535))

    msg = PingMessage(defs.OMNISCAN450_OS_MONO_PROFILE)
    msg.ping_number = ping_number
    msg.start_mm = 0
    msg.length_mm = LENGTH_MM
    msg.timestamp_ms = int(time.monotonic() * 1000) & 0xFFFF_FFFF
    msg.ping_hz = 800_000
    msg.gain_index = 6
    msg.num_results = NUM_SAMPLES
    msg.sos_dmps = 15_000
    msg.channel_number = 0
    msg.reserved = 0
    msg.pulse_duration_sec = 0.000125
    msg.analog_gain = 1.0
    msg.max_pwr_db = -40.0
    msg.min_pwr_db = -90.0
    msg.transducer_heading_deg = 0.0
    msg.vehicle_heading_deg = float(ping_number % 360)
    msg.pwr_results = bytearray(struct.pack(f"<{NUM_SAMPLES}H", *samples))
    return bytes(msg.pack_msg_data())


async def handle_status(request):
    return web.json_response({"sessions": [{"session_id": SESSION_ID}]})


async def handle_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    sid = request.rel_url.query.get("session_id", "?")
    print(f"[mock] client connected  session_id={sid}")

    ping_number = 0
    try:
        while not ws.closed:
            # target visible for pings 100-200 out of every 300
            target = 100 if (ping_number % 300) in range(100, 200) else None
            pkt = _build_packet(ping_number, target_sample=target)
            await ws.send_bytes(pkt)
            ping_number += 1
            await asyncio.sleep(0.05)  # 20 Hz
    except Exception as e:
        print(f"[mock] client disconnected: {e}")

    return ws


app = web.Application()
app.router.add_get("/status", handle_status)
app.router.add_get("/connect_ws", handle_ws)

if __name__ == "__main__":
    print(f"[mock] SonarLink server  →  http://{HOST}:{PORT}")
    print(f'[mock] Point main.py at HOST = "{HOST}"')
    web.run_app(app, host=HOST, port=PORT, print=None)
