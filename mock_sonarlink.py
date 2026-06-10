"""
mock_sonarlink.py
A local HTTP + WebSocket server that mimics Cerulean SonarLink, so main.py
can run end-to-end without the boat.

Endpoints:
  GET /status              → JSON with one fake session
  GET /connect_ws          → WebSocket; waits for os_ping_params before
                             streaming os_mono_profile packets at 20 Hz

Usage:
  # Terminal 1 — start the mock server
  python mock_sonarlink.py

  # Terminal 2 — run the pipeline pointed at localhost
  HOST=127.0.0.1 python main.py
  # (or edit HOST in main.py to "127.0.0.1" temporarily)

The mock will not stream any pings until the client sends os_ping_params
with enable=1, matching real SonarLink behaviour. Once started, it emits
a bright target near sample 100 for pings 100-200 out of every 300.
"""

import asyncio
import struct
import time

import numpy as np
from aiohttp import web
from brping.pingmessage import PingMessage, PingParser
import brping.definitions as defs

HOST = "127.0.0.1"
PORT = 7077
SESSION_ID = "mock-0001"

NUM_SAMPLES = 200
LENGTH_MM = 10_000


def _build_packet(ping_number, target_sample=None, num_results=NUM_SAMPLES, length_mm=LENGTH_MM):
    rng = np.random.default_rng(ping_number)  # deterministic per ping
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
    msg.channel_number = 0
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
            pkt = _build_packet(
                ping_number,
                target_sample=target,
                num_results=ping_params.get("num_results", NUM_SAMPLES),
                length_mm=ping_params.get("length_mm", LENGTH_MM),
            )
            await ws.send_bytes(pkt)
            ping_number += 1
            await asyncio.sleep(0.05)  # 20 Hz

    try:
        await asyncio.gather(recv_loop(), send_loop(), return_exceptions=True)
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
