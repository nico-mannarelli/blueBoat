"""
sonar_ws.py
Connects to a Cerulean SonarLink WebSocket and hands raw bytes to a callback.

Why this module exists on its own: the connection logic (finding the session,
opening the socket, reconnecting) has nothing to do with how we parse or detect.
Keeping it separate means the parser and detector never have to know the socket
exists -- they just receive bytes.
"""

import time

import requests
import websocket
from brping.pingmessage import PingMessage
from brping import definitions

_OS_PING_PARAMS = definitions.OMNISCAN450_OS_PING_PARAMS


class SonarLinkClient:
    def __init__(
        self,
        host="192.168.2.2",
        port=7077,
        on_bytes=None,
        length_mm=30_000,   # sonar range: 30 m default
        num_results=600,    # samples per ping (200-1200)
        gain_index=-1,      # -1 = auto gain
    ):
        self.host = host
        self.port = port
        self.on_bytes = on_bytes
        self.length_mm = length_mm
        self.num_results = num_results
        self.gain_index = gain_index
        self.session_id = None
        self._ws = None

    # ---- session discovery -------------------------------------------------
    def fetch_session_id(self):
        """Query /status and grab the first active session.

        SonarLink reassigns the session id whenever SonarView reconnects, so we
        never hardcode it -- we look it up fresh every time we (re)connect.
        """
        url = f"http://{self.host}:{self.port}/status"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        status = resp.json()

        sessions = status.get("sessions", [])
        if not sessions:
            raise RuntimeError(
                "SonarLink reports no active sessions. "
                "Open SonarView and start the sonar first."
            )
        self.session_id = sessions[0]["session_id"]
        return self.session_id

    # ---- command building --------------------------------------------------
    @staticmethod
    def _build_os_ping_params(
        length_mm=30_000,
        num_results=600,
        gain_index=-1,
        enable=1,
    ):
        """Return a serialised os_ping_params packet (ID 2197).

        Sending this with enable=1 tells the sonar to start pinging.
        Without it, SonarLink stays connected but the sonar is silent.
        """
        msg = PingMessage(_OS_PING_PARAMS)
        msg.start_mm = 0
        msg.length_mm = length_mm
        msg.msec_per_ping = 0           # 0 = maximum ping rate
        msg.reserved_1 = 0.0
        msg.reserved_2 = 0.0
        msg.pulse_len_percent = 0.002   # typical value from docs
        msg.filter_duration_percent = 0.0015
        msg.gain_index = gain_index     # -1 = auto, 0-7 = manual
        msg.num_results = num_results
        msg.enable = enable
        msg.reserved_3 = 0
        msg.reserved_4 = 0
        msg.reserved_5 = 0
        return bytes(msg.pack_msg_data())

    def start_ping(self):
        """Send os_ping_params with enable=1 to begin sonar pinging."""
        pkt = self._build_os_ping_params(
            length_mm=self.length_mm,
            num_results=self.num_results,
            gain_index=self.gain_index,
            enable=1,
        )
        self._ws.send(pkt, opcode=websocket.ABNF.OPCODE_BINARY)
        print(
            f"[ws] start_ping  length={self.length_mm}mm "
            f"num_results={self.num_results}  gain={self.gain_index}"
        )

    def stop_ping(self):
        """Send os_ping_params with enable=0 to halt sonar pinging."""
        pkt = self._build_os_ping_params(enable=0)
        self._ws.send(pkt, opcode=websocket.ABNF.OPCODE_BINARY)
        print("[ws] stop_ping sent")

    # ---- websocket callbacks ----------------------------------------------
    def _on_message(self, ws, message):
        if isinstance(message, (bytes, bytearray)):
            if self.on_bytes:
                self.on_bytes(message)

    def _on_open(self, ws):
        print(f"[ws] connected  session_id={self.session_id}")
        self.start_ping()

    def _on_error(self, ws, error):
        print(f"[ws] error: {error}")

    def _on_close(self, ws, code, msg):
        print(f"[ws] closed (code={code})")

    # ---- run loop ----------------------------------------------------------
    def run_forever(self, auto_reconnect=True):
        while True:
            try:
                self.fetch_session_id()
                url = (
                    f"ws://{self.host}:{self.port}"
                    f"/connect_ws?session_id={self.session_id}"
                )
                self._ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_open=self._on_open,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(skip_utf8_validation=True)
            except Exception as e:
                print(f"[ws] connection failed: {e}")

            if not auto_reconnect:
                break
            print("[ws] reconnecting in 3s...")
            time.sleep(3)
