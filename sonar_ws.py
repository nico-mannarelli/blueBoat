"""
sonar_ws.py
Connects to a Cerulean SonarLink WebSocket and hands raw bytes to a callback.

Why this module exists on its own: the connection logic (finding the session,
opening the socket, reconnecting) has nothing to do with how we parse or detect.
Keeping it separate means the parser and detector never have to know the socket
exists -- they just receive bytes.
"""

import json
import time

import requests
import websocket


class SonarLinkClient:
    def __init__(self, host="192.168.2.2", port=7077, on_bytes=None):
        self.host = host
        self.port = port
        self.on_bytes = on_bytes          # callback(raw_bytes) -> None
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

    # ---- websocket callbacks ----------------------------------------------
    def _on_message(self, ws, message):
        # Binary frames arrive as bytes; text frames as str. We only care about
        # binary (the Cerulean packets), but guard against str just in case.
        if isinstance(message, (bytes, bytearray)):
            if self.on_bytes:
                self.on_bytes(message)

    def _on_open(self, ws):
        print(f"[ws] connected, session_id={self.session_id}")

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
