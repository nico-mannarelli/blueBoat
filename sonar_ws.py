"""
sonar_ws.py
Connects to a Cerulean SonarLink WebSocket and hands raw bytes to a callback.

Why this module exists on its own: the connection logic (finding the session,
opening the socket, reconnecting) has nothing to do with how we parse or detect.
Keeping it separate means the parser and detector never have to know the socket
exists -- they just receive bytes.
"""

import threading
import time

import requests
import websocket
from brping.pingmessage import PingMessage
from brping import definitions

from pymavlink import mavutil
from mavcon import connection

import shared_states

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
        idle_timeout=0.0,   # >0: end the run after this many seconds of no sonar
                            # data (survey over). 0 disables the watchdog.
    ):
        self.host = host
        self.port = port
        self.on_bytes = on_bytes
        self.length_mm = length_mm
        self.num_results = num_results
        self.gain_index = gain_index
        self.idle_timeout = idle_timeout
        self.session_id = None
        self._ws = None
        self._stop = False        # set by stop() to end the run for good
        self._last_rx = None      # time of last sonar byte (for the watchdog)
        self._watchdog = None

    def stop(self):
        """End the run deterministically: stop reconnecting and close the socket
        so run_forever() returns and the caller's end-of-survey handoff fires."""
        self._stop = True
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass

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
            self._last_rx = time.time()
            if self.on_bytes:
                self.on_bytes(message)

    def _on_open(self, ws):
        print(f"[ws] connected  session_id={self.session_id}")
        self.start_ping()

    def _on_error(self, ws, error):
        print(f"[ws] error: {error}")

    def _on_close(self, ws, code, msg):
        print(f"[ws] closed (code={code})")

    # ---- idle watchdog (optional backstop) ---------------------------------
    def _watch_idle(self):
        """If no sonar data arrives for idle_timeout seconds after the survey
        has started, end the run. Off unless idle_timeout > 0."""
        while not self._stop:
            time.sleep(1.0)
            if self._last_rx is None:        # not started yet — wait
                continue
            if time.time() - self._last_rx > self.idle_timeout:
                print(f"[ws] no sonar data for {self.idle_timeout:.0f}s "
                      "— ending survey")
                self.stop()
                break



    # should have sonar run and check for mission sequence at same time,
    # then for the second mission the sonar can run forever
    def start_sonar_thread(self, auto_reconnect=True):
        def _run():
            while not self._stop:
                try:
                    # Always fetch a fresh session ID before connecting
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

                    print("[ws] sonar connecting...")
                    self._ws.run_forever()   # BLOCKS this thread only

                except Exception as e:
                    print(f"[ws] sonar connection failed: {e}")

                # Stop or no reconnect → exit thread
                if self._stop or not auto_reconnect:
                    print("[ws] sonar thread exiting (stop or no reconnect)")
                    break

                print("[ws] sonar reconnecting in 3s...")
                time.sleep(3)

        # Launch sonar websocket in background thread
        threading.Thread(target=_run, daemon=True, name="sonar-ws").start()
        print("[ws] sonar thread started")

    # 
    def run_first_mission(self, missionEnd, auto_reconnect=True):
        if self.idle_timeout and self.idle_timeout > 0 and self._watchdog is None:
            self._watchdog = threading.Thread(target=self._watch_idle,
                                              daemon=True, name="sonar-idle")
            self._watchdog.start()

        self.start_sonar_thread()

        connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
                0,
                mavutil.mavlink.MAVLINK_MSG_ID_MISSION_CURRENT, 500000, 0, 0, 0, 0, 0
                )
        
        msg = connection.recv_match(
                    type=['MISSION_CURRENT'],
                    blocking=True, timeout=5
                )
        
        count = 0
        while True and missionEnd != 0:
            try:
                msg = connection.recv_match(
                    type=['MISSION_CURRENT'],
                    blocking=True, timeout=5
                )

                if msg.mission_state == 5:
                    print("Mission 1 complete!")
                    break

                # if msg:
                #     print(msg.seq)
                #     #print(count) # for testing
                #     if msg.seq == missionEnd: #msg.seq = missionEnd 
                #         print("first mission complete!")
                #         break
                count += 1
                    
            except Exception as e:
                print(f"[ws] connection failed: {e}")

    #           
    def run_second_mission(self, auto_reconnect=True):
        shared_states.mission_1_png = False
        if self.idle_timeout and self.idle_timeout > 0 and self._watchdog is None:
            self._watchdog = threading.Thread(target=self._watch_idle,
                                              daemon=True, name="sonar-idle")
            self._watchdog.start()

        self.start_sonar_thread()

        connection.mav.command_long_send(
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
                0,
                mavutil.mavlink.MAVLINK_MSG_ID_MISSION_CURRENT, 500000, 0, 0, 0, 0, 0
                )
        mission_2_images = [(0,0)]
        count = 0
        while True:
            try:
                msg = connection.recv_match(blocking=True, timeout=5)
                print(msg)
                print("!!!!!!!")
                type = msg.get_type() 
                if type == "MISSION_CURRENT":
                    print(msg.seq)
                    #print(count) # for testing
                    if msg.seq == 4: #msg.seq = missionEnd 
                        print("second mission complete!")
                        print(mission_2_images)
                        break
                # if type == "MISSION_CURRENT":
                #     if msg.mission_stat
                # e == 5:
                #         print("Mission 2 complete!")
                #         print(mission_2_images)
                #         break

                # detects when a waypoint is reached, 
                # need to add how to get corresponding image
                # second mission images saved as 2-----.png
                elif type == "MISSION_ITEM_REACHED":
                    print(f"Reached waypoint: {msg.seq}")
                    print(f"Id of most recent image: {shared_states.current_id}")
                    mission_2_images.append((msg.seq, shared_states.current_id))

                count += 1
                    
            except Exception as e:
                print(f"[ws] connection failed: {e}")

    # runs forever, need ctrl c to stop
    # ---- run loop ----------------------------------------------------------
    # def run_forever(self, auto_reconnect=True):
    #     if self.idle_timeout and self.idle_timeout > 0 and self._watchdog is None:
    #         self._watchdog = threading.Thread(target=self._watch_idle,
    #                                           daemon=True, name="sonar-idle")
    #         self._watchdog.start()
    #     while not self._stop:
    #         try:
    #             self.fetch_session_id()
    #             url = (
    #                 f"ws://{self.host}:{self.port}"
    #                 f"/connect_ws?session_id={self.session_id}"
    #             )
    #             self._ws = websocket.WebSocketApp(
    #                 url,
    #                 on_message=self._on_message,
    #                 on_open=self._on_open,
    #                 on_error=self._on_error,
    #                 on_close=self._on_close,
    #             )
    #             self._ws.run_forever(skip_utf8_validation=True)
    #         except Exception as e:
    #             print(f"[ws] connection failed: {e}")

    #         if self._stop or not auto_reconnect:
    #             break
    #         print("[ws] reconnecting in 3s...")
    #         time.sleep(3)
