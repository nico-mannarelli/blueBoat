"""
test_mission_fixes.py
Unit tests for the mission-2 fixes. No boat, no network, no heavy deps needed:

    python3 test_mission_fixes.py

Covers:
  - shared_states.current_id tracks the NEWEST contact id (off-by-one fix)
  - SonarLinkClient._reset_for_new_run() clears the stop latch + watchdog
  - run_second_mission: ignores recv timeouts (None), records waypoint image
    ids, completes on mission_state == 5 (NOT on seq alone), and supports the
    missionEnd fallback for firmware without mission_state
  - MAVLinkClient.reset_mission_tracking() lets completion fire once per
    mission instead of once per client lifetime
"""

import sys
import types
import unittest


# ---- stub hardware/heavy modules BEFORE importing the code under test ------

def _stub(name, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _FakeMav:
    def command_long_send(self, *a, **k):
        pass


class FakeConnection:
    """Scripted stand-in for pymavlink's connection. recv_match() plays back
    self.script; when the script is exhausted it raises SystemExit so a loop
    that failed to break turns into a test failure instead of a hang
    (SystemExit is not caught by `except Exception`)."""

    def __init__(self):
        self.mav = _FakeMav()
        self.target_system = 1
        self.target_component = 1
        self.script = []

    def recv_match(self, *a, **k):
        if not self.script:
            raise SystemExit("recv_match called past end of scripted messages "
                             "- the loop under test failed to break")
        return self.script.pop(0)


fake_connection = FakeConnection()
_stub("mavcon", connection=fake_connection)   # must precede sonar_ws import

try:
    import websocket  # noqa: F401
except ImportError:
    _stub("websocket", WebSocketApp=object,
          ABNF=types.SimpleNamespace(OPCODE_BINARY=2))

try:
    from brping import definitions  # noqa: F401
except ImportError:
    br = _stub("brping",
               definitions=types.SimpleNamespace(OMNISCAN450_OS_PING_PARAMS=2197))
    pm = _stub("brping.pingmessage", PingMessage=object)
    br.pingmessage = pm

try:
    from pymavlink import mavutil  # noqa: F401
except ImportError:
    pmav = _stub("pymavlink")
    pmav.mavutil = types.SimpleNamespace(
        mavlink=types.SimpleNamespace(MAV_CMD_REQUEST_MESSAGE=512,
                                      MAVLINK_MSG_ID_MISSION_CURRENT=42))

try:
    from shapely.geometry import Point, Polygon  # noqa: F401
except ImportError:
    sh = _stub("shapely")
    geo = _stub("shapely.geometry", Point=object, Polygon=object)
    sh.geometry = geo

try:
    import requests  # noqa: F401
except ImportError:
    _stub("requests")

import shared_states
from detection_log import DetectionLog
from sonar_ws import SonarLinkClient
from mavlink_client import MAVLinkClient


class Msg:
    """Minimal fake pymavlink message."""

    def __init__(self, mtype, **fields):
        self._mtype = mtype
        for k, v in fields.items():
            setattr(self, k, v)

    def get_type(self):
        return self._mtype


# ---- tests ------------------------------------------------------------------

class TestCurrentId(unittest.TestCase):
    def setUp(self):
        shared_states.current_id = 0

    def test_current_id_is_newest_contact_id(self):
        log = DetectionLog()
        c1 = log.add(41.0, -71.0)
        self.assertEqual(c1["id"], 1)
        self.assertEqual(shared_states.current_id, 1)   # was 2 before the fix

        c2 = log.add(41.1, -71.1)                       # far away -> new contact
        self.assertEqual(c2["id"], 2)
        self.assertEqual(shared_states.current_id, 2)

    def test_merged_sighting_does_not_bump_current_id(self):
        log = DetectionLog()
        log.add(41.0, -71.0)
        log.add(41.0, -71.0)                            # same spot -> merged
        self.assertEqual(shared_states.current_id, 1)
        self.assertEqual(len(log), 1)


class TestResetForNewRun(unittest.TestCase):
    def test_stop_latch_is_cleared(self):
        s = SonarLinkClient()
        s.stop()
        self.assertTrue(s._stop)
        s._reset_for_new_run()
        self.assertFalse(s._stop)
        self.assertIsNone(s._watchdog)
        self.assertIsNone(s._last_rx)


class TestRunSecondMission(unittest.TestCase):
    def setUp(self):
        shared_states.mission_1_png = True
        shared_states.current_id = 0
        self.sonar = SonarLinkClient()
        self.sonar._stop = True                 # as left behind by mission 1
        self.sonar.start_sonar_thread = lambda **k: None   # no real websocket

    def test_completes_on_mission_state_not_on_seq(self):
        shared_states.current_id = 7
        fake_connection.script = [
            None,                                          # recv timeout
            Msg("HEARTBEAT"),                              # unrelated message
            Msg("MISSION_ITEM_REACHED", seq=1),
            Msg("MISSION_CURRENT", seq=4, mission_state=3),  # seq==4, ACTIVE:
            Msg("MISSION_ITEM_REACHED", seq=2),              # must NOT break
            Msg("MISSION_CURRENT", seq=4, mission_state=5),  # COMPLETE
        ]
        images = self.sonar.run_second_mission()
        self.assertEqual(fake_connection.script, [])   # consumed everything
        self.assertIn((1, 7), images)
        self.assertIn((2, 7), images)
        self.assertFalse(shared_states.mission_1_png)  # 2-prefixed pngs active
        self.assertTrue(self.sonar._stop)              # sonar stopped at end

    def test_missionend_fallback_without_mission_state(self):
        fake_connection.script = [
            Msg("MISSION_ITEM_REACHED", seq=1),
            Msg("MISSION_ITEM_REACHED", seq=2),
            Msg("MISSION_ITEM_REACHED", seq=3),   # missionEnd -> break
        ]
        images = self.sonar.run_second_mission(missionEnd=3)
        self.assertEqual(fake_connection.script, [])
        self.assertEqual([s for s, _ in images[1:]], [1, 2, 3])

    def test_stop_latch_reset_so_sonar_can_start(self):
        started = []
        self.sonar.start_sonar_thread = lambda **k: started.append(
            self.sonar._stop)
        fake_connection.script = [Msg("MISSION_CURRENT", seq=0,
                                      mission_state=5)]
        self.sonar.run_second_mission()
        # start_sonar_thread must be called with _stop already cleared,
        # otherwise its thread loop exits before ever connecting
        self.assertEqual(started, [False])


class TestRunFirstMission(unittest.TestCase):
    def test_waypoint_fallback_without_mission_current(self):
        # Old firmware/SITL never streams MISSION_CURRENT — mission 1 must
        # still complete via MISSION_ITEM_REACHED at the last waypoint.
        s = SonarLinkClient()
        s.start_sonar_thread = lambda **k: None
        fake_connection.script = [
            None,                                        # pre-loop initial recv
            Msg("MISSION_CURRENT", seq=1, mission_state=0),
            Msg("MISSION_ITEM_REACHED", seq=2),
            Msg("MISSION_ITEM_REACHED", seq=4),          # missionEnd -> break
        ]
        s.run_first_mission(4)
        self.assertEqual(fake_connection.script, [])
        self.assertTrue(s._stop)     # explicit stop fired after completion


class TestMissionCompleteReset(unittest.TestCase):
    def test_completion_fires_once_per_mission_after_reset(self):
        fired = []
        mc = MAVLinkClient(on_mission_complete=lambda: fired.append(1))
        packet = {"message": {"type": "MISSION_CURRENT", "mission_state": 5}}

        mc._handle(packet)
        mc._handle(packet)                      # duplicate within mission 1
        self.assertEqual(len(fired), 1)         # one-shot within a mission

        mc.reset_mission_tracking()             # between missions
        self.assertIsNone(mc.state.mission_state)

        mc._handle(packet)                      # mission 2 completes
        self.assertEqual(len(fired), 2)         # fires again after reset


if __name__ == "__main__":
    unittest.main(verbosity=2)
