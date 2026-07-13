# Testing the two-mission autonomy chain

Three levels, cheapest first. Run levels 1–2 after ANY change to
`sonar_ws.py`, `mavlink_client.py`, `main.py`, or `detection_log.py`;
run level 3 once with the boat on the bench before a field day.

---

## Level 1 — unit tests (any machine, ~1 second)

```
python3 test_mission_fixes.py
```

No boat, no network, no SITL. Covers: the `_stop` reset, the image-id
off-by-one, mission-2 completion on `mission_state == 5` (not `seq == 4`),
the `missionEnd` waypoint fallback, recv-timeout handling, and
`reset_mission_tracking()` letting completion fire once per mission.

All 7 tests must pass. If one fails, do not bother with level 2.

---

## Level 2 — full two-mission simulation (no hardware)

This is the test that reproduces the field failure. It runs the REAL
firmware (ArduPilot SITL) driving a simulated boat through the real
mission state machine, with the sonar faked by `mock_sonarlink.py`.

### One-time setup

QGroundControl has no built-in simulator, so SITL runs as a separate
process and QGC connects to it automatically. On the Windows
groundstation laptop use WSL (`wsl --install`, then Ubuntu); on
macOS/Linux run it natively:

```
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y   # ubuntu/WSL
python3 -m pip install MAVProxy
```

### Start the simulated boat

```
Tools/autotest/sim_vehicle.py -v Rover -f motorboat --map --console
```

SITL broadcasts MAVLink on UDP 14550 — **QGroundControl picks it up
automatically** when opened on the same machine (a "Vehicle 1"
connection appears; under WSL2, if QGC runs on the Windows side, add a
comm link to the WSL IP shown by `hostname -I`). Extra TCP ports for
the pipeline are on `127.0.0.1:5762` and `5763`.

### Start the fake sonar

In the repo folder:

```
python mock_sonarlink.py
```

This serves SonarLink on `:7077` and a fake mavlink2rest on `:6040`, both
on localhost.

### Run the pipeline against the sim

```
# PowerShell (Windows)
$env:HOST = "127.0.0.1"
$env:MAVCON_URL = "tcp:127.0.0.1:5762"
python main.py

# bash (Linux/macOS)
HOST=127.0.0.1 MAVCON_URL=tcp:127.0.0.1:5762 python main.py
```

Then in QGroundControl: **Plan view** → add a short 4-waypoint survey →
**Upload**, then back in **Fly view** arm and start the mission (the
"Start Mission" slider). The sim boat drives it in real time.

> If the mock sonar produced no contacts by mission end, `main.py` skips
> the mission-2 upload (by design — it never uploads an empty mission).
> To still exercise mission 2, upload and start a second mission manually
> from QGroundControl once you see "Mission 1 complete!".

### Pass criteria — watch the console for ALL of these, in order

Mission 1:
1. `[ws] sonar thread started` and sonar data flowing
2. `Mission 1 complete!` when the last waypoint is reached
3. `[ws] sonar thread exiting` — mission-1 sonar shuts down
4. coords file written with `(lat, lon, id)` triples
5. uploader launches (`[main] launching uploader: python mavlink.py`)

Mission 2 — these four are the regression checks for the field failure:

6. `[ws] sonar connecting...` appears AGAIN (sonar restarted — fix #1)
7. `Reached waypoint: N` / `Id of most recent image: M` prints WHILE
   sonar data is still streaming (MAVLink not blocked)
8. new images named `2xxxxx.png` in `sonar_web/data/images/`
9. `second mission complete!` with a populated list of
   `(waypoint, image_id)` pairs — and it must NOT print before the sim
   boat has actually finished (early exit = the old `seq == 4` bug)

---

## Level 3 — bench check with the real boat (office, one-time per firmware)

The two things SITL cannot prove. Boat powered on the desk, connected
to its network:

**A. Does this firmware populate `mission_state`?**

```
python3 - <<'EOF'
from pymavlink import mavutil
c = mavutil.mavlink_connection('tcp:192.168.2.2:5777')
c.wait_heartbeat()
m = c.recv_match(type=['MISSION_CURRENT'], blocking=True, timeout=15)
print("no MISSION_CURRENT in 15s" if m is None else
      f"mission_state present: {'mission_state' in m.get_fieldnames()}  value={getattr(m, 'mission_state', None)}")
EOF
```

- `present: True` → completion detection works as coded.
- `present: False` or no message → mission completion will NEVER fire on
  the water. Use the fallback: call
  `sonar.run_second_mission(missionEnd=<last waypoint #>)` in `main.py`,
  and fix the MISSION_CURRENT stream rate before the field day.

**B. Are the services up?** (browser or curl)

- `http://192.168.2.2:6040/v1/mavlink` → JSON = mavlink2rest alive
- `http://192.168.2.2:7077/status` → must list an active session with
  SonarView open = SonarLink alive

Mission progression (waypoints actually being reached) cannot be tested
indoors — the boat can't move and has no GPS fix. That part is exactly
what Level 2 covers; don't try to verify it on the bench.
