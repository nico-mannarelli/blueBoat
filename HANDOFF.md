# Sending detections to another program

The pipeline now collects every georeferenced detection into a single
de-duplicated list of **contacts**, held in memory by `DetectionLog`
(`detection_log.py`). This is the thing you hand off to a revisit planner.

At the end of a run (`replay_xtf.py` or `main.py`) the list is printed and is
available as the `log` object. Nothing is written or uploaded automatically —
this file shows you the options.

## The list you're handing off

```python
log.contacts        # list[dict], newest first — the live objects
log.to_records()    # list[dict] of plain JSON-safe rows (best for handoff)
log.to_geojson()    # GeoJSON FeatureCollection (QGIS / Leaflet / KML import)
log.to_csv()        # CSV text
```

Each contact looks like:

```python
{ "id": 3, "lat": 37.421031, "lon": -122.175890, "range_m": 9.1,
  "size_px": [11, 33], "source": "cfar", "score": 12.0, "hits": 4 }
```

`hits` is how many times the contact was re-sighted — a cheap confidence proxy.
Filter on it (e.g. `hits >= 2`) to drop one-off flickers before revisiting.

## Option 0 — a plain Python coords array (for a script that takes `coords = [...]`)

If the other program just wants a list of points like
`coords = [(lat, lon, 0), ...]`, use `log.to_coords()` / `log.to_coords_literal()`:

```python
log.to_coords()                      # [(37.421031, -122.17589, 0), ...]
log.to_coords(third=0, min_hits=2)   # 3rd slot value + drop one-ping flickers
log.to_coords(largest=50)            # only the 50 BIGGEST contacts, biggest first
log.to_coords_literal()              # 'coords = [(37.421031, -122.17589, 0), ...]'
log.to_coords_literal(largest=50)    # same, limited to the 50 largest by area
```

`third` fills the 3rd element of each tuple (altitude, a flag, or just 0 — set
it to whatever her script expects there). `largest=N` keeps only the N biggest
contacts by detection area (size_px width×height), ordered biggest-first — use
it to hand off only the most substantial shapes (logs, big rocks) and ignore
small clutter. To write it straight to a file her script can import or paste:

```python
with open("contacts_coords.py", "w") as f:
    f.write(log.to_coords_literal(largest=50) + "\n")
```

Both entry points can emit this for you at the end of a run:

```bash
# largest 50 contacts, dropping one-ping flickers
python replay_xtf.py scan.xtf --detector blob --detect-gamma 1.8 \
    --coords contacts_coords.py --coords-largest 50 --coords-min-hits 2

COORDS_OUT=contacts_coords.py COORDS_LARGEST=50 COORDS_MIN_HITS=2 python main.py
```

### Sending the largest 50 to your coworker's MAVLink script

Her script consumes a `coords = [(lat, lon, 0), ...]` array. Two ways to get
the largest 50 contacts into it:

**If her script imports a coords file** — point our run at it and you're done:

```bash
python replay_xtf.py scan.xtf --detector blob --detect-gamma 1.8 \
    --coords contacts_coords.py --coords-largest 50 --coords-min-hits 2
```

`contacts_coords.py` now holds `coords = [(lat, lon, 0), ...]` with exactly the
50 largest contacts, biggest first. Her script does `from contacts_coords import
coords` (or paste the line in).

**If you call her script in-process** — hand it the list directly:

```python
from her_mavlink_script import send_waypoints   # whatever her entry point is
send_waypoints(log.to_coords(largest=50, min_hits=2))
```

Either way the array is georeferenced: each (lat, lon) comes from the vehicle
GPS fix at the moment the contact was detected, so the boat can drive straight
to them. The run that produces the list must have detection ON (the coords come
*from* detection) — see the note below.

Note: the coordinates come *from detection* — it's what finds the contacts. So
the run that produces the list must have detection on. The clean
no-detection view (`--no-detect` / `NO_DETECT=1`) is for surveying/eyeballing;
it produces no contacts and therefore no coords.

## Option A — write a file the other program reads (simplest)

Good when the "other program" is QGroundControl, Mission Planner, QGIS, or any
tool the operator drives. Drop this at the end of `main()` / after the replay
loop:

```python
import json
with open("contacts.geojson", "w") as f:
    json.dump(log.to_geojson(), f, indent=2)
with open("contacts.csv", "w") as f:
    f.write(log.to_csv())
```

A QGroundControl **`.plan`** file (revisit each contact as a waypoint) is just
JSON — build it directly:

```python
def to_qgc_plan(contacts, alt=0.0):
    items = [{
        "type": "SimpleItem", "command": 16,        # MAV_CMD_NAV_WAYPOINT
        "frame": 3,                                  # MAV_FRAME_GLOBAL_RELATIVE_ALT
        "params": [0, 0, 0, None, c["lat"], c["lon"], alt],
        "autoContinue": True, "doJumpId": i + 1,
    } for i, c in enumerate(contacts)]
    return {"fileType": "Plan", "version": 1, "groundStation": "BlueBoat",
            "mission": {"version": 2, "cruiseSpeed": 2, "firmwareType": 12,
                        "vehicleType": 2, "items": items,
                        "plannedHomePosition": [contacts[0]["lat"],
                                                contacts[0]["lon"], 0]}}

json.dump(to_qgc_plan(log.to_records()), open("revisit.plan", "w"), indent=2)
```

## Option B — push a mission to the boat over MAVLink (most automated)

The boat already runs **mavlink2rest** on `:6040` (we read GPS from it). The
cleanest way to actually upload waypoints is `pymavlink`, which handles the
mission-upload handshake for you. This makes the boat re-drive the contacts.

```python
from pymavlink import mavutil

def upload_revisit(contacts, conn="udpout:192.168.2.2:14550"):
    m = mavutil.mavlink_connection(conn)
    m.wait_heartbeat()
    sysid, compid = m.target_system, m.target_component
    m.mav.mission_count_send(sysid, compid, len(contacts))
    for i, c in enumerate(contacts):
        m.recv_match(type="MISSION_REQUEST_INT", blocking=True, timeout=5)
        m.mav.mission_item_int_send(
            sysid, compid, i,
            6,                       # MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
            16,                      # MAV_CMD_NAV_WAYPOINT
            0, 1, 0, 5, 0, 0,        # hold/accept-radius etc. (param 4 = yaw)
            int(c["lat"] * 1e7), int(c["lon"] * 1e7), 0.0)
    m.recv_match(type="MISSION_ACK", blocking=True, timeout=5)

upload_revisit([c for c in log.to_records() if c["hits"] >= 2])
```

Notes for a surface vessel: altitude is irrelevant (use 0); set a sensible
acceptance radius (param 3) so the boat counts the waypoint as reached; consider
`MAV_CMD_NAV_LOITER_TIME` instead of `NAV_WAYPOINT` if you want it to dwell over
each contact for a closer scan. Upload to the **autopilot** sysid (usually 1),
not to mavlink2rest's own component.

## Option C — stream to a separate process (decoupled)

If the revisit logic lives in its own program, don't couple it in-process —
emit each new contact over a socket or stdout and let the other side consume it:

```python
import json, socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# inside the detection callback, when log.add() returns a *new* contact:
sock.sendto(json.dumps(record).encode(), ("127.0.0.1", 9100))
```

ZeroMQ `PUB`/`SUB` or an MQTT topic work the same way and survive restarts of
either side.

## Recommendation

For your workflow — survey first, revisit after the mission — Option A to
produce a `revisit.plan` plus Option B to push it is the sweet spot: you get a
file the operator can inspect/edit in QGroundControl **and** the option to
upload it straight to the boat. Gate both on `hits >= 2` so you only revisit
contacts the detector saw on more than one ping.
