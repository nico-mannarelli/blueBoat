"""
planner_handoff.py
The seam between this survey program and the revisit step. When a live survey
ends, main.py hands the largest-N contacts here and this module gets them to the
boat so it re-drives the contacts after the survey.

Default behaviour: write the contacts to `contacts_coords.py`, an importable
module (`coords = [(lat, lon, 0), ...]`). Your coworker's MAVLink script imports
that file and does the actual upload — this program does NOT touch the autopilot.

Dispatch order in send_to_planner():
  1. a custom planner if wired (PLANNER_ENTRY / _resolve_entry) — takes priority,
  2. else (default) write the importable coords file for the coworker's script,
  3. optional: if UPLOAD_MAVLINK=1, this program uploads the mission itself
     instead — only turn this on if you want to be the uploader.

Key env knobs:
  UPLOAD_MAVLINK=1            let THIS program upload the mission (off by default)
  PLANNER_MAV_CONN=...        pymavlink connection to the AUTOPILOT, if uploading
                             (default udpout:$HOST:14550)
  PLANNER_ACCEPT_RADIUS=5     waypoint acceptance radius, metres
  PLANNER_DWELL_S=0           dwell seconds over each contact (>0 = LOITER_TIME)
  PLANNER_ENTRY=mod:func      route to a custom planner instead of the file

To route to a custom planner that consumes `coords = [(lat, lon, 0), ...]`,
set PLANNER_ENTRY="revisit_planner:plan_revisit" or hard-wire _resolve_entry().
"""

import importlib
import os

# Optional: "module:function" string, e.g. "revisit_planner:plan_revisit".
# Set this (or edit _resolve_entry below) to point at the revisit planner. When
# set it takes priority over the built-in MAVLink upload below.
PLANNER_ENTRY = os.environ.get("PLANNER_ENTRY")

# Optional built-in upload: push the contacts to the boat as a MAVLink mission
# directly from this program. OFF by default so the upload stays your coworker's
# job (she imports contacts_coords.py and uploads). Set UPLOAD_MAVLINK=1 only if
# you want THIS program to be the uploader instead.
UPLOAD_MAVLINK = os.environ.get("UPLOAD_MAVLINK", "0") not in ("0", "", "false", "False")
# pymavlink connection string to the AUTOPILOT (not mavlink2rest). Default targets
# the boat's standard MAVLink UDP endpoint; override for your link, e.g.
# PLANNER_MAV_CONN="udpout:192.168.2.2:14550" or "tcp:192.168.2.2:5777".
PLANNER_MAV_CONN = os.environ.get(
    "PLANNER_MAV_CONN", f"udpout:{os.environ.get('HOST', '192.168.2.2')}:14550")
# Waypoint acceptance radius (m) — how close the boat must get to count a
# contact as reached — and an optional dwell (s) over each one for a closer look.
PLANNER_ACCEPT_RADIUS = float(os.environ.get("PLANNER_ACCEPT_RADIUS", "5"))
PLANNER_DWELL_S = float(os.environ.get("PLANNER_DWELL_S", "0"))

# Where the fallback writes the coords array if upload is off/fails.
PLANNER_COORDS_FILE = os.environ.get(
    "PLANNER_COORDS_FILE", os.environ.get("COORDS_OUT", "contacts_coords.py")
)


def _resolve_entry():
    """Return the planner callable, or None if nothing is wired."""
    # --- HARD-WIRE THE PLANNER HERE (option 1) ----------------------------
    # from revisit_planner import plan_revisit
    # return plan_revisit
    # ----------------------------------------------------------------------
    if PLANNER_ENTRY:                       # option 3: env-configured
        mod_name, _, func_name = PLANNER_ENTRY.partition(":")
        if not func_name:
            raise ValueError(
                f'PLANNER_ENTRY must be "module:function", got {PLANNER_ENTRY!r}'
            )
        mod = importlib.import_module(mod_name)
        return getattr(mod, func_name)
    return None


def upload_mission_mavlink(coords, conn=None, accept_radius=None, dwell_s=None):
    """Upload `coords` to the boat's autopilot as a MAVLink mission so it
    re-drives the contacts. Each point becomes a NAV_WAYPOINT (or NAV_LOITER_TIME
    if dwell_s > 0). Runs the standard mission-upload handshake via pymavlink.

    coords : [(lat, lon, _), ...]. The 3rd slot is ignored (surface vessel, alt 0).

    NOTE: this REPLACES the autopilot's current mission. It is meant to run at
    end of survey, after the survey mission has finished. Raises on failure so
    the caller can fall back to writing the coords file.
    """
    from pymavlink import mavutil

    conn = conn or PLANNER_MAV_CONN
    accept_radius = PLANNER_ACCEPT_RADIUS if accept_radius is None else accept_radius
    dwell_s = PLANNER_DWELL_S if dwell_s is None else dwell_s

    print(f"[planner] connecting to autopilot {conn} ...")
    m = mavutil.mavlink_connection(conn)
    if m.wait_heartbeat(timeout=15) is None:
        raise RuntimeError(f"no MAVLink heartbeat from {conn} within 15s")
    sysid, compid = m.target_system, m.target_component
    print(f"[planner] heartbeat from sys {sysid} comp {compid}; "
          f"uploading {len(coords)} waypoint(s)")

    cmd = (mavutil.mavlink.MAV_CMD_NAV_LOITER_TIME if dwell_s > 0
           else mavutil.mavlink.MAV_CMD_NAV_WAYPOINT)
    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT

    def send_item(seq):
        lat, lon = coords[seq][0], coords[seq][1]
        # param1: hold/loiter time (s); param2: accept radius; param4: yaw (NaN=any)
        p1 = dwell_s if dwell_s > 0 else 0.0
        m.mav.mission_item_int_send(
            sysid, compid, seq, frame, cmd,
            1 if seq == 0 else 0,          # current
            1,                             # autocontinue
            p1, accept_radius, 0.0, float("nan"),
            int(round(lat * 1e7)), int(round(lon * 1e7)), 0.0)

    m.mav.mission_count_send(sysid, compid, len(coords))
    acked = 0
    while acked < len(coords):
        req = m.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST"],
                           blocking=True, timeout=10)
        if req is None:
            raise RuntimeError("autopilot stopped requesting mission items (timeout)")
        send_item(req.seq)
        acked = req.seq + 1

    ack = m.recv_match(type="MISSION_ACK", blocking=True, timeout=10)
    ok = ack is not None and ack.type == mavutil.mavlink.MAV_MISSION_ACCEPTED
    if not ok:
        raise RuntimeError(f"mission not accepted (ack={getattr(ack, 'type', None)})")
    print(f"[planner] mission accepted by autopilot ({len(coords)} waypoints)")
    return True


def _write_fallback(coords, path):
    from export_coords import write_coords_file
    write_coords_file(coords, path=path)
    return path


def send_to_planner(coords):
    """Hand the contact coords onward at end of survey.

    Dispatch order:
      1. a custom planner if wired (PLANNER_ENTRY / _resolve_entry),
      2. else upload a MAVLink mission to the boat (default),
      3. else (upload off or failed) write the importable coords file so the
         contacts are never lost.

    coords : list of (lat, lon, third) tuples — already filtered/limited by the
             caller (main.py passes the largest 50, hits >= 2).

    Returns True if a planner/upload handled it, False if it fell back to a file.
    Never raises.
    """
    if not coords:
        print("[planner] no contacts to hand off — nothing sent.")
        return False

    entry = _resolve_entry()
    if entry is not None:
        entry(coords)
        print(f"[planner] sent {len(coords)} contact(s) to "
              f"{getattr(entry, '__module__', '?')}."
              f"{getattr(entry, '__name__', 'planner')}()")
        return True

    if UPLOAD_MAVLINK:
        try:
            upload_mission_mavlink(coords)
            return True
        except Exception as e:
            print(f"[planner] MAVLink upload failed: {e}")
            path = _write_fallback(coords, PLANNER_COORDS_FILE)
            print(f"[planner] saved {len(coords)} coord(s) to {path} instead "
                  "(import them or retry the upload).")
            return False

    path = _write_fallback(coords, PLANNER_COORDS_FILE)
    mod = os.path.splitext(os.path.basename(path))[0]
    print(f"[planner] wrote {len(coords)} coord(s) to {path} for the revisit "
          f"script (it does: from {mod} import coords)")
    return False
