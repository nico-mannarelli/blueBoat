"""
planner_handoff.py
The single seam between this survey program and the revisit *planner*. When a
live survey ends, main.py hands the largest-N contacts here and this module gets
them to the planner, which builds a revisit plan to run *after* the current
mission.

There is exactly ONE thing to wire: point `_resolve_entry` at the planner's
entry point. Everything else (selecting/limiting contacts, the end-of-survey
trigger) already lives in main.py and detection_log.py.

--------------------------------------------------------------------------
Wiring the planner in
--------------------------------------------------------------------------
The planner consumes a coords array shaped `coords = [(lat, lon, 0), ...]`.
Pick whichever of these matches how the planner is packaged and delete the rest:

  1. The planner exposes a function (e.g. `plan_revisit(coords)`):
         from revisit_planner import plan_revisit
         def _resolve_entry(): return plan_revisit

  2. The planner only reads a coords file on disk:
         leave PLANNER_ENTRY unset — the fallback writes that file
         (set COORDS_OUT / PLANNER_COORDS_FILE to the path it imports).

  3. The entry point is configurable without editing code:
         set env PLANNER_ENTRY="revisit_planner:plan_revisit" before running.

If nothing is wired, this module still writes the coords file so the contacts
are never lost — the run just prints how to connect the planner.
"""

import importlib
import os

# Optional: "module:function" string, e.g. "revisit_planner:plan_revisit".
# Set this (or edit _resolve_entry below) to point at the revisit planner.
PLANNER_ENTRY = os.environ.get("PLANNER_ENTRY")

# Where the fallback writes the coords array if the planner isn't wired yet.
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


def _write_fallback(coords, path):
    from export_coords import write_coords_file
    write_coords_file(coords, path=path)
    return path


def send_to_planner(coords):
    """Hand the contact coords to the revisit planner.

    coords : list of (lat, lon, third) tuples — already filtered/limited by the
             caller (main.py passes the largest 50, hits >= 2).

    Returns True if the planner was called, False if it fell back to writing
    the coords file (planner not wired). Never raises on a missing planner.
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

    path = _write_fallback(coords, PLANNER_COORDS_FILE)
    print(f"[planner] planner not wired — wrote {len(coords)} coord(s) to {path}")
    print("[planner] connect it by editing _resolve_entry() in planner_handoff.py "
          'or setting PLANNER_ENTRY="module:function".')
    return False
