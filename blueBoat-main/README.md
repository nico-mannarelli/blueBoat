# BlueBoat Sonar Detection

Real-time underwater object detection for a BlueBoat autonomous surface vessel
using a Cerulean OmniScan 450 side-scan sonar. The pipeline parses live sonar
data, detects anomalies on a waterfall image, and georeferences each detection
to a GPS position.

```
mavlink2rest WS ─────────────────────────────────┐
                                                  ▼
SonarLink WS  →  parser  →  WaterfallDetector  →  georeference → lat/lon
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.12+. The default detector runs entirely on CPU (numpy +
OpenCV). PyTorch is **optional** and only needed to run CFAR on a GPU
(`cfar_backend="torch"`) on the edge server — see `requirements.txt`.

## Core pipeline

Data flows: **raw bytes → parsed pings → waterfall image → detections → lat/lon**.

| File | Role |
|------|------|
| `sonar_ws.py` | WebSocket client for SonarLink (`192.168.2.2:7077`). Fetches a session ID, sends `os_ping_params` to start pinging, streams raw bytes to a callback. |
| `sonar_parse.py` | Parses Cerulean Ping packets into dicts. Decodes `os_mono_profile` (sonar samples + `channel_number`) and `MAVLINK_WRAPPER` (embedded GPS). |
| `sonar_detect.py` | The detection engine (`WaterfallDetector`). Builds the flat-fielded waterfall and runs **CFAR** anomaly detection (adaptive per-pixel threshold, no baseline or labels needed) with optional shadow gating, plus an alternate classical Hough + ROI detector. NMS, nadir masking, and box-merging included. **All detection logic lives here.** |
| `mavlink_client.py` | Connects to mavlink2rest (`:6040`) for GPS + heading, kept in a thread-safe `VehicleState`. |
| `sonar_dashboard.py` | Composes the operator window: large waterfall, range ruler, colour bar, nadir line, labeled detection boxes, telemetry/compass HUD, live contacts list, and a track mini-map. numpy + OpenCV only, cosmetic (replaces the bare `cv2.imshow`). |
| `detection_log.py` | `DetectionLog` — accumulates georeferenced detections into a de-duplicated in-memory list of *contacts* (the mission hand-off list). `to_records/to_coords/to_geojson/to_csv` helpers. |
| `sonar_display.py` | Palette LUTs + gamma for the waterfall colourisation (used by the dashboard). |
| `export_coords.py` | Writes the contacts to an importable Python file (`coords = [(lat, lon, 0), ...]`), or fills a hard-coded array (e.g. `WAYPOINTS`) in another file in place. |
| `planner_handoff.py` | The single seam to a downstream revisit *planner*: at end of survey `main.py` passes the contacts here; wire the planner's entry point into `_resolve_entry()`. |
| `mavlink.py` | Mission controller — uploads a waypoint mission to the boat over MAVLink (`upload_mission`, `set_orbit`, geofence checks). Reads its `WAYPOINTS` array; run as a script when connected to the boat. |
| `mav_fence` | Uploads the geofence polygon (operating area) to the boat over MAVLink. Run as a script. |

`main.py`, `replay_xtf.py`, and `mock_sonarlink.py` are three different *sources*
feeding the same detector: the live boat, a recorded file, or a simulator.
`label_xtf.py` / `score_detector.py` hand-label a scan and score the detector
(precision/recall) against those labels; `preview_dashboard.py` renders the
dashboard headlessly.

## Running

### Against the real boat
```bash
python main.py                      # connects to 192.168.2.2
```
Opens a live waterfall window per channel and prints georeferenced detections.

### Offline, on a recorded XTF (main test rig)
```bash
python replay_xtf.py scan.xtf --probe       # inspect file metadata
python replay_xtf.py scan.xtf               # real-time playback
python replay_xtf.py scan.xtf --speed 4     # 4x faster
python replay_xtf.py scan.xtf --speed 0     # max speed, no delay
python replay_xtf.py scan.xtf --channel 1   # single channel (default: both, combined swath)
python replay_xtf.py scan.xtf --detector blob   # try a different detector (cfar|blob|roi|classical|both)
```
Replays a SonarView `.xtf` through the exact production detector.

### Display palette
The live windows colourise the waterfall. Choose a palette with the
`SONAR_PALETTE` env var — `blue` (default), `amber` (SonarView-style), or `gray`:
```bash
SONAR_PALETTE=amber python replay_xtf.py scan.xtf
```
Purely cosmetic — detection is unaffected.

### Without hardware (mock boat)
Two terminals:
```bash
# Terminal 1 — fake SonarLink + mavlink2rest server
python mock_sonarlink.py

# Terminal 2 — pipeline pointed at localhost
HOST=127.0.0.1 python main.py
```

### Synthetic detector check
```bash
python preview_dashboard.py         # headless dashboard render → dashboard_preview.png
```

The live windows (`main.py`, `replay_xtf.py`) render through `sonar_dashboard.py`
— a composited operator console rather than a raw grayscale window. Press **Q**
(or Esc) to quit. Every detection is folded into a de-duplicated contact list
(`detection_log.py`), printed at the end of a run.

## Contact hand-off

Every run produces a de-duplicated list of *contacts* (georeferenced lat/lon).
There are three ways to get them out.

### 1. Importable coords file (default)
`replay_xtf.py` writes the contacts to **`contacts.py`** by default — an
importable array another program loads with one line:
```python
from contacts import coords        # coords = [(lat, lon, 0), ...]
```
`contacts.py` is committed (starting empty), so the import always works even
before the first scan. Each run overwrites it.
```bash
python replay_xtf.py scan.xtf                          # writes contacts.py
python replay_xtf.py scan.xtf --coords out.py          # different filename
python replay_xtf.py scan.xtf --coords-largest 50 --coords-min-hits 2   # only the real contacts
python replay_xtf.py scan.xtf --no-coords              # skip writing
```

### 2. Fill a waypoint array in place
Drop the contacts straight into the mission controller's `WAYPOINTS` array
without touching the rest of that file (a `.bak` backup is written first). The
points are written as `(lat, lon)` pairs to match the controller's format:
```bash
python replay_xtf.py scan.xtf --populate mavlink.py        # fills WAYPOINTS in mavlink.py
python replay_xtf.py scan.xtf --populate target.py --populate-var POINTS   # different file/array
```

### 3. End-of-survey planner call (live)
`main.py` hands the largest contacts to a downstream revisit planner through
`planner_handoff.py` at end of survey — wire the planner's function into
`_resolve_entry()`, or set `COORDS_OUT=...` to just write the coords file.

## Full run: scan → mission

Run the scan, fill the mission controller's waypoints, and launch it
automatically — one command:
```bash
python replay_xtf.py scan.xtf \
  --populate mavlink.py --coords-largest 50 --coords-min-hits 2 \
  --run-after "python mavlink.py"
```
Sequence: scan → CFAR contacts → `WAYPOINTS` filled in `mavlink.py` → the sonar
window closes → `mavlink.py` launches (connects to the boat and uploads the
revisit mission). `--run-after` is **skipped if the scan is interrupted**
(Ctrl-C), so no mission launches on an aborted run.

## Tests

```bash
python test_parse.py                # parser correctness (7 tests)
python test_stress.py               # adversarial / edge cases (59 tests)
```

- **test_parse.py** — builds real binary packets with `brping` and asserts the
  parser decodes them correctly: field round-trips, dB conversion math, packets
  split or concatenated across `feed()` calls, handshake messages, and
  `MAVLINK_WRAPPER` GPS extraction.
- **test_stress.py** — tries to crash or corrupt every component: garbage and
  truncated bytes, zero/edge-case sample counts, malformed `MAVLINK_WRAPPER`
  payloads, concurrent `VehicleState` access from 16 threads, malformed
  mavlink2rest messages, georeferencing edge cases (equator, poles, no fix), and
  `WaterfallDetector` edge cases (tiny images, variable widths, saturated/blank
  pings).

## Detection notes

Real seafloor backscatter has high dynamic range and a range-dependent (TVG)
brightness profile. The detector flat-fields each column (divides by its
along-track median) so the static range pattern is removed and targets stand
out locally — this is essential; a naive histogram-equalize amplifies seabed
speckle into thousands of false detections.

**CFAR (default detector).** Constant-False-Alarm-Rate detection estimates the
local seabed background from a band of reference cells around each pixel (a
guard band excludes the pixel's neighbourhood so a bright target can't bias its
own background) and flags the pixel when it exceeds `local_mean + k·local_std`.
This needs **no labels and no pre-recorded baseline** — the background is
estimated from the data itself, per pixel, so it adapts to whatever bottom the
boat is over. Two design choices make it work on real data:

- **dB domain.** CFAR runs on a dB-excess image — each column flat-fielded by
  *subtracting* its along-track median (TVG is additive in dB), so seabed ≈ 0 dB
  and a target is a positive dB excess. Working in dB rather than linear keeps a
  few hot clutter pixels from inflating the local variance and pushing the
  threshold out of reach of a genuine target.
- **Horizontal-band reference.** Background is estimated from range-direction
  (left/right) neighbours only. A target running along-track — a log at roughly
  constant range — always has clean seabed to its left and right whatever its
  length, so CFAR fires along its whole extent. A square reference window would
  let the target fill its own along-track reference cells and self-suppress its
  interior (the target would show *no* detection down its middle).

Window sums use an integral image (summed-area table) for O(1) per-pixel cost,
which is also why the optional `cfar_backend="torch"` GPU path on the edge
server is trivially parallel. The numpy and torch backends produce identical
results. Sensitivity is set by `cfar_k` (higher = fewer detections). Validated
on a real recording: the known along-track log is detected as a single tall
contact at ~14 anomalies/window.

**Shadow gating** (`shadow_gate=True`, off by default) requires a dark acoustic
shadow on a hit's far-range side. This rejects bright clutter for *proud*
objects (mines, rocks), but a flat-lying target like a half-buried log casts no
usable shadow, so the gate is off by default and the known log is kept.

**Box-merging** unions CFAR fragments within `cfar_merge_gap` px so an elongated
target reports as one contact rather than several.

GPU on the edge server: set `cfar_backend="torch"`. CFAR itself is light enough
to run real-time on CPU; the GPU's real value is classifying the surviving
candidates (a future step, once confirmed/dismissed detections accumulate into
labels) and survey mosaicking.

Known limitations / next steps:
- Seabed clutter (rocks/debris) is still detected; separating target *types*
  needs shape/shadow analysis or a classifier (no labels yet).
- The per-channel port/starboard side mapping (`CHANNEL_SIDE_OFFSET` in
  `main.py`) mirrors the XTF order but is unconfirmed against the live stream.
