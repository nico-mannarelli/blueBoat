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

Requires Python 3.12+.

## Core pipeline

Data flows: **raw bytes → parsed pings → waterfall image → detections → lat/lon**.

| File | Role |
|------|------|
| `sonar_ws.py` | WebSocket client for SonarLink (`192.168.2.2:7077`). Fetches a session ID, sends `os_ping_params` to start pinging, streams raw bytes to a callback. |
| `sonar_parse.py` | Parses Cerulean Ping packets into dicts. Decodes `os_mono_profile` (sonar samples + `channel_number`) and `MAVLINK_WRAPPER` (embedded GPS). |
| `sonar_detect.py` | The detection engine (`WaterfallDetector`). Builds the flat-fielded waterfall image and runs Hough + ROI detection, NMS, nadir masking, and a contrast filter. **All detection logic lives here.** |
| `mavlink_client.py` | Connects to mavlink2rest (`:6040`) for GPS + heading, kept in a thread-safe `VehicleState`. |

`main.py`, `replay_xtf.py`, and `mock_sonarlink.py` are three different *sources*
feeding the same detector: the live boat, a recorded file, or a simulator.

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
```
Replays a SonarView `.xtf` through the exact production detector.

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
python simulate.py                  # noise + one bright target → waterfall.png
```

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
along-track median), percentile-stretches, and despeckles — this is essential;
a naive histogram-equalize amplifies seabed speckle into thousands of false
detections. Validated against a real recording: ~34x fewer false positives than
the original pipeline while still reliably detecting a known target.

Known limitations / next steps:
- The elongated targets fragment into several boxes (no shape-merge step yet).
- Seabed clutter (rocks/debris) is still detected; separating target *types*
  needs shape/shadow analysis.
- The per-channel port/starboard side mapping (`CHANNEL_SIDE_OFFSET` in
  `main.py`) mirrors the XTF order but is unconfirmed against the live stream.
