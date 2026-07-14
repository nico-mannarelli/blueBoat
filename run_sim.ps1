# Sim test launcher for the two-mission pipeline (see TESTING.md Level 2).
# Sets every env var the sim needs, then runs main.py with a run.log capture.
#
# Usage (from the repo folder, with SITL + MAVProxy + mock_sonarlink running):
#   .\run_sim.ps1          # mission 1 ends at waypoint seq 4
#   .\run_sim.ps1 13       # mission 1 ends at waypoint seq 13
#
# NEVER use this on the real boat — the defaults here are sim-only.

$env:PYTHONUTF8 = "1"
$env:HOST = "127.0.0.1"
$env:MAVCON_URL = "tcp:127.0.0.1:5762"
$env:PLANNER_MIN_HITS = "1"
$env:RUN_AFTER = "py mavlink.py"
$env:MISSION1_END = if ($args[0]) { "$($args[0])" } else { "4" }

Write-Host "[run_sim] MISSION1_END=$env:MISSION1_END  PLANNER_MIN_HITS=$env:PLANNER_MIN_HITS  MAVCON_URL=$env:MAVCON_URL"
py main.py *>&1 | Tee-Object -FilePath run.log
