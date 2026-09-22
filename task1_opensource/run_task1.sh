#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ORCALAB_PYTHON:-/home/dan/miniconda3/envs/orcalab/bin/python}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
MAP_DIR="${ROOT}/data/world_anchored_rtabmap_20260821T195241+0800"
if [[ $# -gt 0 && -d "$1" ]]; then
  MAP_DIR="$1"
  shift
fi
SPEED="${1:-0.8}"
shift || true

set +u
source "${ROS_SETUP}"
set -u
export PYTHONPATH="${ROOT}:${ROOT}/src:${PYTHONPATH:-}"
# Force local-only scoring even when the user's shell has a central URL set.
export ORCA_SCORING_SERVER_URL=""
export ORCA_SCORING_VIDEO_ENABLED="0"
export ORCA_SCORING_USE_SCREEN_CAPTURE="false"
[[ -f "${MAP_DIR}/rtabmap.db" ]] || { echo "missing ${MAP_DIR}/rtabmap.db" >&2; exit 2; }
[[ -f "${MAP_DIR}/mapping_manifest.json" ]] || { echo "missing mapping_manifest.json" >&2; exit 2; }
[[ -f "${MAP_DIR}/map/task1_rtabmap.yaml" ]] || { echo "missing map/task1_rtabmap.yaml" >&2; exit 2; }

SESSION_DIR="${ROOT}/data/validation_$(date +%Y%m%dT%H%M%S)"
exec "${PYTHON_BIN}" "${ROOT}/src/run_world_anchored_rtabmap_localization.py" \
  --reset-scene \
  --database "${MAP_DIR}/rtabmap.db" \
  --mapping-manifest "${MAP_DIR}/mapping_manifest.json" \
  --calibration "${MAP_DIR}/world_to_map_calibration.json" \
  --validation-map-yaml "${MAP_DIR}/map/task1_rtabmap.yaml" \
  --session-dir "${SESSION_DIR}" \
  --input global_validation \
  --speed "${SPEED}" --validation-speed "${SPEED}" \
  --rgbd-fps 30 --camera-render-hz 30 \
  --validation-route-timeout 180 \
  "$@"
