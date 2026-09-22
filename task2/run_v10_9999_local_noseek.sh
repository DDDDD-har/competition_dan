#!/usr/bin/env bash
# v10/9999 本机 Task2 纯推理。
# 策略动作原样执行，不改末端、不等待凑分。
# 先另开终端把策略服务起在 localhost:8010，再跑本脚本。
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
EPISODES="${EPISODES:-1}"
LOG_DIR="$ROOT/logs/v10_9999_${EPISODES}x4"
mkdir -p "$LOG_DIR"

export DISPLAY="${DISPLAY:-:1}"
# The task-specific client lives here, but its OrcaLab runtime modules live in
# the full SouthGrid source tree. Auto-detect the local checkout when possible;
# SOUTHGRID_SRC remains the override for other machines.
if [[ -z "${SOUTHGRID_SRC:-}" ]]; then
  for candidate in \
    "$ROOT" \
    "$ROOT/../src" \
    "$ROOT/../../SouthGrid/src" \
    "/home/dan/simulation/SouthGrid/src"; do
    if [[ -f "$candidate/conf/g1_omnipicker_conf.py" ]]; then
      SOUTHGRID_SRC="$candidate"
      break
    fi
  done
fi
if [[ -z "${SOUTHGRID_SRC:-}" ]]; then
  echo "找不到 SouthGrid/src。请设置 SOUTHGRID_SRC 为包含 conf/g1_omnipicker_conf.py 的 src 目录" >&2
  exit 1
fi
if [[ ! -f "$SOUTHGRID_SRC/conf/g1_omnipicker_conf.py" ]]; then
  echo "SOUTHGRID_SRC 不是有效的 SouthGrid/src：$SOUTHGRID_SRC" >&2
  echo "需要存在：$SOUTHGRID_SRC/conf/g1_omnipicker_conf.py" >&2
  exit 1
fi
OPENPI_CLIENT_SRC="${OPENPI_CLIENT_SRC:-}"
if [[ -z "$OPENPI_CLIENT_SRC" && -f "/home/dan/simulation/openpi/packages/openpi-client/src/openpi_client/__init__.py" ]]; then
  OPENPI_CLIENT_SRC="/home/dan/simulation/openpi/packages/openpi-client/src"
fi
if [[ -n "${ORCA_GYM_ROOT:-}" ]]; then
  export PYTHONPATH="${ORCA_GYM_ROOT}:${SOUTHGRID_SRC}${OPENPI_CLIENT_SRC:+:$OPENPI_CLIENT_SRC}${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="${SOUTHGRID_SRC}${OPENPI_CLIENT_SRC:+:$OPENPI_CLIENT_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi
export NUMBA_DISABLE_JIT=1
export SOUTHGRID_SRC
export OPENPI_CLIENT_SRC
export ORCA_SCORING_SERVER_URL=
export ORCA_SCORING_VIDEO_ENABLED=0

echo "[$(date '+%H:%M:%S')] start v10 pure ${EPISODES}x4" | tee "$LOG_DIR/runner.log"
cd "$ROOT"
"$PYTHON" -u eval_g1_omnipicker_lerobot.py \
  --host localhost --port 8010 \
  --exec_horizon 50 \
  --action_repeat 10 \
  --early_stop_on_touch \
  --max_steps 2000 \
  --episodes "$EPISODES" \
  --targets red green blue yellow \
  --task-id task2_button_press \
  --robot-id g1_omnipicker \
  --local-only \
  --no_preview --no_head_video \
  > "$LOG_DIR/eval.log" 2>&1
rc=$?
echo "[$(date '+%H:%M:%S')] done exit=$rc" | tee -a "$LOG_DIR/runner.log"
exit $rc
