#!/usr/bin/env bash
# v10/9999 本机 Task2：纯策略，不加 score-seek。
# 对照 v5 无 seek 36.96、v7 39.70、v8 39.95、v9 39.93。禁止 --score-seek。
# 官方评分走 :9010（Cursor 常占 :9000）。不覆盖 v5–v9 脚本。不上报。
set -uo pipefail

EVAL_DIR=/home/dan/simulation/SouthGrid/src/examples/inference/g1_omnipicker
PYTHON=/home/dan/miniconda3/envs/orcalab_lerobot/bin/python
EPISODES="${EPISODES:-3}"
LOG_DIR="$EVAL_DIR/logs/v10_9999_noseek_${EPISODES}x4"
mkdir -p "$LOG_DIR"

export DISPLAY=:1
export PYTHONPATH=/home/dan/simulation/OrcaGym:/home/dan/simulation/SouthGrid/src
export NUMBA_DISABLE_JIT=1
export ORCA_SCORING_SERVER_URL=
export ORCA_SCORING_VIDEO_ENABLED=0
export ORCA_SCORING_BASE_URL=http://127.0.0.1:9010

echo "[$(date '+%H:%M:%S')] start v10 noseek ${EPISODES}x4 (hold-render-hz=0, local-only, :9010)" | tee "$LOG_DIR/runner.log"
cd "$EVAL_DIR"
"$PYTHON" -u eval_g1_omnipicker_lerobot.py \
  --task_config ../../dataCollection/common/example.yaml \
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
  --no-score-seek \
  --hold-render-hz 0 \
  --score-hold-s 0 \
  --min-score-span 22 \
  --p2-wait hold_at_third \
  --no_preview --no_head_video \
  > "$LOG_DIR/eval.log" 2>&1
rc=$?
echo "[$(date '+%H:%M:%S')] done exit=$rc" | tee -a "$LOG_DIR/runner.log"
exit $rc
