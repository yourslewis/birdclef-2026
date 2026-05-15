#!/usr/bin/env bash
# Launch the public946 blended-teacher NFNet pseudo-label smoke only when trainer SSH is healthy.
# This intentionally avoids foreground SSH training runs: preflight fast, sync, launch nohup, print log/pid.

set -euo pipefail

TRAINER_HOST="${TRAINER_HOST:-yourslewis@192.168.0.10}"
REMOTE_REPO="${REMOTE_REPO:-~/birdclef-2026}"
REMOTE_VENV="${REMOTE_VENV:-~/kaggle_envs/s6e3/bin/python}"
CUDA_VISIBLE_DEVICES_REMOTE="${CUDA_VISIBLE_DEVICES_REMOTE:-0}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-8}"
LOCAL_CONFIG="${1:-configs/birdclef/pl_public946_sed85_rankblend15_nfnet_5s_lr1e4_smoke_20260515.json}"
REMOTE_CONFIG="${REMOTE_CONFIG:-$LOCAL_CONFIG}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_NAME="${LOG_NAME:-pl_public946_sed85_rankblend15_nfnet_5s_lr1e4_smoke_${STAMP}.log}"
REMOTE_LOG="logs/${LOG_NAME}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout="$CONNECT_TIMEOUT" -o ConnectionAttempts=1 -o ServerAliveInterval=5 -o ServerAliveCountMax=1)

if [[ ! -f "$LOCAL_CONFIG" ]]; then
  echo "missing local config: $LOCAL_CONFIG" >&2
  exit 2
fi

echo "[preflight] checking trainer SSH: ${TRAINER_HOST}" >&2
if ! ssh "${SSH_OPTS[@]}" "$TRAINER_HOST" 'echo trainer-ssh-ok' >/dev/null; then
  echo "[blocked] trainer SSH preflight failed; not launching remote GPU job" >&2
  exit 75
fi

echo "[sync] config -> ${TRAINER_HOST}:${REMOTE_REPO}/${LOCAL_CONFIG}" >&2
rsync -az -e "ssh ${SSH_OPTS[*]}" "$LOCAL_CONFIG" "$TRAINER_HOST:${REMOTE_REPO}/${LOCAL_CONFIG}"

echo "[launch] starting durable NFNet smoke on ${TRAINER_HOST}" >&2
ssh "${SSH_OPTS[@]}" "$TRAINER_HOST" "cd ${REMOTE_REPO} && mkdir -p logs && nohup env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_REMOTE} ${REMOTE_VENV} scripts/birdclef_pseudolabel_student_train.py --config ${REMOTE_CONFIG} > ${REMOTE_LOG} 2>&1 < /dev/null & echo \\$! ${REMOTE_LOG}"
