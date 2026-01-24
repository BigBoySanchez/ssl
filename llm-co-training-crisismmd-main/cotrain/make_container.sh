#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ENTITY="jacoba-california-state-university-east-bay"
PROJECT="lg-cotrain-humaid"
IMAGE="cahsi/disaster-ssl:cuda12-py2.2"

# ──────────────────────────────────────────────
# ARGS
# ──────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <GPU_ID> <SWEEP_ID>"
    exit 1
fi

GPU_ID=$1
SWEEP_ID=$2

# ──────────────────────────────────────────────
# DATA & PATHS
# ──────────────────────────────────────────────
# Match mounts from run_agents.sh
DATA_MOUNT="${HOME}/data:/workspace/ssl/data"
ARTIFACT_MOUNT="/data/${USER}:/workspace/ssl/artifacts"

# ──────────────────────────────────────────────
# CONTAINER SETUP
# ──────────────────────────────────────────────
CNAME="cotrain-manual-${GPU_ID}-${SWEEP_ID}"

echo "🚀 Launching manual agent: ${CNAME}"
echo "   GPU: ${GPU_ID}"
echo "   Sweep: ${SWEEP_ID}"

# Remove existing if any
docker rm -f "${CNAME}" >/dev/null 2>&1 || true

# Run container (One-Liner)
docker run -d \
    --gpus "device=${GPU_ID}" \
    -v ${DATA_MOUNT} \
    -v ${ARTIFACT_MOUNT} \
    --name "${CNAME}" \
    "${IMAGE}" \
    bash -c '
        mkdir -p /workspace/ssl/artifacts && \
        apt-get update -y && apt-get install -y --no-install-recommends git && \
        cd /workspace/ssl && \
        cd llm-co-training-crisismmd-main/cotrain && \
        echo "[Manual Agent] Starting sweep '${SWEEP_ID}' on GPU '${GPU_ID}'" && \
        wandb agent --count 1 '${ENTITY}'/'${PROJECT}'/'${SWEEP_ID}'
    '
