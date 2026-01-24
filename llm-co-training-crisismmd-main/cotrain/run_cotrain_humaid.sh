#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ENTITY="jacoba-california-state-university-east-bay"
PROJECT="lg-cotrain-humaid"
IMAGE="cahsi-cotrain:test"
SWEEP_ID_FILE="sweep_ids.txt"

# Configurable GPU list
declare -a GPUS=(
  "0"
  "1"
  "2"
)
NUM_GPUS=${#GPUS[@]}

# ──────────────────────────────────────────────
# DATA & PATHS
# ──────────────────────────────────────────────
# User requested specific mounts
HOME_SSL_MOUNT="${HOME}/ssl:/workspace/ssl"

# ──────────────────────────────────────────────
# LOAD SWEEPS
# ──────────────────────────────────────────────
if [[ ! -f "$SWEEP_ID_FILE" ]]; then
  echo "❌ Missing $SWEEP_ID_FILE. Run initialize_sweeps.py first."
  exit 1
fi
mapfile -t SWEEP_IDS < "$SWEEP_ID_FILE"
TOTAL_SWEEPS=${#SWEEP_IDS[@]}

if (( TOTAL_SWEEPS == 0 )); then
  echo "❌ No sweep IDs found."
  exit 1
fi

echo "📋 Loaded $TOTAL_SWEEPS sweeps"
echo "🧠 Starting $NUM_GPUS GPU agents"
echo "───────────────────────────────────────────────"

# ──────────────────────────────────────────────
# FUNCTION TO LAUNCH AN AGENT CONTAINER
# ──────────────────────────────────────────────
launch_agent() {
    local gpu_idx=$1
    local gpu_id=${GPUS[$gpu_idx]}
    
    # Assign sweep ID round-robin
    local sweep_idx=$(( gpu_idx % TOTAL_SWEEPS ))
    local sweep_id=${SWEEP_IDS[$sweep_idx]}
    
    local cname="cotrain-test-${gpu_id}"

    echo "🚀 Preparing Worker for GPU ${gpu_id} → Sweep ${sweep_id}"
    
    # Check if WANDB_API_KEY is set
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "⚠️  WANDB_API_KEY is not set! The container will likely fail."
    fi

    # Dry Run: Echo commands instead of running
    echo "  [DRY RUN] docker rm -f \"${cname}\""
    echo "  [DRY RUN] docker run -d --gpus \"device=${gpu_id}\" \\"
    echo "    -v ${HOME_SSL_MOUNT} \\"
    echo "    -e WANDB_API_KEY=${WANDB_API_KEY:-<MISSING_KEY>} \\"
    echo "    --name \"${cname}\" \\"
    echo "    \"${IMAGE}\" \\"
    echo "    bash -c '"
    echo "      cd /workspace/ssl/llm-co-training-crisismmd-main/cotrain && \\"
    echo "      wandb agent --count 5 ${ENTITY}/${PROJECT}/${sweep_id}"
    echo "    '"
}

# ──────────────────────────────────────────────
# INITIAL LAUNCH
# ──────────────────────────────────────────────
for ((i=0; i<NUM_GPUS; i++)); do
  launch_agent "$i"
done

# NOTE: Original event-monitoring loop removed as we are just launching agents.

