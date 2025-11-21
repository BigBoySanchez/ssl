#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ENTITY="jacoba-california-state-university-east-bay"
PROJECT="humaid_ssl"
IMAGE="cahsi/disaster-ssl:cuda12-py2.2"
MAX_GPUS=7
SWEEP_ID_FILE="sweep_ids.txt"   # one sweep ID per line
INCREMENT=4                     # stride over (event×lbcl) combos
START_OFFSET=3                  # start combo index (0-based) before stride

# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
DATA_MOUNT="${HOME}/data:/workspace/ssl/data"

# ──────────────────────────────────────────────
# EVENTS AND LBCL COMBOS
# ──────────────────────────────────────────────
declare -a EVENTS=(
  "california_wildfires_2018"
  "canada_wildfires_2016"
  "cyclone_idai_2019"
  "hurricane_dorian_2019"
  "hurricane_florence_2018"
  "hurricane_harvey_2017"
  "hurricane_irma_2017"
  "hurricane_maria_2017"
  "kaikoura_earthquake_2016"
  "kerala_floods_2018"
)
declare -a LBCLS=(5 10 25 50)

NUM_EVENTS=${#EVENTS[@]}     # 10
NUM_LBCLS=${#LBCLS[@]}       # 4
NUM_COMBOS=$(( NUM_EVENTS * NUM_LBCLS ))  # 40

# ──────────────────────────────────────────────
# LOAD SWEEPS
# ──────────────────────────────────────────────
if [[ ! -f "$SWEEP_ID_FILE" ]]; then
  echo "❌ Missing $SWEEP_ID_FILE"
  exit 1
fi
mapfile -t SWEEP_IDS < "$SWEEP_ID_FILE"
total_sweeps=${#SWEEP_IDS[@]}

# Require at least one sweep per (event, lbcl) combo
if (( total_sweeps < NUM_COMBOS )); then
  echo "❌ Need at least ${NUM_COMBOS} sweep IDs (have ${total_sweeps})."
  exit 1
fi

echo "📋 Loaded $total_sweeps sweeps"
echo "🧠 Starting $MAX_GPUS GPU agents (offset ${START_OFFSET}, increment ${INCREMENT})"
echo "───────────────────────────────────────────────"

# ──────────────────────────────────────────────
# FUNCTION TO LAUNCH AN AGENT CONTAINER
# ──────────────────────────────────────────────
launch_agent() {
    local gpu_id=$1
    local event=$2
    local lbcl=$3
    local sweep_id=$4
    local cname="cahsi-gpu${gpu_id}"

    echo "🚀 Launching ${cname} → ${event} ${lbcl}lbcl (sweep ${sweep_id})"
    docker run -d --gpus "device=${gpu_id}" \
      -e EVENT_NAME="${event}" \
      -e LBCL_SIZE="${lbcl}" \
      -v ${DATA_MOUNT} \
      -v /data/${USER}:/workspace/ssl/artifacts \
      --name "${cname}" \
      "${IMAGE}" \
      bash -c '
        mkdir -p /workspace/ssl/artifacts && \
        apt-get update -y && apt-get install -y --no-install-recommends git && \
        cd /workspace/ssl && \
        git fetch origin && git reset --hard origin/main && \
        cd verifymatch && \
        echo "[Agent '${gpu_id}'] Running sweep '${sweep_id}' ('${event}' '${lbcl}'lbcl)" && \
        wandb agent --count 10 '${ENTITY}'/'${PROJECT}'/'${sweep_id}' && \
        echo "[Agent '${gpu_id}'] Sweep '${sweep_id}' finished."
    '
}

# Helper: compute (event_idx, lbcl_idx) from a linear combo index
compute_indices() {
  local combo_idx=$1
  # wrap combo index across all combos
  local wrapped_idx=$(( combo_idx % NUM_COMBOS ))
  EVENT_IDX=$(( (wrapped_idx / NUM_LBCLS) % NUM_EVENTS ))
  LBCL_IDX=$(( wrapped_idx % NUM_LBCLS ))
}

# ──────────────────────────────────────────────
# INITIAL LAUNCH (fill GPUs)
# ──────────────────────────────────────────────
for ((gpu=0; gpu<MAX_GPUS; gpu++)); do
  combo_idx=$(( START_OFFSET + gpu * INCREMENT ))
  compute_indices "$combo_idx"
  event=${EVENTS[$EVENT_IDX]}
  lbcl=${LBCLS[$LBCL_IDX]}

  # 🔒 Enforced mapping: sweep_ids[event_idx * NUM_LBCLS + lbcl_idx]
  sweep_flat_idx=$(( EVENT_IDX * NUM_LBCLS + LBCL_IDX ))
  if (( sweep_flat_idx >= total_sweeps )); then
    echo "❌ sweep index ${sweep_flat_idx} out of range (have ${total_sweeps})."
    exit 1
  fi
  sweep_id=${SWEEP_IDS[$sweep_flat_idx]}

  launch_agent "$gpu" "$event" "$lbcl" "$sweep_id"
done

# ──────────────────────────────────────────────
# EVENT-DRIVEN MONITOR: KEEP CONTAINERS ALIVE
# ──────────────────────────────────────────────
echo "📡 Watching for stopped containers..."

current_job=$MAX_GPUS  # after initial batch

docker events --filter 'event=die' --format '{{.Actor.Attributes.name}}' |
while read -r cname; do
  if [[ $cname == cahsi-gpu* ]]; then
    gpu_id="${cname//[!0-9]/}"
    echo "⚡ ${cname} stopped → restarting agent (GPU ${gpu_id})..."

    combo_idx=$(( START_OFFSET + current_job * INCREMENT ))
    compute_indices "$combo_idx"
    event=${EVENTS[$EVENT_IDX]}
    lbcl=${LBCLS[$LBCL_IDX]}

    # 🔒 Enforced mapping: sweep_ids[event_idx * NUM_LBCLS + lbcl_idx]
    sweep_flat_idx=$(( EVENT_IDX * NUM_LBCLS + LBCL_IDX ))
    if (( sweep_flat_idx >= total_sweeps )); then
      echo "❌ sweep index ${sweep_flat_idx} out of range (have ${total_sweeps})."
      exit 1
    fi
    sweep_id=${SWEEP_IDS[$sweep_flat_idx]}

    ((current_job++))

    docker rm -f "$cname" >/dev/null 2>&1 || true
    launch_agent "$gpu_id" "$event" "$lbcl" "$sweep_id"
  fi
done
