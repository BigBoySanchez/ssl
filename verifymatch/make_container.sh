#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ENTITY="jacoba-california-state-university-east-bay"
PROJECT="humaid_ssl"
IMAGE="cahsi/disaster-ssl:cuda12-py2.2"
MAX_GPUS=7
SWEEP_ID_FILE="/workspace/ssl/artifacts/sweep_ids.txt"   # one sweep ID per line

# ──────────────────────────────────────────────
# DATA / ARTIFACT PATHS
# ──────────────────────────────────────────────
DATA_MOUNT="~/data:/workspace/ssl/data"
ARTIFACT_MOUNT="~/artifacts:/workspace/ssl/artifacts"

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

# ──────────────────────────────────────────────
# LOAD SWEEPS
# ──────────────────────────────────────────────
if [[ ! -f "$SWEEP_ID_FILE" ]]; then
  echo "❌ Missing $SWEEP_ID_FILE"
  exit 1
fi
mapfile -t SWEEP_IDS < "$SWEEP_ID_FILE"
total_sweeps=${#SWEEP_IDS[@]}

echo "📋 Loaded $total_sweeps sweeps"
echo "🧠 Starting $MAX_GPUS GPU agents"
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
      -v ${ARTIFACT_MOUNT} \
      --name "${cname}" \
      "${IMAGE}" \
      bash -c "
        apt-get update -y && apt-get install git -y
        cd ./ssl &&
        git pull &&
        cd ./verifymatch &&
        echo '[Agent ${gpu_id}] Running sweep ${sweep_id} (${event} ${lbcl}lbcl)' &&
        wandb agent --count 30 ${ENTITY}/${PROJECT}/${sweep_id} &&
        echo '[Agent ${gpu_id}] Sweep ${sweep_id} finished.'
      "
}

# ──────────────────────────────────────────────
# INITIAL LAUNCH (fill GPUs)
# ──────────────────────────────────────────────
for ((gpu=0; gpu<MAX_GPUS; gpu++)); do
  event_idx=$(( gpu / ${#LBCLS[@]} % ${#EVENTS[@]} ))
  lbcl_idx=$(( gpu % ${#LBCLS[@]} ))
  event=${EVENTS[$event_idx]}
  lbcl=${LBCLS[$lbcl_idx]}
  sweep_idx=$(( gpu % total_sweeps ))
  sweep_id=${SWEEP_IDS[$sweep_idx]}
  launch_agent "$gpu" "$event" "$lbcl" "$sweep_id"
done

# ──────────────────────────────────────────────
# EVENT-DRIVEN MONITOR: KEEP CONTAINERS ALIVE (ORDERED)
# ──────────────────────────────────────────────
echo "📡 Watching for stopped containers..."

current_job=$MAX_GPUS  # we've already launched the first N agents

docker events --filter 'event=die' --format '{{.Actor.Attributes.name}}' |
while read -r cname; do
  if [[ $cname == cahsi-gpu* ]]; then
    gpu_id="${cname//[!0-9]/}"
    echo "⚡ ${cname} stopped → restarting agent (GPU ${gpu_id})..."

    # orderly iteration through event × lbcl × sweep combos
    event_idx=$(( current_job / ${#LBCLS[@]} % ${#EVENTS[@]} ))
    lbcl_idx=$(( current_job % ${#LBCLS[@]} ))
    sweep_idx=$(( current_job % total_sweeps ))

    event=${EVENTS[$event_idx]}
    lbcl=${LBCLS[$lbcl_idx]}
    sweep_id=${SWEEP_IDS[$sweep_idx]}

    # advance job pointer
    ((current_job++))

    # clean up and restart container
    docker rm -f "$cname" >/dev/null 2>&1 || true
    launch_agent "$gpu_id" "$event" "$lbcl" "$sweep_id"
  fi
done

