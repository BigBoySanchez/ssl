#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ENTITY="jacoba-california-state-university-east-bay"
PROJECT="lg-cotrain-humaid"
IMAGE="cahsi/disaster-ssl:cuda12-py2.2"
SWEEP_ID_FILE="sweep_ids.txt"
COUNT=5

# Workers: define sets of GPUs. 
# We have 3 workers: 0 (gpus 0,1), 1 (gpus 2,3), 2 (gpus 4,5)
declare -a WORKERS=(
  "0,1"
  "2,3"
  "4,5"
)
NUM_WORKERS=${#WORKERS[@]}

# ──────────────────────────────────────────────
# DATA & PATHS
# ──────────────────────────────────────────────
DATA_MOUNT="${HOME}/data:/workspace/ssl/data"
ARTIFACT_MOUNT="/data/${USER}:/workspace/ssl/artifacts"

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
echo "🧠 Starting $NUM_WORKERS GPU agents"
echo "───────────────────────────────────────────────"

# ──────────────────────────────────────────────
# FUNCTION TO LAUNCH AN AGENT CONTAINER
# ──────────────────────────────────────────────
launch_agent() {
    local worker_idx=$1
    local sweep_idx=$2
    
    # Calculate sweep ID using modulo for infinite looping
    local safe_sweep_idx=$(( sweep_idx % TOTAL_SWEEPS ))
    local sweep_id=${SWEEP_IDS[$safe_sweep_idx]}
    local gpu_ids=${WORKERS[$worker_idx]}
    
    # Unique container name: cotrain-workerX-sweepY
    # Adding timestamp or random suffix might refer to unique job instances, 
    # but here we use fixed worker names so we can track them easily.
    # Actually, make_container uses distinct names per job capability.
    # We will use "cotrain-worker-${worker_idx}" to identify the slot.
    local cname="cotrain-worker-${worker_idx}"

    echo "🚀 Launching Worker ${worker_idx} (GPUs ${gpu_ids}) → Sweep ${sweep_id} (${safe_sweep_idx}/${TOTAL_SWEEPS})"
    
    # Remove existing container if it exists (cleanup from previous run)
    docker rm -f "${cname}" >/dev/null 2>&1 || true

    docker run -d --gpus "device=${gpu_ids}" \
      -v ${DATA_MOUNT} \
      -v ${ARTIFACT_MOUNT} \
      --name "${cname}" \
      "${IMAGE}" \
      bash -c '
        mkdir -p /workspace/ssl/artifacts && \
        apt-get update -y && apt-get install -y --no-install-recommends git && \
        cd /workspace/ssl && \
        # Fetch latest code. Assuming the repo is "llm-co-training-crisismmd-main" or similar.
        # Since we do not know the exact git remote URL or if it is mounted, 
        # we will assume the standard flow: git fetch if it exists, or likely just use what is there.
        # Wait, make_container.sh does: git fetch origin && git reset --hard origin/main
        # We will assume valid git repo in /workspace/ssl or we might need to clone it.
        # If the container image already has the repo...
        # Let us try to update strictly if possible, otherwise just run.
        if [ -d .git ]; then git fetch origin && git reset --hard origin/main; fi && \
        cd llm-co-training-crisismmd-main/cotrain && \
        echo "[Worker] Starting agent for sweep '${sweep_idx}' ('${sweep_id}')" && \
        wandb agent --count '${COUNT}' '${ENTITY}'/'${PROJECT}'/'${sweep_id}' && \
        echo "[Worker] Finished batch."
    '
}

# ──────────────────────────────────────────────
# INITIAL LAUNCH
# ──────────────────────────────────────────────
# We need to map worker_idx to a "current sweep index".
# We'll maintain the state of "next job index" globally.
CURRENT_JOB_IDX=0

for ((i=0; i<NUM_WORKERS; i++)); do
  launch_agent "$i" "$CURRENT_JOB_IDX"
  ((CURRENT_JOB_IDX++))
done

# ──────────────────────────────────────────────
# EVENT-DRIVEN MONITOR: KEEP CONTAINERS ALIVE
# ──────────────────────────────────────────────
echo "📡 Watching for stopped containers..."

# Listen for 'die' events from our specific containers
docker events --filter 'event=die' --format '{{.Actor.Attributes.name}}' |
while read -r cname; do
  if [[ $cname == cotrain-worker-* ]]; then
    # Extract worker index from name "cotrain-worker-X"
    worker_idx="${cname##*-}"
    
    echo "⚡ ${cname} stopped → restarting with next sweep..."
    
    # Launch next job
    launch_agent "$worker_idx" "$CURRENT_JOB_IDX"
    
    ((CURRENT_JOB_IDX++))
    
    # Optional: Removing the dead container is done inside launch_agent via rm -f
    # But we can also prune here if we wanted. launch_agent does it.
  fi
done
