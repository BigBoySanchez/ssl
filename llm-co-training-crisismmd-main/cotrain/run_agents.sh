#!/usr/bin/env bash
set -euo pipefail

# Configuration
ENTITY="jacoba-california-state-university-east-bay"
PROJECT="lg-cotrain-humaid"
SWEEP_FILE="sweep_ids.txt"
COUNT=5

# Workers Configuration
# Format: "GPU_IDS"
WORKERS=(
  "0,1"
  "2,3"
  "4,5"
)

if [[ ! -f "$SWEEP_FILE" ]]; then
  echo "❌ Error: $SWEEP_FILE not found. Run initialize_sweeps.py first."
  exit 1
fi

echo "📋 Loading sweep IDs from $SWEEP_FILE..."
mapfile -t SWEEP_IDS < "$SWEEP_FILE"
NUM_SWEEPS=${#SWEEP_IDS[@]}

if (( NUM_SWEEPS == 0 )); then
  echo "❌ Error: No sweep IDs found in $SWEEP_FILE."
  exit 1
fi

echo "🚀 Starting ${#WORKERS[@]} worker processes looping over $NUM_SWEEPS sweeps indefinitely..."
echo "---------------------------------------------------"

# Function for a single worker
run_worker() {
  local worker_id=$1
  local gpu_ids=$2
  
  echo "[Worker $worker_id] Started on GPUs $gpu_ids"
  
  # Infinite loop over the sweeps
  while true; do
    for (( i=0; i<NUM_SWEEPS; i++ )); do
      local sweep_id="${SWEEP_IDS[$i]}"
      local full_sweep_path="${ENTITY}/${PROJECT}/${sweep_id}"
      
      echo "[Worker $worker_id] Running sweep $sweep_id ($((i+1))/$NUM_SWEEPS)..."
      
      # Run wandb agent
      # We explicitly set CUDA_VISIBLE_DEVICES for this agent process
      CUDA_VISIBLE_DEVICES="$gpu_ids" wandb agent \
        --count "$COUNT" \
        "$full_sweep_path" \
        > "agent_${worker_id}_sweep_${sweep_id}.log" 2>&1
        
      echo "[Worker $worker_id] Finished sweep $sweep_id batch."
    done
    echo "[Worker $worker_id] Completed one full pass of all sweeps. Restarting loop..."
  done
}

# Launch workers in background
pids=()
for i in "${!WORKERS[@]}"; do
  gpu_config="${WORKERS[$i]}"
  run_worker "$i" "$gpu_config" &
  pids+=($!)
done

# Wait for all workers (they basically run forever until killed)
wait
