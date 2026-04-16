#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# 5LB/CL RERUN - Direct runs using best HPs
# No sweeps. Just 30 deterministic runs.
# ──────────────────────────────────────────────
IMAGE="cahsi-cotrain:test"
MAX_GPUS=7
HP_FILE="best_5lb_hps.json"

if [[ ! -f "$HP_FILE" ]]; then
  echo "❌ Missing $HP_FILE — run extract_best_hps.py first"
  exit 1
fi

# Read the JSON array into individual entries
NUM_JOBS=$(python3 -c "import json; print(len(json.load(open('$HP_FILE'))))")
echo "📋 $NUM_JOBS direct runs to execute"

# ──────────────────────────────────────────────
# LAUNCH FUNCTION
# ──────────────────────────────────────────────
launch_run() {
    local gpu_id=$1
    local job_idx=$2
    local cname="cahsi-aum-5lb-gpu${gpu_id}"

    # Extract config for this job and write to a file
    # (avoids quoting issues when interpolating into docker bash -c)
    local config_file="run_config_${job_idx}.sh"
    python3 -c "
import json
jobs = json.load(open('$HP_FILE'))
j = jobs[$job_idx]
hp = j['hyperparameters']
event = j['event']
set_num = j['set_num']

args = f'--event {event} --lbcl 5 --set_num {set_num}'
for k, v in hp.items():
    args += f' --{k} {v}'

with open('$config_file', 'w') as f:
    f.write(args)
"

    if [[ ! -f "$config_file" ]]; then
      echo "❌ Failed to generate config for job $job_idx"
      return 1
    fi

    local config
    config=$(cat "$config_file")
    
    local event
    event=$(python3 -c "import json; print(json.load(open('$HP_FILE'))[$job_idx]['event'])")
    local set_num
    set_num=$(python3 -c "import json; print(json.load(open('$HP_FILE'))[$job_idx]['set_num'])")

    echo "🚀 GPU${gpu_id} → ${event} 5lbcl set${set_num} (job ${job_idx})"
    echo "   Config: ${config}"

    # Copy the config file into the mounted volume so the container can read it
    cp "$config_file" "${HOME}/ssl/AUM-ST-Mixup/${config_file}"

    docker run -d --gpus "device=${gpu_id}" \
      -e DEBUG="${DEBUG:-}" \
      -e HF_TOKEN="${HF_TOKEN}" \
      -e WANDB_API_KEY="${WANDB_API_KEY}" \
      -v ${HOME}/ssl:/workspace/ssl \
      -v /tmp/humaid_ssl:/workspace/ssl/artifacts \
      --name "${cname}" \
      "${IMAGE}" \
      bash -c '
        set -x
        cd /workspace/ssl/AUM-ST-Mixup
        pip install wandb torchmetrics aum==1.0.2 matplotlib
        CONFIG=$(cat /workspace/ssl/AUM-ST-Mixup/'"${config_file}"')
        echo "=== Running with args: $CONFIG ==="
        python run_aum_mixup_st.py $CONFIG 2>&1
        EXIT_CODE=$?
        echo "=== Exited with code: $EXIT_CODE ==="
        rm -f /workspace/ssl/AUM-ST-Mixup/'"${config_file}"'
      '
}

# ──────────────────────────────────────────────
# INITIAL LAUNCH
# ──────────────────────────────────────────────
next_job=0
for ((gpu=0; gpu<MAX_GPUS && next_job<NUM_JOBS; gpu++)); do
  launch_run "$gpu" "$next_job"
  ((next_job++))
done

active_agents=$((next_job < MAX_GPUS ? next_job : MAX_GPUS))

# ──────────────────────────────────────────────
# MONITOR AND REFILL
# ──────────────────────────────────────────────
echo "📡 Monitoring... ($active_agents active, $((NUM_JOBS - next_job)) queued)"

while read -r cname; do
  if [[ $cname == cahsi-aum-5lb-gpu* ]]; then
    gpu_id="${cname//[!0-9]/}"
    echo "⚡ ${cname} finished (GPU ${gpu_id})"
    ((active_agents--))

    if (( next_job < NUM_JOBS )); then
      docker rm -f "$cname" >/dev/null 2>&1 || true
      launch_run "$gpu_id" "$next_job"
      ((active_agents++))
      ((next_job++))
    fi

    echo "📊 Active: ${active_agents} | Done: $((next_job - active_agents))/${NUM_JOBS}"

    if (( active_agents == 0 )); then
      echo "🎉 All 5lb/cl reruns completed!"
      exit 0
    fi
  fi
done < <(docker events --filter 'event=die' --format '{{.Actor.Attributes.name}}')
