#!/usr/bin/env bash

EVENT_NAMES=(
  california_wildfires_2018
  canada_wildfires_2016    
  cyclone_idai_2019        
  hurricane_dorian_2019    
  hurricane_florence_2018  
  hurricane_harvey_2017    
  hurricane_irma_2017
  hurricane_maria_2017
  kaikoura_earthquake_2016
  kerala_floods_2018 
)

LBCL_VALUES=(
  5
  10
  25
  50
)

mkdir -p logs


# Flatten tasks into an array
TASKS=()
for event_name in "${EVENT_NAMES[@]}"; do
  for lbcl in "${LBCL_VALUES[@]}"; do
    TASKS+=("$event_name $lbcl")
  done
done

# GPU Management
GPU_PAIRS=("0,1" "2,3" "4,5")
SLOT_PIDS=(0 0 0) # Store PIDs for each slot (0 means free)

echo "Total tasks: ${#TASKS[@]}"

# Process loop
IDX=0
NUM_TASKS=${#TASKS[@]}

while [ $IDX -lt $NUM_TASKS ]; do
  LAUNCHED=false
  
  for i in "${!GPU_PAIRS[@]}"; do
    PID=${SLOT_PIDS[$i]}
    
    # Check if slot is free (PID=0 or process dead)
    if [ "$PID" -eq 0 ] || ! kill -0 "$PID" 2>/dev/null; then
      # Slot is free, launch next task
      
      # Parse task
      TASK="${TASKS[$IDX]}"
      read -r event_name lbcl <<< "$TASK"
      
      DEVICE=${GPU_PAIRS[$i]}
      LOG_FILE="logs/${event_name}_${lbcl}.log"
      
      echo "[Task $((IDX+1))/$NUM_TASKS] Starting $event_name (lbcl=$lbcl) on GPUs $DEVICE. Log: $LOG_FILE"
      
      python main_bertweet.py \
        --dataset humaid \
        --hf_model_id_short N/A \
        --plm_id roberta-base \
        --metric_combination cv \
        --setup_local_logging \
        --seed 1234 \
        --pseudo_label_dir anh_4o \
        --event "$event_name" \
        --lbcl "${lbcl}" \
        --set_num 1 \
        --data_dir ../../data \
        --cuda_devices="$DEVICE" \
        > "$LOG_FILE" 2>&1 &
      
      # Save PID
      SLOT_PIDS[$i]=$!
      
      IDX=$((IDX+1))
      LAUNCHED=true
      
      # Break if we've scheduled the last task
      if [ $IDX -ge $NUM_TASKS ]; then break; fi
    fi
  done
  
  # If no task was launched in this pass (all slots busy), wait a bit
  if [ "$LAUNCHED" = false ]; then
    sleep 5
  else
    # Small sleep even if launched, to avoid race conditions or hammering
    sleep 1
  fi
done

echo "All tasks scheduled. Waiting for remaining jobs..."
wait
echo "All done."
