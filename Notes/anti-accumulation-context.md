# Anti-Accumulation Refactor Context

## Problem Statement
We encountered Out-Of-Memory (OOM) errors when running the co-training script in single-GPU mode.
- **Root Cause**: In single-GPU mode (`cuda_devices="0"` or similar), both classifiers (`model_1` and `model_2`) are instantiated on the exact same device.
- **Consequence**: This doubles the VRAM usage (parameters + gradients + optimizer states for 2 models), making it impossible to run reasonable batch sizes without crashing, even with gradient accumulation.

## The Solution: "Anti-Accumulation"
Instead of using software tricks (gradient accumulation) to simulate larger batches on a single choked GPU, we switched to a **hardware-based solution**.

1.  **Use 2 GPUs**: We leverage the server's multi-GPU capability.
    - `model_1` -> `cuda:0`
    - `model_2` -> `cuda:1`
    - This physically isolates the memory footprint of each model, halving the VRAM load per card.
2.  **Strip Accumulation Logic**: We removed the "auto-calculation" logic that tried to force an effective batch size of 64.
    - **Old Logic**: `accumulation_steps = 64 / BATCH_SIZE` (forced opaque training dynamics).
    - **New Logic**: `accumulation_steps = 1`.
    - **Effect**: `batch_size` now controls the *exact* number of samples per update. This is transparent and easier to tune.

## Code Changes Implemented

### 1. `cotrain/main_bertweet.py`
- Hardcoded accumulation steps to 1.
- Removed the automatic scaling logic.
- **Crucial**: The script's `set_environment` function already handles specific GPU assignment. If `torch.cuda.device_count() >= 2`, it automatically puts models on separate cards.

### 2. Sweep Infrastructure (`cotrain/run_sweep_wrapper.py`)
- Removed `--accumulation_steps` argument parsing.
- Stopped passing this flag to the subprocess.
- Removed it from `wandb.init` config.

### 3. Sweep Generation (`cotrain/initialize_sweeps.py`, `cotrain/generate_sweep.py`)
- Removed `accumulation_steps_min`, `accumulation_steps_max` from fixed parameters.
- Removed specific CLI arguments for accumulation limits.
- Removed `accumulation_steps` from the W&B sweep configuration dictionary.

## Docker Container Requirements (Next Steps)
For the next session (creating the Docker container):
- **GPU Requirement**: The container **MUST** have access to at least 2 GPUs.
    - Docker flag: `--gpus '"device=0,1"'` (or `all`).
- **Configuration**:
    - `cuda_devices` arg: `"0,1"`.
- **Expected Behavior**:
    - The code will spawn instances on logically distinct devices.
    - No manual `accumulation_steps` parameter needs to be passed.
    - Tuning focus: Purely on `batch_size` (e.g., finding the max batch size that fits on *one* card, effectively doubling throughput).
