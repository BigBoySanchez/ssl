# HPO Implementation Status for Supervised Baseline (`bert_ft.py`)

## Goal
Enable HPO for the supervised baseline (`bert_ft.py`) using WandB sweeps, mirroring the `verifymatch` workflow (120 combos: 10 events × 4 lbcl × 3 sets).

## Accomplished
1.  **Scripts Created**:
    *   `supervised/make_sweep.py`: Generates `bertweet_sweep_ids.txt`.
    *   `supervised/make_container.sh`: Launcher for Docker agents (uses `bertweet_sweep_ids.txt`).
2.  **`supervised/bert_ft.py` Modifications**:
    *   **Auto Mode**: Added `--event`, `--lbcl`, `--set_num`, `--project_name`.
    *   **Dynamic Paths**: Automatically resolves paths to:
        *   Dev/Test: `../data/humaid/joined`
        *   Train: `../data/humaid/anh_4o_mini/sep/{lbcl}lb/{set_num}/labeled.tsv` (with fallback to `anh_4o`).
    *   **Event Filtering**: Implemented pandas-based in-memory filtering to ensure strict event matching (`df["event"] == args.event`).
    *   **WandB**: Integrated `wandb.init` and `wandb.log`.
    *   **Argparse Fixes**:
        *   Renamed list args to `--lrs_list`, `--epochs_list`, `--batch_sizes_list`.
        *   Added single-value HPO args: `--learning_rate`, `--epochs`, `--batch_size` (to fix "unrecognized arguments").
        *   Logic prioritizes: CLI Arg > WandB Config > List Default.

## Current State
*   `bert_ft.py` syntax verified.
*   `make_sweep.py` updated to save to `bertweet_sweep_ids.txt`.
*   `make_container.sh` updated to use `bertweet_sweep_ids.txt`.

## Next Steps
1.  **Launch**: Run `cd supervised && ./make_container.sh` to start agents.
2.  **Monitor**: Watch WandB project `humaid_supervised_hpo` (ensure project name aligns with user preference, currently set to `humaid_supervised_hpo`).
3.  **Verify**: Ensure paths resolve correctly on the specific execution machine (logic uses relative paths from `__file__`).
