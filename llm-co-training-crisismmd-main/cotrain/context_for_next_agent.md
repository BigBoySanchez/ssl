# Handover Context for Next Agent

## Objective
Run and debug the `llm-co-training-crisismmd-main/cotrain/run_tests.ps1` script using the newly integrated HumAID dataset.

## Current State
- **Data Location**: `d:\Downloads\Git-Stuff\ssl\data\humaid`
    - **Originals**: `dev/text_only.json`, `test/text_only.json`
    - **Pseudo-labels Source**: `anh_4o/all_events.tsv` (TSV format)
- **Environment**: Micromamba environment `humaid_env` (Python 3.10) is set up with all dependencies.
- **Code Changes**:
    - `data_utils.py`: Added `load_humaid_dataset` to handle TSV loading and 10-class mapping.
    - `data_processor.py`: Updated `TextOnlyProcessor` to use dynamic label maps.
- **Verification**: `verify_load.py` confirms that 44,373 records can be loaded successfully from the TSV.

## Key Data Schema (in `all_events.tsv`)
- `class_label`: **Gold Label** (String)
- `label`: **Pseudo Label** (String)
- `tweet_id`: Unique ID
- `tweet_text`: Input text
- **Mapping**: Both strings must be mapped using `get_humaid_label_map()` (0-9).

## Verification Details
The script `verify_load.py` in `cotrain/` can be used to re-verify the loader.

## Recommendations for Next Steps
1.  **Activate Env**: `micromamba activate humaid_env`
2.  **Debug run_tests.ps1**: The script currently uses dummy data generation which was replaced by real data paths in `data_utils.py`. The next agent should ensure `run_tests.ps1` correctly points to the real `data` directory and `pseudo_labels` subdirectory.
3.  **Pathing**: Note that `data_dir` in the code often expects a relative path like `../../data` if running from `cotrain/`.
