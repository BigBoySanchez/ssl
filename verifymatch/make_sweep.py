import wandb, subprocess, copy, os

# ───────────────────────────────
# Core sweep configuration
# ───────────────────────────────
BASE_SWEEP = {
    "name": "humaid_ssl_sweep",
    "program": "train.py",
    "method": "grid",
    "metric": {
        "name": "dev_macro-F1",   # your training script logs this at the end
        "goal": "maximize"
    },
    "parameters": {
        # === optimization hyperparameters ===
        "learning_rate": {"values": [2e-5, 2e-6, 4e-5]},
        "weight_decay": {"values": [0.0]},
        "batch_size": {"values": [32, 16, 8]},
        "epochs": {"values": [18, 12, 8]},

        # === semi-supervised control ===
        "T": {"values": [0.5]},           # temperature
        "mixup_loss_weight": {"values": [1.0]}, # consistency weight

        # === stability & regularization ===
        "label_smoothing": {"values": [0.3]},
        "max_grad_norm": {"values": [1.0]},
        "th": {"values": [0.7]},                # pseudo-label threshold
        "pseudo_label_by_normalized": {"values": [False]},
        "unlabeled_batch_size": {"values": [32]},

        # === fixed metadata ===
        "task": {"value": "HumAID"},
        "model": {"value": "vinai/bertweet-base"},
        "max_seq_length": {"value": 128},
        
        # placeholders to be overwritten
        "set_num": {"values": [1]},
        "event": {"value": "placeholder"},
        "lbcl": {"value": 5},
    }
}

# ───────────────────────────────
# Fill in your events + lbcl sizes here
# ───────────────────────────────
ENTITY = "jacoba-california-state-university-east-bay"

EVENTS = [
    "california_wildfires_2018",
    "canada_wildfires_2016",
    "cyclone_idai_2019",
    "hurricane_dorian_2019",
    "hurricane_florence_2018",
    "hurricane_harvey_2017",
    "hurricane_irma_2017",
    "hurricane_maria_2017",
    "kaikoura_earthquake_2016",
    "kerala_floods_2018",
]
LBCL_SIZES = [
    5, 
    10, 
    25, 
    50,
]
SET_NUMS = [1, 2, 3]

# ───────────────────────────────
# Sweep creation loop
# ───────────────────────────────
ids = []
for lbcl in LBCL_SIZES:
    for event in EVENTS:
        for set_num in SET_NUMS:
            sweep_cfg = copy.deepcopy(BASE_SWEEP)
            sweep_cfg["name"] = f"{event}_{lbcl}lbcl_set{set_num}"
            sweep_cfg["description"] = (
                f"Grid search for {event} ({lbcl}lbcl) set{set_num}"
            )
            
            # Inject fixed values for this specific sweep
            # Since it's a grid search for hyparams, these become single-value "grids"
            # forcing this sweep to only run for this specific dataset configuration
            sweep_cfg["parameters"]["set_num"] = {"values": [set_num]}
            sweep_cfg["parameters"]["event"] = {"value": event}
            sweep_cfg["parameters"]["lbcl"] = {"value": lbcl}

            project = f"humaid_ssl_category_match"

            sweep_id = wandb.sweep(sweep=sweep_cfg, project=project, entity=ENTITY)
            ids.append(sweep_id)
            print(f"🌀 Created sweep: {event} {lbcl}lbcl set{set_num} → {sweep_id}")

print(f"Total sweeps created: {len(ids)}")
print(f"ids: {ids}")
