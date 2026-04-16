"""
Pull the best hyperparameter config for each 5lb/cl (event, set) combo
from the WandB project and write a JSON file that the submission script
can consume directly.
"""
import wandb
import json
import re
import os
import argparse

ENTITY = "jacoba-california-state-university-east-bay"
PROJECT = "humaid_aum_mixup_st_hpo"

EVENTS = [
    "california_wildfires_2018", "canada_wildfires_2016", "cyclone_idai_2019",
    "hurricane_dorian_2019", "hurricane_florence_2018", "hurricane_harvey_2017",
    "hurricane_irma_2017", "hurricane_maria_2017", "kaikoura_earthquake_2016",
    "kerala_floods_2018",
]
SETS = [1, 2, 3]

HP_KEYS = [
    "sup_epochs", "unsup_epochs", "sup_batch_size", "unsup_batch_size",
    "alpha", "T", "sample_size", "unsup_size", "temp_scaling",
    "sample_scheme", "N_base", "label_smoothing", "hidden_dropout_prob",
    "attention_probs_dropout_prob", "dense_dropout",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="best_5lb_hps.json")
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    sweeps = list(api.project(PROJECT, entity=ENTITY).sweeps())

    # Filter to 5lb sweeps only
    target_sweeps = {}
    for s in sweeps:
        m = re.match(r"aum_mixup_(.+)_5lbcl_set(\d+)", s.name)
        if not m:
            continue
        event, set_num = m.group(1), int(m.group(2))
        key = (event, set_num)
        # If multiple sweeps match (e.g. reruns), keep the one with more runs
        if key not in target_sweeps or len(list(s.runs)) > len(list(target_sweeps[key].runs)):
            target_sweeps[key] = s

    print(f"Found {len(target_sweeps)} matching 5lb sweeps")

    results = []
    missing = []

    for event in EVENTS:
        for set_num in SETS:
            key = (event, set_num)
            if key not in target_sweeps:
                print(f"⚠️  Missing sweep for {event} set{set_num}")
                missing.append(key)
                continue

            sweep = target_sweeps[key]
            runs = [r for r in sweep.runs if r.state in ("finished", "crashed", "failed")]

            if not runs:
                print(f"⚠️  No finished runs for {event} set{set_num}")
                missing.append(key)
                continue

            # Find best run by dev F1
            best_run = None
            best_dev = -1.0
            for r in runs:
                dev = r.summary.get("dev_macro-F1") or r.summary.get("dev_f1")
                if dev is not None and float(dev) > best_dev:
                    best_dev = float(dev)
                    best_run = r

            if best_run is None:
                print(f"⚠️  No run with dev F1 for {event} set{set_num}")
                missing.append(key)
                continue

            # Extract HP config
            config = best_run.config
            hp = {}
            for k in HP_KEYS:
                if k in config:
                    hp[k] = config[k]

            entry = {
                "event": event,
                "set_num": set_num,
                "lbcl": 5,
                "best_dev_f1": best_dev,
                "source_run_id": best_run.id,
                "source_sweep": sweep.name,
                "hyperparameters": hp,
            }
            results.append(entry)
            print(f"✅ {event} set{set_num}: dev_f1={best_dev:.4f} (run {best_run.id})")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Extracted {len(results)}/30 configs → {args.output}")
    if missing:
        print(f"Missing {len(missing)}: {missing}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
