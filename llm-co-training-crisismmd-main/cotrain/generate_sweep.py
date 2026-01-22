#!/usr/bin/env python3
"""
Script to generate a Wandb sweep configuration YAML file for co-training hyperparameter tuning.
"""

import argparse
import yaml
import os

def generate_sweep_yaml(event, lbcl, set_num, output_file="sweep.yaml"):
    """Generate sweep configuration."""
    
    # Define hyperparameter ranges
    sweep_config = {
        "program": "run_sweep_wrapper.py",
        "method": "grid",  # or "random", "bayes"
        "metric": {
            "name": "avg_f1",
            "goal": "maximize"
        },
        "parameters": {
            "lr": {
                "values": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
            },
            "num_epochs": {
                "values": [5, 10, 15, 20]
            },
            "epoch_patience": {
                "values": [3, 5, 7, 10]
            }
        },
        "command": [
            "${env}",
            "python",
            "${program}",
            "--event", event,
            "--lbcl", lbcl,
            "--lr", "${lr}",
            "--num_epochs", "${num_epochs}",
            "--epoch_patience", "${epoch_patience}"
        ]
    }
    
    # Write to YAML file
    with open(output_file, 'w') as f:
        yaml.dump(sweep_config, f, default_flow_style=False)
    
    print(f"Sweep configuration saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Wandb sweep YAML for co-training")
    parser.add_argument("--event", type=str, required=True, help="Event name")
    parser.add_argument("--lbcl", type=str, required=True, help="Labeled count per class")
    parser.add_argument("--set_num", type=str, required=True, help="Set number")
    parser.add_argument("--output", type=str, default="sweep.yaml", help="Output YAML file")
    
    args = parser.parse_args()
    
    generate_sweep_yaml(args.event, args.lbcl, args.set_num, args.output)