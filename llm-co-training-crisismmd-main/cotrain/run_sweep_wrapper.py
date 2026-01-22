#!/usr/bin/env python3
"""
Wrapper script to run co-training for all 3 sets and average the results.
"""

import subprocess
import argparse
import re
import numpy as np
import wandb

def run_single_set(args, set_num):
    """Run the main script for a single set and return the metrics."""
    cmd = [
        "python", "main_bertweet.py",
        "--dataset", args.dataset,
        "--hf_model_id_short", args.hf_model_id_short,
        "--plm_id", args.plm_id,
        "--metric_combination", args.metric_combination,
        "--seed", str(args.seed),
        "--pseudo_label_dir", args.pseudo_label_dir,
        "--data_dir", args.data_dir,
        "--cuda_devices", args.cuda_devices,
        "--preds_file", args.preds_file,
        "--event", args.event,
        "--lbcl", args.lbcl,
        "--set_num", str(set_num),
        "--lr", str(args.lr),
        "--num_epochs", str(args.num_epochs),
        "--epoch_patience", str(args.epoch_patience)
    ]
    
    if args.setup_local_logging:
        cmd.insert(3, "--setup_local_logging")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # Parse the final results from output
    # Look for the line with F1, Accuracy, ECE
    match = re.search(r'F1: ([\d.]+), .* Accuracy: ([\d.]+), ECE: ([\d.]+)', output)
    if match:
        f1 = float(match.group(1))
        acc = float(match.group(2))
        ece = float(match.group(3))
        return f1, acc, ece
    else:
        print(f"Failed to parse results for set {set_num}")
        print(output)
        return None

def main():
    parser = argparse.ArgumentParser(description="Run co-training sweep over 3 sets")
    parser.add_argument("--dataset", type=str, default="humaid", help="Dataset name")
    parser.add_argument("--hf_model_id_short", type=str, default="N/A", help="Short HF model id")
    parser.add_argument("--plm_id", type=str, default="roberta-base", help="Backbone PLM id")
    parser.add_argument("--metric_combination", type=str, default="cv", help="Metric combination")
    parser.add_argument("--setup_local_logging", action="store_true", default=False, help="Setup local logging")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--pseudo_label_dir", type=str, default="anh_4o", help="Directory containing LLM pseudo labels")
    parser.add_argument("--data_dir", type=str, default="../../data", help="Base data directory")
    parser.add_argument("--cuda_devices", type=str, default="0,1", help="CUDA devices")
    parser.add_argument("--preds_file", type=str, default="preds.json", help="Predictions output file")
    parser.add_argument("--event", type=str, required=True, help="Event name")
    parser.add_argument("--lbcl", type=str, required=True, help="Labeled count per class")
    parser.add_argument("--set_num", type=str, help="Set number (not used in wrapper)")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--epoch_patience", type=int, required=True, help="Epoch patience")
    
    args = parser.parse_args()
    
    # Initialize wandb for this sweep run
    wandb.init(
        project="cotrain-hyperparameter-tuning",
        name=f"{args.event}_{args.lbcl}_lr{args.lr}_ep{args.num_epochs}_pat{args.epoch_patience}",
        config={
            "dataset": args.dataset,
            "hf_model_id_short": args.hf_model_id_short,
            "plm_id": args.plm_id,
            "metric_combination": args.metric_combination,
            "seed": args.seed,
            "pseudo_label_dir": args.pseudo_label_dir,
            "event": args.event,
            "lbcl": args.lbcl,
            "data_dir": args.data_dir,
            "cuda_devices": args.cuda_devices,
            "preds_file": args.preds_file,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "epoch_patience": args.epoch_patience
        }
    )
    
    results = []
    for set_num in [1, 2, 3]:
        print(f"Running set {set_num}")
        metrics = run_single_set(args, set_num)
        if metrics:
            f1, acc, ece = metrics
            results.append(metrics)
            # Log individual set results
            wandb.log({
                f"set_{set_num}_f1": f1,
                f"set_{set_num}_accuracy": acc,
                f"set_{set_num}_ece": ece
            })
        else:
            print(f"No results for set {set_num}")
    
    if results:
        # Calculate averages
        avg_f1 = np.mean([r[0] for r in results])
        avg_acc = np.mean([r[1] for r in results])
        avg_ece = np.mean([r[2] for r in results])
        
        # Log averages
        wandb.log({
            "avg_f1": avg_f1,
            "avg_accuracy": avg_acc,
            "avg_ece": avg_ece
        })
        
        print(f"Average F1: {avg_f1:.4f}, Average Accuracy: {avg_acc:.4f}, Average ECE: {avg_ece:.4f}")
    
    wandb.finish()

if __name__ == "__main__":
    main()