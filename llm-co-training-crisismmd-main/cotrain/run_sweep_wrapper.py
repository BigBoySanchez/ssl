#!/usr/bin/env python3
"""
Wrapper script to run co-training for all 3 sets and average the results.
"""

import subprocess
import argparse
import re
import numpy as np
import wandb

def run_single_set(event, lbcl, set_num, lr, num_epochs, epoch_patience):
    """Run the main script for a single set and return the metrics."""
    cmd = [
        "python", "main_bertweet.py",
        "--dataset", "humaid",
        "--plm_id", "roberta-base",
        "--hf_model_id_short", "N/A",
        "--metric_combination", "cv",
        "--setup_local_logging",
        "--seed", "1234",
        "--pseudo_label_dir", "anh_4o",
        "--data_dir", "../../data",
        "--cuda_devices", "0,1",
        "--preds_file", "preds.json",
        "--use_wandb",
        "--event", event,
        "--lbcl", lbcl,
        "--set_num", str(set_num),
        "--lr", str(lr),
        "--num_epochs", str(num_epochs),
        "--epoch_patience", str(epoch_patience)
    ]
    
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
    parser.add_argument("--event", type=str, required=True, help="Event name")
    parser.add_argument("--lbcl", type=str, required=True, help="Labeled count per class")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--epoch_patience", type=int, required=True, help="Epoch patience")
    
    args = parser.parse_args()
    
    # Initialize wandb for this sweep run
    wandb.init(
        project="cotrain-hyperparameter-tuning",
        name=f"{args.event}_{args.lbcl}_lr{args.lr}_ep{args.num_epochs}_pat{args.epoch_patience}",
        config={
            "event": args.event,
            "lbcl": args.lbcl,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "epoch_patience": args.epoch_patience
        }
    )
    
    results = []
    for set_num in [1, 2, 3]:
        print(f"Running set {set_num}")
        metrics = run_single_set(args.event, args.lbcl, set_num, args.lr, args.num_epochs, args.epoch_patience)
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