
import wandb
import argparse
import subprocess
import json
import os
import numpy as np

def run_trial(args, wandb_config):
    """
    Runs main_bertweet.py for set_num 1, 2, 3 and averages the results.
    """
    f1_scores = []
    accuracies = []
    eces = []
    
    # Base command construction
    cmd_base = [
        "python", "main_bertweet.py",
        "--dataset", "humaid",
        "--hf_model_id_short", "roberta", # fixed for now or from config? user said "event + lbcl combination should have their own sweep", so these might be fixed or passed.
        "--plm_id", wandb_config.get("plm_id", "bert-tweet"),
        "--event", args.event,
        "--lbcl", str(args.lbcl),
        "--exp_name", f"sweep_{args.event}_{args.lbcl}",
        # Hyperparameters from sweep config
        "--seed", str(wandb_config.get("seed", 1234)),
        # Add other potential hyperparameters here if they are in main_bertweet.py arguments
        # For example, learning_rate is hardcoded in main_bertweet.py currently, 
        # so if we want to tune it, we need to add it as an argument to main_bertweet.py first.
        # Wait, main_bertweet.py has:
        # training_params = { 'learning_rate': 1e-5, ... }
        # It does NOT take learning_rate from args.
        # I MUST fix main_bertweet.py to accept learning_rate and other params if I want to tune them.
    ]
    
    # Checking what params to tune.
    # User said "Hyper Parameter Tuning". Usually includes LR, Batch Size, etc.
    # main_bertweet.py line 441 has hardcoded learning_rate.
    
    # Let's assume for now I will need to patch main_bertweet.py to accept more args 
    # OR the user is happy with what is there? 
    # "I need to do Hyper Parameter Tuning now." -> Implies modifying code to accept params.
    
    # I will first finish this script assuming main_bertweet.py accepts them, 
    # then I will go back and add those args to main_bertweet.py.
    
    if "learning_rate" in wandb_config:
        cmd_base.extend(["--learning_rate", str(wandb_config["learning_rate"])])
        
    if "epoch_patience" in wandb_config:
        cmd_base.extend(["--epoch_patience", str(wandb_config["epoch_patience"])])
        
    if "num_epochs" in wandb_config:
        cmd_base.extend(["--num_epochs", str(wandb_config["num_epochs"])])

    # Accumulate results
    all_preds = {}

    for set_num in [1, 2, 3]:
        print(f"--- Running Set {set_num} ---")
        preds_file = f"preds_set_{set_num}.json"
        
        cmd = cmd_base + [
            "--set_num", str(set_num),
            "--preds_file", preds_file
        ]
        
        # print("Running command:", " ".join(cmd))
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            output = result.stdout
            # print(output) # Optional: print output to see training progress logs
            
            # Parse results from stdout
            found_metrics = False
            for line in output.splitlines():
                if line.startswith("JSON_OUTPUT:"):
                    try:
                        metrics_str = line.replace("JSON_OUTPUT:", "").strip()
                        res = json.loads(metrics_str)
                        f1_scores.append(res['f1'])
                        accuracies.append(res['accuracy'])
                        eces.append(res['ece'])
                        found_metrics = True
                        break
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from line: {line}")
            
            if not found_metrics:
                print(f"Warning: No metrics found in output for set {set_num}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running set {set_num}: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            continue

        # Read and artifact preds
        if os.path.exists(preds_file):
            artifact = wandb.Artifact(f"preds_set_{set_num}", type="predictions")
            artifact.add_file(preds_file)
            wandb.log_artifact(artifact)
            
            # Clean up preds file to save space/clutter
            os.remove(preds_file)
            
    # Compute averages
    if f1_scores:
        avg_f1 = np.mean(f1_scores)
        avg_acc = np.mean(accuracies)
        avg_ece = np.mean(eces)
        
        wandb.log({
            "f1": avg_f1,
            "accuracy": avg_acc,
            "ece": avg_ece,
            "f1_std": np.std(f1_scores),
            "ece_std": np.std(eces)
        })
        print(f"Final Averaged Results: F1={avg_f1:.4f}, Acc={avg_acc:.4f}, ECE={avg_ece:.4f}")
    else:
        print("No results collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=str, required=True)
    parser.add_argument("--lbcl", type=str, required=True)
    args = parser.parse_args()

    wandb.init()
    run_trial(args, wandb.config)
