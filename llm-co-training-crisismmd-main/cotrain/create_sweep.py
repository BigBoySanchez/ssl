import argparse
import yaml
import subprocess
import os

def create_sweep_config(event, lbcl, project_name="llmcot_sweeps"):
    """
    Creates a sweep configuration file for a specific event and lbcl.
    """
    config = {
        "program": "sweep_runner.py",
        "method": "bayes",
        "metric": {
            "name": "f1",
            "goal": "maximize"
        },
        "parameters": {
            "learning_rate": {"values": [1e-5, 2e-5, 3e-5, 5e-5]},
            "epoch_patience": {"values": [3, 5, 8]},
            "num_epochs": {"values": [10, 15, 20]},
            "seed": {"values": [42, 1234, 100]},
             # plm_id is hardcoded/default in runner, but can be added here if needed
        },
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--event", event,
            "--lbcl", str(lbcl),
            # W&B will append the parameters here
            "${args}"
        ]
    }
    
    filename = f"sweep_config_{event}_{lbcl}.yaml"
    with open(filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return filename

def main():
    parser = argparse.ArgumentParser(description="Create a W&B sweep for a specific model (event + lbcl).")
    parser.add_argument("--event", type=str, required=True, help="Event name")
    parser.add_argument("--lbcl", type=str, required=True, help="Labeled count per class")
    parser.add_argument("--project", type=str, default="llmcot_sweeps", help="W&B Project Name")
    
    args = parser.parse_args()
    
    # Create the config file
    config_file = create_sweep_config(args.event, args.lbcl, args.project)
    print(f"Created config file: {config_file}")
    
    # Initialize the sweep
    cmd = ["wandb", "sweep", config_file, "--project", args.project]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        
        # Extract sweep ID from output if possible, but the user can just see the output
        print("\n" + "="*50)
        print(f"Sweep initialized for Event: {args.event}, LBCL: {args.lbcl}")
        print("To start the agent, run the command shown in the output above, usually:")
        print("wandb agent <SWEEP_ID>")
        print("="*50 + "\n")
        
    except subprocess.CalledProcessError as e:
        print("Error creating sweep:")
        print(e.stderr)

if __name__ == "__main__":
    main()
