# Hyperparameter Tuning
Done using wandb sweep

### Sweep Config
1 sweeper per model

sweep config:
- lr = ?
- dataset = humaid
- hf_model_id_short = N/A 
- plm_id = roberta-base 
- metric_combination = cv
- setup_local_logging
- seed = 1234
- pseudo_label_dir = anh_4o
- event = ?
- lbcl = ?
- set_num = ?
- data_dir = ../../data
- cuda_devices = 0,1
- num_epochs = ?
- epoch_patience = ?
- preds_file = preds.json

#### 1. Initialization (Weight Generation) & 2. Co-training
- **Epochs**: num_epochs
- **Patience**: epoch_patience
- **Learning Rate**: lr
- **Optimizer**: ?
- **Scheduler**: ?
- **Accumulation Steps**: ?

#### 3. Fine-tuning
- **Epochs**: num_epochs
- **Patience**: epoch_patience


I need to do Hyper Parameter Tuning for now. We need to use Weights and Biases to track everything. Specifically:

1. Make the existing cotrain code compatible with the hyperparameters in ... , (Note: '?' means we're tuning this arg)
2. Make a script that takes an event and lbcl and outputs a sweep.yaml, passing those in along with set_num and the hyperparams listed in..  
3. Average results over all 3 sets (logging macro-F1, accuracy, and ECE) while logging individual results for each set (note: you are allowed to make a wrapper to do this)

Generate a detailed plan, but make it understandable for a CS undergrad.

# Plan: Hyperparameter Tuning for Co-Training with Wandb
Integrate Weights and Biases (wandb) into the existing co-training code to enable hyperparameter sweeps, create a script for generating sweep configurations, and add logging for averaged results across multiple runs. This will allow systematic tuning of learning rate, epochs, and patience while tracking metrics like macro-F1, accuracy, and ECE.

## Steps
- Update main_bertweet.py to add wandb integration, make hyperparameters (lr, num_epochs, epoch_patience) configurable via command-line args, and pass them to trainer classes.
- Modify trainer classes in trainer_classes.py to use the new hyperparameter args instead of hardcoded values, and add wandb logging for metrics during training and evaluation.
- Create a new script generate_sweep.py in cotrain that takes event, lbcl, and set_num as inputs, defines hyperparameter ranges from the prompt, and outputs a sweep.yaml file.
- Add result averaging by creating a wrapper script run_sweep_wrapper.py that runs the main script three times (one per set), logs individual results to wandb, and computes/ logs averages for macro-F1, accuracy, and ECE.
- Further Considerations
- Decide on hyperparameter ranges for lr (e.g., 1e-5 to 1e-3), num_epochs (e.g., 5-20), and epoch_patience (e.g., 3-10) based on typical BERTweet tuning.
- Ensure ECE calculation is added if not present, possibly using a library like calibration or custom code in utils.py.
- Test the sweep locally first without wandb to verify compatibility, then switch to wandb for full tracking.