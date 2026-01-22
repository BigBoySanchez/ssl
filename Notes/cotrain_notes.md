## GENERAL NOTES
- We shouldn't aim for full fine-tuning with wandb yet
  - We should get a few working models, then score them (macro-F1)
- READABILITY AND SIMPLICITY ARE KING FOR THIS PROJECT

## main_bertweet.py
- **3-Stage Training Pipeline**:
    1.  **Initialization** (lines 450-463):
        *   Trains 2 models independently on labeled data.
        *   Creates baseline for pseudo-labeling.
    2.  **Co-Training** (lines 474-488):
        *   Models "teach each other" (Model A labels for B, vice versa).
        *   Produces `co_training_df` (soft labels) and best checkpoints.
        *   *Note:* Intermediate models are deleted to save space.
    3.  **Final Fine-tuning** (lines 514-535):
        *   **NOT** hyperparameter tuning.
        *   Re-initializes fresh models and loads best co-trained weights.
        *   Trains for **100 epochs** (vs 10 in earlier stages) to stabilize and converge on high-quality pseudo-labels.

- Main training file
- Runs experiments using comet
  - I could learn it, but sticking to wandb may be simpler
> def load_imb_dataset_helper(dataset, N, pseudo_label_shot, processed_dir, data_dir, use_correct_labels_only=None, mnli_split=None):
>    """Helper function to load datasets with long-tailed label distribution."""
- What is this ^
> If not using multiset, make both training sets the same
- ^ Doesn't make sense b/c cotraining is inherently "multi-set"
- Does eval too



## trainer_classes.py
### WeightGenerator
- Initializes weights for the 2 classifiers
### CoTrainer
- Houses the actual training code
### DualModelTrainer
- Fine tunes the models
  - How do u fine tune an existing model?

## TEMP
run this: `python main_bertweet.py --dataset humaid --hf_model_id_short N/A --plm_id roberta-base --metric_combination cv --setup_local_logging --seed 1234 --pseudo_label_dir anh_4o --event california_wildfires_2018 --lbcl 50 --set_num 1 --data_dir ../../data --cuda_devices=0,1 2>&1`

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
- **Optimizer**: AdamW (weight_decay=0.01)
- **Scheduler**: Linear (0 warmup)
- **Accumulation Steps**: 64 / BATCH_SIZE

#### 3. Fine-tuning
- **Epochs**: num_epochs
- **Patience**: epoch_patience


I need to do Hyper Parameter Tuning for @llm-co-training-crisismmd-main/cotrain now. We need to use Weights and Biases to track everything. Specifically:

1. Make a script that takes an event and lbcl and outputs a sweep.yaml, passing those in along with 