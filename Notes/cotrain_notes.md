## GENERAL NOTES
- We shouldn't aim for full fine-tuning with wandb yet
  - We should get a few working models, then score them (macro-F1)
- READABILITY AND SIMPLICITY ARE KING FOR THIS PROJECT

## main_bertweet.py
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