$dataDir = "data"
$pseudoDir = "data/pseudo_labels"
$humaidDir = "$dataDir/humaid"
$humaidPsDir = "$pseudoDir/humaid"

New-Item -ItemType Directory -Force -Path "$humaidDir/dev"
New-Item -ItemType Directory -Force -Path "$humaidDir/test"
New-Item -ItemType Directory -Force -Path "$humaidPsDir"

# Create dummy JSON with required fields
# Label 0: rescue_volunteering_or_donation_effort (based on my map)
# Using 'ori_label' and 'gen_label' as integer 0 because I noticed the loading might map it.
# Wait, loading maps 'ori_label' string to int if map is provided.
# But TextOnlyProcessor expects 'label' (int) or 'ori_label' (string?).
# data_utils lines 179: trainingSet_1['label'] = trainingSet_1['ori_label']
# So it expects ori_label to be the class label.
# TextOnlyProcessor handles conversion if labeled.
# But my data_utils load_dataset_helper maps labels IF loaded from dev/test.
# For training sets, it just passes them through.
# Let's use strings matching the key in label_map to be safe.

$dummyJson = '{
    "1": {
        "tweet_id": "1", 
        "tweet_text": "Help needed", 
        "label": "affected_individuals", 
        "ori_label": "affected_individuals", 
        "gen_label": "affected_individuals", 
        "aug_0": "Help", 
        "aug_1": "Help needed now"
    },
    "2": {
        "tweet_id": "2", 
        "tweet_text": "Rescue team arrived", 
        "label": "rescue_volunteering_or_donation_effort", 
        "ori_label": "rescue_volunteering_or_donation_effort", 
        "gen_label": "rescue_volunteering_or_donation_effort", 
        "aug_0": "Rescue", 
        "aug_1": "Rescue team is here"
    }
}'

$dummyJson | Out-File "$humaidDir/dev/text_only.json" -Encoding ascii
$dummyJson | Out-File "$humaidDir/test/text_only.json" -Encoding ascii
$dummyJson | Out-File "$humaidPsDir/train_set1_0shot.json" -Encoding ascii
$dummyJson | Out-File "$humaidPsDir/train_set2_0shot.json" -Encoding ascii
$dummyJson | Out-File "$humaidPsDir/unlabeled_train_0shot.json" -Encoding ascii

Write-Host "Starting dry run..."
python main_bertweet.py --dataset humaid --labeled_sample_idx 0 --pseudo_label_shot 0 --no_co_training --data_dir data --pseudo_label_dir data/pseudo_labels --setup_local_logging --use_correct_labels_only True

if ($LASTEXITCODE -eq 0) {
    Write-Host "Dry run SUCCESS!"
} else {
    Write-Host "Dry run FAILED with code $LASTEXITCODE"
}
