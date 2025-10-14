import subprocess
import itertools
import sys
import os
import pandas as pd

# Configurable params
LB_PER_CLASS = [5]
SETS = [1, 2, 3]
# LB_PER_CLASS = [5, 10, 25, 50]
# SETS = [1, 2, 3]

# Paths to your scripts
TRAIN_SCRIPT = r"..\verifymatch\train.py"
BERT_FT_SCRIPT = r"..\supervised\bert_ft.py"

# You can define your custom args inline here later
TRAIN_ARGS_TEMPLATE = [
    "python", TRAIN_SCRIPT,
    "--device", "0",
    "--model", "bert-base-uncased",
    "--task", "HumAID",
    "--do_train",
    "--do_evaluate",
    "--ssl",
    "--mixup",
    "--sharpening",
    "--epochs", "3",
    "--batch_size", "32",
    "--unlabeled_batch_size", "32",
    "--learning_rate", "2e-5",
    "--weight_decay", "0",
    "--max_grad_norm", "1.0",
    "--T", "0.5",
    "--pseudo_label_by_normalized"
]

BERT_FT_ARGS_TEMPLATE = [
    "python", BERT_FT_SCRIPT,
    "--label_col", "class_label",
    "--raw_format", "tsvdir",
    "--lrs", "5e-5",
    "--epochs", "3",
    "--batch_sizes", "16",
    "--max_length", "128",
    "--seed", "42",
]


def run_and_stream(cmd, prefix):
    """Run a command and stream its stdout to this process's stdout."""
    print(f"Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()
    finally:
        if proc.stdout:
            proc.stdout.close()
    return proc.wait()

def get_events(tsv_folder):
    files = [os.path.join(tsv_folder, f"{split}.tsv") for split in ["train", "dev", "test"]]
    df = pd.concat(
        (pd.read_csv(f, sep="\t", usecols=[0]) for f in files),
        ignore_index=True
    )
    return set(df.iloc[:, 0].unique())

def separate_event(event, tsv_file, outfile_name):
    """
    Gather all rows with the specified event label from the TSV file.
    Save them to a new TSV file in: ./temp/{event}_{outfile_name}.tsv
    Returns the output file path.
    """
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)

    # Build output path safely
    output_path = os.path.join("temp", f"{event}_{outfile_name}.tsv")

    # Check if the input file exists
    if not os.path.exists(tsv_file):
        raise FileNotFoundError(f"TSV file not found: {tsv_file}")

    # Read and filter
    df = pd.read_csv(tsv_file, sep="\t")
    if df.empty:
        print(f"Warning: {tsv_file} is empty.")
        return None

    # Assuming the event column is the first column
    event_df = df[df.iloc[:, 0] == event]

    # Only save if there are matching rows
    if event_df.empty:
        print(f"No rows found for event '{event}' in {tsv_file}.")
        return None

    # Save filtered rows
    event_df.to_csv(output_path, sep="\t", index=False)

    return output_path

def separate_event_folder(tsv_folder, event, outfile_name):
    """
    Like separate_event but for all files in a folder.
    Returns files with the desired event in:
        ./temp/{event}_{outfile_name}/{original_filename}.tsv
    """
    # Create output folder safely
    out_folder = os.path.join("temp", f"{event}_{outfile_name}")
    os.makedirs(out_folder, exist_ok=True)

    # Process each split file
    for split in ["train", "dev", "test"]:
        input_path = os.path.join(tsv_folder, f"{split}.tsv")

        # Skip missing files gracefully
        if not os.path.exists(input_path):
            continue

        # Read and filter
        df = pd.read_csv(input_path, sep="\t")
        if df.empty:
            continue

        # Assuming the event column is the first column
        event_df = df[df.iloc[:, 0] == event]

        # Write only if filtered data exists
        if not event_df.empty:
            output_path = os.path.join(out_folder, f"{split}.tsv")
            event_df.to_csv(output_path, sep="\t", index=False)

    return out_folder

def main():
    EVENTS = get_events(r"..\data\humaid\joined")
    # TODO: ts is for testing only, remove later
    EVENTS = ["california_wildfires_2018"]

    for event, lbcl, set_num in itertools.product(EVENTS, LB_PER_CLASS, SETS):
        tag = f"{event}_lb{lbcl}_set{set_num}"
        print(f"\n=== Running combo: {tag} ===", flush=True)

        dev_path = separate_event(event, r"..\data\humaid\joined\dev.tsv", "dev")
        test_path = separate_event(event, r"..\data\humaid\joined\test.tsv", "test")
        joined_path = separate_event_folder(r"..\data\humaid\joined", event, "joined")
        train_labeled_path = separate_event(event, fr"..\data\humaid\plabel\train\sep\{lbcl}lb\{set_num}\labeled.tsv", "labeled")
        train_unlabeled_path = separate_event(event, fr"..\data\humaid\plabel\train\sep\{lbcl}lb\{set_num}\unlabeled.tsv", "unlabeled")

        # --------- Run train.py ---------
        train_cmd = TRAIN_ARGS_TEMPLATE + [
            fr"--labeled_train_path", train_labeled_path,
            fr"--unlabeled_train_path", train_unlabeled_path,
            fr"--ckpt_path", fr"..\artifacts\humaid\ckpt3\humaid_10_10_{event}_{lbcl}_{set_num}.ckpt",
            fr"--output_path", fr"..\artifacts\humaid\out3\humaid_10_10_{event}_{lbcl}_{set_num}.json",
            fr"--dev_path", dev_path,
            fr"--test_path", test_path,
        ]
        code = run_and_stream(train_cmd, f"train[{tag}]")
        if code != 0:
            print(f"[ERROR] train.py failed for {tag} with exit code {code}", file=sys.stderr)
            exit(code)

        # --------- Run bert_ft.py w/ bertweet ---------
        bert_cmd = BERT_FT_ARGS_TEMPLATE + [
            fr"--output_dir", fr"..\artifacts\humaid\bertweet3\humaid_bertweet_ft_{event}_{lbcl}_{set_num}",
            fr"--train_path", train_labeled_path,
            fr"--dataset_path", joined_path,
            fr"--model_name", fr"vinai/bertweet-base",
        ]
        code = run_and_stream(bert_cmd, f"bertweet[{tag}]")
        if code != 0:
            print(f"[ERROR] bert_ft.py (bertweet) failed for {tag} with exit code {code}", file=sys.stderr)
            exit(code)
        
        # --------- Run bert_ft.py w/ bert-base-uncased ---------
        bert_cmd = BERT_FT_ARGS_TEMPLATE + [
            fr"--output_dir", fr"..\artifacts\humaid\bert3\humaid_bert_ft_{event}_{lbcl}_{set_num}",
            fr"--train_path", train_labeled_path,
            fr"--dataset_path", joined_path,
        ]
        code = run_and_stream(bert_cmd, f"bert[{tag}]")
        if code != 0:
            print(f"[ERROR] bert_ft.py failed for {tag} with exit code {code}", file=sys.stderr)
            exit(code)

    print("\nAll combinations complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
