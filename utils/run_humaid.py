import subprocess
import itertools
import sys

# Configurable params
LB_PER_CLASS = [5, 10, 25, 50]
SETS = [1, 2, 3]

# Paths to your scripts
TRAIN_SCRIPT = r"..\verifymatch\train.py"
BERT_FT_SCRIPT = r"..\supervised\bert_ft.py"

# You can define your custom args inline here later
TRAIN_ARGS_TEMPLATE = [
    "python", TRAIN_SCRIPT,
    "--task", "HumAID",
    "--model", "bert-base-uncased",
    "--batch_size", "16",
    "--learning_rate", "3e-5",
    "--epochs", "5",
    "--mixup",
    "--mixup_loss_weight", "1",
    "--ssl",
    "--sharpening",
    "--T", "0.5",
    "--pseudo_label_by_normalized", 
    "--dev_path", r"..\data\humaid\joined\dev.tsv",
    "--test_path", r"..\data\humaid\joined\test.tsv",
    "--do_train",
    "--do_evaluate",
]

BERT_FT_ARGS_TEMPLATE = [
    "python", BERT_FT_SCRIPT,
    "--dataset_path", r"..\data\humaid\joined",
    "--label_col", "class_label",
    "--raw_format", "tsvdir",
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

def main():
    for lbcl, set_num in itertools.product(LB_PER_CLASS, SETS):
        tag = f"lb{lbcl}_set{set_num}"
        print(f"\n=== Running combo: {tag} ===", flush=True)

        # --------- Run train.py ---------
        train_cmd = TRAIN_ARGS_TEMPLATE + [
            fr"--labeled_train_path", fr"..\data\humaid\plabel\train\sep\{lbcl}lb\{set_num}\labeled.tsv",
            fr"--unlabeled_train_path",  fr"..\data\humaid\plabel\train\sep\{lbcl}lb\{set_num}\unlabeled.tsv",
            fr"--ckpt_path", fr"..\artifacts\humaid\ckpt\humaid_10_10_{lbcl}_{set_num}.ckpt",
            fr"--output_path", fr"..\artifacts\humaid\out\humaid_10_10_{lbcl}_{set_num}.json",
        ]
        code = run_and_stream(train_cmd, f"train[{tag}]")
        if code != 0:
            print(f"[ERROR] train.py failed for {tag} with exit code {code}", file=sys.stderr)
            exit(code)

        # --------- Run bert_ft.py w/ bertweet ---------
        bert_cmd = BERT_FT_ARGS_TEMPLATE + [
            fr"--output_dir", fr"..\artifacts\humaid\bertweet\humaid_bertweet_ft_{lbcl}_{set_num}",
            fr"--train_path", fr"..\data\humaid\plabel\train\sep\{lbcl}lb\{set_num}\labeled.tsv",
            fr"--model_name", fr"vinai/bertweet-base"
        ]
        code = run_and_stream(bert_cmd, f"bertweet[{tag}]")
        if code != 0:
            print(f"[ERROR] bert_ft.py (bertweet) failed for {tag} with exit code {code}", file=sys.stderr)
            exit(code)
        
        # --------- Run bert_ft.py w/ bert-base-uncased ---------
        bert_cmd = BERT_FT_ARGS_TEMPLATE + [
            fr"--output_dir", fr"..\artifacts\humaid\bert\humaid_bert_ft_{lbcl}_{set_num}",
            fr"--train_path", fr"..\data\humaid\plabel\train\sep\{lbcl}lb\{set_num}\labeled.tsv"
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
