# optuna_sweep.py
import optuna, subprocess, sys, re, os, datetime, pathlib

LABELED   = r"..\data\crisismmd2inf\plabel\train\labeled"
UNLABELED = r"..\data\crisismmd2inf\plabel\train\unlabeled"
DEV_PATH  = r"..\data\crisismmd2inf\raw\dev"
TEST_PATH = r"..\data\crisismmd2inf\raw\test"

LOG_DIR = "logs_optuna"
os.makedirs(LOG_DIR, exist_ok=True)
ACC_RE = re.compile(r"eval acc\s*=\s*([0-9.]+)")

def run_cmd(args_list):
    return subprocess.run(args_list, capture_output=True, text=True)

def parse_acc(text: str):
    m = ACC_RE.findall(text)
    if m:
        return float(m[-1])
    m2 = re.findall(r"accuracy\s*=\s*([0-9.]+)", text)
    if m2:
        return float(m2[-1])
    return None

def objective(trial: optuna.Trial) -> float:
    # ---- search space ----
    lr   = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    T    = trial.suggest_categorical("T", [0.3, 0.5, 0.7])
    mixw = trial.suggest_categorical("mixup_loss_weight", [0.5, 0.7, 1.0])
    bs   = trial.suggest_categorical("batch_size", [16, 32])
    ubs  = trial.suggest_categorical("unlabeled_batch_size", [32, 64])
    wd   = trial.suggest_categorical("weight_decay", [0.0, 0.01])
    ep   = trial.suggest_int("epochs", 3, 5)
    ls   = trial.suggest_categorical("label_smoothing", [0.0, 0.1, 0.2])

    # unique output/ckpt per trial
    stamp = f"t{trial.number}"
    ckpt  = os.path.join("..", "ckpt", f"crisismmdinf_10_9_{stamp}.ckpt")
    out   = os.path.join("..", "output", f"crisismmdinf_10_9_{stamp}.json")
    pathlib.Path(os.path.dirname(ckpt)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "train.py",
        "--task", "CrisisMMDINF",
        "--model", "bert-base-uncased",
        "--max_seq_length", "128",
        "--batch_size", str(bs),
        "--unlabeled_batch_size", str(ubs),
        "--learning_rate", str(lr),
        "--weight_decay", str(wd),
        "--epochs", str(ep),
        "--mixup", "--ssl", "--sharpening", "--pseudo_label_by_normalized",
        "--mixup_loss_weight", str(mixw),
        "--T", str(T),
        "--label_smoothing", str(ls),
        "--labeled_train_path", LABELED,
        "--unlabeled_train_path", UNLABELED,
        "--dev_path", DEV_PATH,
        "--test_path", TEST_PATH,
        "--ckpt_path", ckpt,
        "--output_path", out,
        "--do_train", "--do_evaluate",
    ]

    p = run_cmd(cmd)
    # write logs for every trial
    base = os.path.join(LOG_DIR, f"trial_{trial.number}")
    with open(base + ".out", "w", encoding="utf-8") as f: f.write(p.stdout)
    with open(base + ".err", "w", encoding="utf-8") as f: f.write(p.stderr)

    if p.returncode != 0:
        # attach a short tail to the trial for quick triage
        tail_out = "\n".join(p.stdout.splitlines()[-30:])
        tail_err = "\n".join(p.stderr.splitlines()[-30:])
        trial.set_user_attr("stdout_tail", tail_out)
        trial.set_user_attr("stderr_tail", tail_err)
        # return a bad score; Optuna will mark as FAIL if we raise, but we prefer continue
        raise RuntimeError(f"rc={p.returncode} (logs: {base}.out / {base}.err)")

    acc = parse_acc(p.stdout)
    if acc is None:
        # no metric found; mark failure but keep going
        trial.set_user_attr("stdout_tail", "\n".join(p.stdout.splitlines()[-30:]))
        trial.set_user_attr("stderr_tail", "\n".join(p.stderr.splitlines()[-30:]))
        raise RuntimeError(f"Could not parse eval acc (logs: {base}.out / {base}.err)")

    trial.report(acc, step=ep)
    return acc

if __name__ == "__main__":
    # Allow RuntimeError trials to fail without killing the study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, show_progress_bar=True, catch=(RuntimeError,))
    print("\nBest trial:", study.best_value, study.best_params)
    print(f"Logs saved under: {LOG_DIR}")
