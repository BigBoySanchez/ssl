# bert_ft_refactored.py
# Example:
#   python bert_ft_refactored.py \
#     --dataset_path data/crisismmd2inf \
#     --output_dir outputs/bert_supervised_min \
#     --text_col tweet_text --label_col label --id_col tweet_id

import argparse, os, json, itertools, time, random
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ----------------------------- Metrics -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ----------------------------- Utils -----------------------------
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
import os

def load_raw_dataset(dataset_path: str,
                     raw_format: str = "auto",
                     tsv_train: str = "train.tsv",
                     tsv_dev:   str = "dev.tsv",
                     tsv_test:  str = "test.tsv",
                     tsv_delim: str = "\t") -> DatasetDict:
    """
    Returns a DatasetDict with keys 'train' (if present), 'dev', 'test'.
    Supports:
      - HF dir saved via save_to_disk (raw_format='hf')
      - Folder containing TSV files (raw_format='tsvdir')
      - 'auto' tries HF first, then TSV folder.
    """
    def _is_dir_with_tsvs(p):
        return os.path.isdir(p) and \
               any(os.path.exists(os.path.join(p, name)) for name in [tsv_train, tsv_dev, tsv_test])

    def _load_hf(p):
        d = load_from_disk(p)
        if isinstance(d, Dataset):  # single split
            return DatasetDict({"train": d})
        return d

    def _load_tsvdir(p):
        files = {}
        fp_train = os.path.join(p, tsv_train)
        fp_dev   = os.path.join(p, tsv_dev)
        fp_test  = os.path.join(p, tsv_test)
        if os.path.exists(fp_train): files["train"] = fp_train
        if os.path.exists(fp_dev):   files["dev"]   = fp_dev
        if os.path.exists(fp_test):  files["test"]  = fp_test
        assert "dev" in files and "test" in files, "TSV folder must contain at least dev/test."
        ds = load_dataset("csv", data_files=files, delimiter=tsv_delim)
        return DatasetDict({k: v for k, v in ds.items()})  # ensure DatasetDict

    if raw_format == "hf":
        return _load_hf(dataset_path)
    if raw_format == "tsvdir":
        return _load_tsvdir(dataset_path)

    # auto mode
    try:
        return _load_hf(dataset_path)
    except Exception:
        if _is_dir_with_tsvs(dataset_path):
            return _load_tsvdir(dataset_path)
        raise ValueError(f"Could not load dataset from '{dataset_path}'. "
                         f"Set --raw_format to 'hf' or 'tsvdir' and check paths.")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_labels(ds: Dataset, label_col: str) -> List[str]:
    # preserve stable order by sorting stringified labels
    labs = sorted(set(ds[label_col]))
    # turn into str to be robust (e.g., labels 0/1 or "not_informative"/"informative")
    return [str(x) for x in labs]


def build_label_maps(label_order: Optional[List[str]], train_like: Dataset, label_col: str):
    if label_order is None:
        label_order = infer_labels(train_like, label_col)
    label2id = {l: i for i, l in enumerate(label_order)}
    id2label = {i: l for l, i in label2id.items()}
    return label_order, label2id, id2label


def tokenize_with_labels(ds: Dataset, tok, text_col: str, label_col: str, label2id: Dict[str, int]) -> Dataset:
    def add_label(ex):  # robust if label is not string
        return {"labels": label2id[str(ex[label_col])]}
    ds = ds.map(add_label)
    ds = ds.map(lambda b: tok(b[text_col], truncation=True), batched=True)
    # Some models (e.g., RoBERTa) don't use token_type_ids; drop if missing
    keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in ds.column_names:
        keep.add("token_type_ids")
    drop = [c for c in ds.column_names if c not in keep]
    return ds.remove_columns(drop)


def load_train_dataset(train_path: Optional[str], train_hf: Optional[str], split: str) -> Dataset:
    """
    Priority:
      1) --train_path (file/dir): auto-detect by extension
      2) --train_hf (load_from_disk dir)
    """
    if train_path:
        # try to infer by extension; supports csv/json/jsonl
        ext = os.path.splitext(train_path)[-1].lower()
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            return load_dataset("csv", data_files=train_path, split="train", delimiter=sep)
        elif ext in (".json", ".jsonl"):
            return load_dataset("json", data_files=train_path, split="train")
        else:
            # If it's a directory with HF arrow, this will raise; caller can fall back
            try:
                dsd = load_from_disk(train_path)
                return dsd[split] if isinstance(dsd, DatasetDict) else dsd
            except Exception as e:
                raise ValueError(f"Could not load --train_path='{train_path}': {e}")
    if train_hf:
        dsd = load_from_disk(train_hf)
        return dsd[split] if isinstance(dsd, DatasetDict) else dsd
    raise ValueError("You must provide either --train_path or --train_hf (or rely on dataset_path train split).")


def select_columns_safe(ds: Dataset, cols: List[str]) -> Dataset:
    keep = [c for c in cols if c in ds.column_names]
    return ds.select_columns(keep)


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True, help="HF dataset saved with save_to_disk (expects dev & test).")
    ap.add_argument("--output_dir", required=True)
    # Train source options:
    ap.add_argument("--train_path", help="Optional local file/dir for train (csv/tsv/json/jsonl or HF dir).")
    ap.add_argument("--train_hf", help="Optional HF dataset dir (load_from_disk) to use as train.")
    ap.add_argument("--train_split_name", default="train")
    ap.add_argument("--dev_split_name", default="dev")
    ap.add_argument("--test_split_name", default="test")
    # Additional raw data options (if using dataset_path as HF dir or TSV dir)
    ap.add_argument("--raw_format", choices=["auto","hf","tsvdir"], default="auto")
    ap.add_argument("--tsv_train", default="train.tsv")
    ap.add_argument("--tsv_dev",   default="dev.tsv")
    ap.add_argument("--tsv_test",  default="test.tsv")
    ap.add_argument("--tsv_delim", default="\t")


    # Columns
    ap.add_argument("--text_col", default="tweet_text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--id_col", default="tweet_id")

    # Labels
    ap.add_argument("--label_order", nargs="*", help="Optional explicit label order, e.g. --label_order not_informative informative")

    # Model / training
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--lrs", nargs="*", type=float, default=[2e-5, 3e-5])
    ap.add_argument("--epochs", nargs="*", type=int, default=[3, 5])
    ap.add_argument("--batch_sizes", nargs="*", type=int, default=[16, 32])
    ap.add_argument("--selection_metric", choices=["f1", "accuracy"], default="f1")
    ap.add_argument("--seed", type=int, default=int(time.time()))
    ap.add_argument("--allow_cpu", action="store_true", help="Allow running without GPU (slower).")
    ap.add_argument("--max_train_samples", type=int, help="Optional cap on training samples for quick trials.")

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # ---------------- GPU check ----------------
    if not torch.cuda.is_available() and not args.allow_cpu:
        print("No GPU detected. Use --allow_cpu to run on CPU.")
        return

    # ---------------- Load base dataset (dev/test required) ----------------
    dsd_raw = load_raw_dataset(
        args.dataset_path,
        raw_format=args.raw_format,
        tsv_train=args.tsv_train, tsv_dev=args.tsv_dev, tsv_test=args.tsv_test,
        tsv_delim=args.tsv_delim,
    )
    assert "dev" in dsd_raw and "test" in dsd_raw, "need dev/test splits"

    dev_raw = dsd_raw[args.dev_split_name]
    test_raw = dsd_raw[args.test_split_name]

    # Train priority: explicit path → explicit HF dir → dataset_path's train split
    if args.train_path or args.train_hf:
        train_raw = load_train_dataset(args.train_path, args.train_hf, args.train_split_name)
    else:
        assert args.train_split_name in dsd_raw, "No train split and no train source provided."
        train_raw = dsd_raw[args.train_split_name]

    print("[debug] columns:", train_raw.column_names)

    # Optional: cap train for quick experiments
    if args.max_train_samples and len(train_raw) > args.max_train_samples:
        train_raw = train_raw.shuffle(seed=args.seed).select(range(args.max_train_samples))

    # ---------------- Build labels/maps ----------------
    label_order, label2id, id2label = build_label_maps(args.label_order, dev_raw, args.label_col)
    with open(os.path.join(args.output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label_order": label_order, "label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
                  f, ensure_ascii=False, indent=2)

    # ---------------- Tokenizer & tokenization ----------------
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = tokenize_with_labels(train_raw, tok, args.text_col, args.label_col, label2id)
    dev_ds   = tokenize_with_labels(dev_raw,   tok, args.text_col, args.label_col, label2id)
    test_ds  = tokenize_with_labels(test_raw,  tok, args.text_col, args.label_col, label2id)
    collator = DataCollatorWithPadding(tokenizer=tok)

    # ---------------- Keep original ids/golds for output ----------------
    test_ids   = test_raw[args.id_col] if args.id_col in test_raw.column_names else list(range(len(test_raw)))
    test_gold_names = [str(x) for x in test_raw[args.label_col]]
    test_gold_ids   = [label2id[name] for name in test_gold_names]

    # ---------------- Grid search on dev ----------------
    best_cfg, best_score = None, -1.0
    summary_rows = []
    for lr, ep, bs in itertools.product(args.lrs, args.epochs, args.batch_sizes):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
        )
        targs = TrainingArguments(
            output_dir=f"{args.output_dir}/trial_lr{lr}_ep{ep}_bs{bs}",
            eval_strategy="epoch", save_strategy="no",
            learning_rate=lr, num_train_epochs=ep,
            per_device_train_batch_size=bs, per_device_eval_batch_size=max(8, bs),
            report_to=[], seed=args.seed, load_best_model_at_end=False
        )
        trainer = Trainer(
            model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
            tokenizer=tok, data_collator=collator, compute_metrics=compute_metrics
        )
        trainer.train()
        eval_out = trainer.evaluate()
        metric_key = f"eval_{args.selection_metric}"
        score = float(eval_out.get(metric_key, float("nan")))
        print(f"[dev] lr={lr} ep={ep} bs={bs} -> {args.selection_metric}={score:.4f}")
        summary_rows.append({"lr": lr, "epochs": ep, "batch_size": bs, **eval_out})
        if score > best_score:
            best_score, best_cfg = score, (lr, ep, bs)

        # Save each trained trial (so you can inspect later)
        trainer.save_model(f"{args.output_dir}/trial_lr{lr}_ep{ep}_bs{bs}")

    # Write grid summary
    with open(os.path.join(args.output_dir, "grid_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    cfg_str = f"[best] cfg={best_cfg} {args.selection_metric}={best_score:.4f} seed={args.seed}"
    print(cfg_str)
    with open(os.path.join(args.output_dir, "best_cfg.txt"), "w", encoding="utf-8") as f:
        f.write(cfg_str + "\n")

    # ---------------- Retrain best on train, predict test ----------------
    lr, ep, bs = best_cfg
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    targs = TrainingArguments(
        output_dir=f"{args.output_dir}/best", eval_strategy="no", save_strategy="no",
        learning_rate=lr, num_train_epochs=ep,
        per_device_train_batch_size=bs, per_device_eval_batch_size=max(8, bs),
        report_to=[], seed=args.seed
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, tokenizer=tok, data_collator=collator)
    trainer.train()
    trainer.save_model(f"{args.output_dir}/best")

    test_pred_logits = trainer.predict(test_ds).predictions
    test_pred_ids = np.argmax(test_pred_logits, axis=-1)
    test_pred_names = [str(id2label[int(p)]) for p in test_pred_ids]

    # Write ints
    csv_int = os.path.join(args.output_dir, "pred_bert_int.csv")
    with open(csv_int, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["id", "gold", "pred"])
        for _id, g, p in zip(test_ids, test_gold_ids, test_pred_ids):
            w.writerow([_id, int(g), int(p)])

    # Write names (handy for quick inspection)
    csv_str = os.path.join(args.output_dir, "pred_bert_str.csv")
    with open(csv_str, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["id", "gold_name", "pred_name"])
        for _id, gname, pname in zip(test_ids, test_gold_names, test_pred_names):
            w.writerow([_id, gname, pname])

    print(f"Wrote: {csv_int}")
    print(f"Wrote: {csv_str}")


if __name__ == "__main__":
    main()
