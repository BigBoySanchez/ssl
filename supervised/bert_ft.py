# bert_ft.py
# Example:
#   python bert_ft.py ^
#     --dataset_path data\crisismmd2inf ^
#     --raw_format tsvdir ^
#     --output_dir outputs\bert_supervised_min ^
#     --text_col tweet_text --label_col label --id_col tweet_id

import argparse, os, json, itertools, time, random, sys
from typing import List, Dict, Optional

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../utils'))
try:
    from run_humaid import get_paths
except ImportError:
    print("Warning: Could not import get_paths from run_humaid. Make sure ../utils exists.")
    get_paths = None

import wandb


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
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ----------------------------- Utils -----------------------------
def load_raw_dataset(dataset_path: str,
                     raw_format: str = "auto",
                     tsv_train: str = "train.tsv",
                     tsv_dev:   str = "dev.tsv",
                     tsv_test:  str = "test.tsv",
                     tsv_delim: str = "\t") -> DatasetDict:
    """Load dataset from HF dir or TSV folder."""
    def _is_dir_with_tsvs(p):
        return os.path.isdir(p) and any(
            os.path.exists(os.path.join(p, name)) for name in [tsv_train, tsv_dev, tsv_test]
        )

    def _load_hf(p):
        d = load_from_disk(p)
        if isinstance(d, Dataset):
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
        return DatasetDict({k: v for k, v in ds.items()})

    if raw_format == "hf":
        return _load_hf(dataset_path)
    if raw_format == "tsvdir":
        return _load_tsvdir(dataset_path)

    try:
        return _load_hf(dataset_path)
    except Exception:
        if _is_dir_with_tsvs(dataset_path):
            return _load_tsvdir(dataset_path)
        raise ValueError(f"Could not load dataset from '{dataset_path}'.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_labels(ds: Dataset, label_col: str) -> List[str]:
    labs = sorted(set(ds[label_col]))
    return [str(x) for x in labs]


def build_label_maps(label_order: Optional[List[str]], train_like: Dataset, label_col: str):
    if label_order is None:
        label_order = infer_labels(train_like, label_col)
    label2id = {l: i for i, l in enumerate(label_order)}
    id2label = {i: l for l, i in label2id.items()}
    return label_order, label2id, id2label


# ---------- CHANGE #2 ONLY: enforce safe max_length truncation ----------
def tokenize_with_labels(ds: Dataset, tok, text_col: str, label_col: str,
                         label2id: Dict[str, int], max_length: int) -> Dataset:
    def add_label(ex):
        return {"labels": label2id[str(ex[label_col])]}
    ds = ds.map(add_label)
    ds = ds.map(lambda b: tok(b[text_col], truncation=True, max_length=max_length), batched=True)

    # Always keep input_ids, attention_mask, labels, and token_type_ids (if any)
    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    drop = [c for c in ds.column_names if c not in keep]
    return ds.remove_columns(drop)


def load_train_dataset(train_path: Optional[str], train_hf: Optional[str], split: str) -> Dataset:
    if train_path:
        ext = os.path.splitext(train_path)[-1].lower()
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            return load_dataset("csv", data_files=train_path, split="train", delimiter=sep)
        elif ext in (".json", ".jsonl"):
            return load_dataset("json", data_files=train_path, split="train")
        else:
            try:
                dsd = load_from_disk(train_path)
                return dsd[split] if isinstance(dsd, DatasetDict) else dsd
            except Exception as e:
                raise ValueError(f"Could not load --train_path='{train_path}': {e}")
    if train_hf:
        dsd = load_from_disk(train_hf)
        return dsd[split] if isinstance(dsd, DatasetDict) else dsd
    raise ValueError("Provide --train_path or --train_hf (or rely on dataset_path train split).")


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_path")
    ap.add_argument("--train_hf")
    ap.add_argument("--train_split_name", default="train")
    ap.add_argument("--dev_split_name", default="dev")
    ap.add_argument("--test_split_name", default="test")
    ap.add_argument("--raw_format", choices=["auto","hf","tsvdir"], default="auto")
    ap.add_argument("--tsv_train", default="train.tsv")
    ap.add_argument("--tsv_dev", default="dev.tsv")
    ap.add_argument("--tsv_test", default="test.tsv")
    ap.add_argument("--tsv_delim", default="\t")

    ap.add_argument("--text_col", default="tweet_text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--id_col", default="tweet_id")

    ap.add_argument("--label_order", nargs="*")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--lrs", nargs="*", type=float, default=[2e-5, 3e-5])
    ap.add_argument("--epochs", nargs="*", type=int, default=[3, 5])
    ap.add_argument("--batch_sizes", nargs="*", type=int, default=[16, 32])
    ap.add_argument("--selection_metric", choices=["f1", "accuracy"], default="f1")
    ap.add_argument("--seed", type=int, default=int(time.time()))
    ap.add_argument("--allow_cpu", action="store_true")
    ap.add_argument("--max_train_samples", type=int)

    # CHANGE #2: allow overriding max_length
    ap.add_argument("--max_length", type=int, default=0,
                    help="If >0, cap sequence length to this; else auto = min(512, model_max-2).")

    # HPO / HumAID Auto args
    ap.add_argument("--event", type=str, help="HumAID event name for auto-path resolution")
    ap.add_argument("--lbcl", type=int, help="Labels per class for auto-path resolution")
    ap.add_argument("--set_num", type=int, help="Set number for auto-path resolution")
    ap.add_argument("--project_name", type=str, default="humaid_supervised_hpo", help="WandB project name")

    args = ap.parse_args()
    
    # Initialize WandB
    if args.event and args.lbcl and args.set_num:
        # Auto-mode: use specific run name config
        wandb.init(project=args.project_name, config=args, reinit=True)
        # Allow wandb sweep to override these
        args.learning_rate = wandb.config.get("learning_rate", args.lrs[0])
        args.epochs = wandb.config.get("epochs", args.epochs[0])
        args.batch_size = wandb.config.get("batch_size", args.batch_sizes[0])
        args.model_name = wandb.config.get("model_name", args.model_name)
        args.max_length = wandb.config.get("max_length", args.max_length)
        
        # Override lrs/epochs/batch_sizes lists to single values for compatibility with downstream loop code
        args.lrs = [args.learning_rate]
        args.epochs = [args.epochs]
        args.batch_sizes = [args.batch_size]

        print(f"✅ Auto-Mode: Event={args.event}, LBCL={args.lbcl}, Set={args.set_num}")
        print(f"   Hyperparams: LR={args.learning_rate}, BS={args.batch_size}, Eps={args.epochs}")

        # Resolve paths
        if get_paths:
            paths = get_paths(args.event, args.lbcl, args.set_num)
            # Use separate_event logic from runs? 
            # get_paths returns: dev_path, test_path, joined_path, train_labeled_path, etc.
            # We map these to args
            args.dataset_path = paths["joined_path"]
            args.train_path = paths["train_labeled_path"]
            args.output_dir = paths["vmatch_out"].replace("vmatch", "sup_bert") + "_ft" # Separate output dir
            
            # Auto-set label columns if needed, though HumAID usually standard
            # args.label_col is default "label" but run_humaid logic passes "class_label"?
            # wait, run_humaid.py passes --label_col class_label
            # We should probably force it or rely on user passing it in make_container args?
            # Ideally we hardcode it for HumAID auto mode
            args.label_col = "class_label"
            
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # GPU check
    if not torch.cuda.is_available() and not args.allow_cpu:
        print("No GPU detected. Use --allow_cpu to run on CPU.")
        return

    # Load data
    dsd_raw = load_raw_dataset(
        args.dataset_path,
        raw_format=args.raw_format,
        tsv_train=args.tsv_train, tsv_dev=args.tsv_dev, tsv_test=args.tsv_test,
        tsv_delim=args.tsv_delim,
    )
    assert "dev" in dsd_raw and "test" in dsd_raw, "need dev/test splits"
    dev_raw = dsd_raw[args.dev_split_name]
    test_raw = dsd_raw[args.test_split_name]

    if args.train_path or args.train_hf:
        train_raw = load_train_dataset(args.train_path, args.train_hf, args.train_split_name)
    else:
        assert args.train_split_name in dsd_raw
        train_raw = dsd_raw[args.train_split_name]

    if args.max_train_samples and len(train_raw) > args.max_train_samples:
        train_raw = train_raw.shuffle(seed=args.seed).select(range(args.max_train_samples))

    label_order, label2id, id2label = build_label_maps(args.label_order, dev_raw, args.label_col)
    with open(os.path.join(args.output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label_order": label_order, "label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
                  f, ensure_ascii=False, indent=2)

    # Tokenizer & tokenization
    tok = AutoTokenizer.from_pretrained(
        args.model_name, 
        use_fast=True,
        normalization=True,
    )
    _tmp_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    model_max = int(getattr(_tmp_model.config, "max_position_embeddings", 512))
    del _tmp_model
    safe_max_length = int(args.max_length) if args.max_length > 0 else min(512, model_max - 2)

    train_ds = tokenize_with_labels(train_raw, tok, args.text_col, args.label_col, label2id, safe_max_length)
    dev_ds   = tokenize_with_labels(dev_raw, tok, args.text_col, args.label_col, label2id, safe_max_length)
    test_ds  = tokenize_with_labels(test_raw, tok, args.text_col, args.label_col, label2id, safe_max_length)
    collator = DataCollatorWithPadding(tokenizer=tok)

    test_ids = test_raw[args.id_col] if args.id_col in test_raw.column_names else list(range(len(test_raw)))
    test_gold_names = [str(x) for x in test_raw[args.label_col]]
    test_gold_ids = [label2id[name] for name in test_gold_names]

    # Grid search
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
            report_to=[], seed=args.seed, load_best_model_at_end=False,
            lr_scheduler_type="constant",
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
        
        if wandb.run:
            wandb.log({
                "lr": lr, "epochs": ep, "batch_size": bs,
                **eval_out,
                "best_score_so_far": best_score
            })

        trainer.save_model(f"{args.output_dir}/trial_lr{lr}_ep{ep}_bs{bs}")

    with open(os.path.join(args.output_dir, "grid_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    cfg_str = f"[best] cfg={best_cfg} {args.selection_metric}={best_score:.4f} seed={args.seed}"
    print(cfg_str)
    with open(os.path.join(args.output_dir, "best_cfg.txt"), "w", encoding="utf-8") as f:
        f.write(cfg_str + "\n")

    # Retrain best & predict
    lr, ep, bs = best_cfg
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
    targs = TrainingArguments(
        output_dir=f"{args.output_dir}/best", eval_strategy="no", save_strategy="no",
        learning_rate=lr, num_train_epochs=ep,
        per_device_train_batch_size=bs, per_device_eval_batch_size=max(8, bs),
        report_to=[], seed=args.seed, lr_scheduler_type="constant",
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, 
        tokenizer=tok, data_collator=collator
    )
    trainer.train()
    trainer.save_model(f"{args.output_dir}/best")

    test_pred_logits = trainer.predict(test_ds).predictions
    test_pred_ids = np.argmax(test_pred_logits, axis=-1)
    test_pred_names = [str(id2label[int(p)]) for p in test_pred_ids]

    csv_int = os.path.join(args.output_dir, "pred_bert_int.csv")
    with open(csv_int, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["id", "gold", "pred"])
        for _id, g, p in zip(test_ids, test_gold_ids, test_pred_ids):
            w.writerow([_id, int(g), int(p)])

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
