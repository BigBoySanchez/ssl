# bert_ft.py
# Usage:
#   python bert_ft.py \
#     --dataset_path data/crisismmd2inf/ \
#     --output_dir outputs/bert_supervised_min \
#     --text_col tweet_text --label_col label --id_col tweet_id

import argparse, os, csv, itertools, numpy as np, random, torch
from datasets import load_from_disk, load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_hf", help="Optional Hugging Face dataset name to use as the train set")

    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--text_col", default="tweet_text")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--id_col", default="tweet_id")

    # minimal grid; tweak via CLI if you like
    ap.add_argument("--lrs", nargs="*", type=float, default=[2e-5, 3e-5])
    ap.add_argument("--epochs", nargs="*", type=int, default=[3, 5])
    ap.add_argument("--batch_sizes", nargs="*", type=int, default=[16, 32])

    ap.add_argument("--seed", type=int, default=int(time.time()))
    ap.add_argument("--selection_metric", choices=["f1", "accuracy"], default="f1")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    dsd_raw = load_from_disk(args.dataset_path)
    assert all(s in dsd_raw for s in ["dev","test"]), "need dev/test splits on disk"

    # --- Choose TRAIN source: external HF dataset if provided, else dataset's train split ---
    if args.train_hf:
        train_raw = load_from_disk(args.train_hf)
    else:
        assert "train" in dsd_raw, "no train split and --train_hf not provided"
        train_raw = dsd_raw["train"]

    # Test stays from the on-disk dataset
    test_raw = dsd_raw["test"]
    # Keep test IDs & golds for later (before we strip columns)
    test_ids   = test_raw[args.id_col]
    test_golds = test_raw[args.label_col]

    # --- Build label maps, tokenize, and trim for training ---
    LABEL_ORDER = ["not_informative", "informative"]
    label2id = {l:i for i, l in enumerate(LABEL_ORDER)}
    id2label = {i:l for l, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Map labels/tokenize per split (train may be external)
    def add_label(ex): return {"labels": label2id[ex[args.label_col]]}
    train_tok = train_raw.map(add_label)
    train_tok = train_tok.map(lambda b: tok(b[args.text_col], truncation=True), batched=True)

    dev_raw  = dsd_raw["dev"]
    dev_tok  = dev_raw.map(add_label)
    dev_tok  = dev_tok.map(lambda b: tok(b[args.text_col], truncation=True), batched=True)

    test_tok = test_raw.map(add_label)
    test_tok = test_tok.map(lambda b: tok(b[args.text_col], truncation=True), batched=True)

    # drop everything not needed for the model (no ids during training)
    needed = {"input_ids","attention_mask","token_type_ids","labels"}
    def trim(ds): return ds.remove_columns([c for c in ds.column_names if c not in needed])
    train_ds, dev_ds = map(trim, [train_tok, dev_tok])
    test_ds = trim(test_tok)

    collator = DataCollatorWithPadding(tokenizer=tok)

    # NOTE: do NOT re-introduce the original `label` (string) column into the training
    # datasets — that column can be a string or nested list and will confuse the
    # DataCollator/tokenizer.pad. We already created integer `labels` above and trimmed
    # train/dev/test to the model-needed columns. If you want the original id/text/label
    # for export, pull them separately from `dsd_raw` or from `dsd["test"]` when
    # preparing outputs.

    # Grid search on dev
    best_cfg, best_score = None, -1.0
    for lr, ep, bs in itertools.product(args.lrs, args.epochs, args.batch_sizes):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
        )
        targs = TrainingArguments(
            output_dir=f"{args.output_dir}/trial_lr{lr}_ep{ep}_bs{bs}",
            eval_strategy="epoch", save_strategy="no",
            learning_rate=lr, num_train_epochs=ep,
            per_device_train_batch_size=bs, per_device_eval_batch_size=max(8, bs),
            report_to=[], seed=args.seed
        )
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                          tokenizer=tok, data_collator=collator, compute_metrics=compute_metrics)
        trainer.train()
        score = trainer.evaluate()[f'eval_{args.selection_metric}']
        print(f"[dev] lr={lr} ep={ep} bs={bs} -> {args.selection_metric}={score:.4f}")
        if score > best_score:
            best_score, best_cfg = score, (lr, ep, bs)

        trainer.save_model(f"{args.output_dir}/trial_lr{lr}_ep{ep}_bs{bs}")

    cfg_str = f"[best] cfg={best_cfg} {args.selection_metric}={best_score:.4f}"
    print(cfg_str)
    with open(os.path.join(args.output_dir, "best_cfg.txt"), "w", encoding="utf-8") as f:
        f.write(cfg_str + "\n")

    # Retrain best on train, then predict test
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
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      tokenizer=tok, data_collator=collator)
    trainer.train()
    trainer.save_model(f"{args.output_dir}/best")

    gold_ids = [label2id[tg] for tg in test_golds]
    pred_ids = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    # pred_labels = [id2label[int(p)] for p in pred_ids]

    with open(os.path.join(args.output_dir, "pred_bert.csv"), "w", newline="", encoding="utf-8") as f:
        import csv; w = csv.writer(f)
        w.writerow(["id", "gold", "pred"])
        for _id, g, p in zip(test_ids, gold_ids, pred_ids):
            w.writerow([_id, g, p])

if __name__ == "__main__":
    main()
