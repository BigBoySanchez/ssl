"""
Fetch a dataset from Hugging Face Hub and, for each split,
- take up to N rows into a "removed" set
- keep the rest in a "kept" set

Saves both as HF save_to_disk directories.

Example:
  python fetch_split_take.py \
    --repo_id xiaoxl/crisismmd2inf \
    --removed_out /data/crisismmd2inf_removed \
    --kept_out /data/crisismmd2inf_kept \
    --max_rows 15000 \
    --keep_cols tweet_text event_name label_text tweet_id
"""

import argparse
import os
from datasets import load_dataset, DatasetDict
from datetime import datetime

def fetch_and_split_take(
    repo_id: str,
    config: str | None,
    revision: str | None,
    removed_out: str,
    kept_out: str,
    max_rows: int = 15000,
    seed: int = 42,
    keep_cols: list[str] | None = None,
):
    print(f"Loading dataset from hub: {repo_id}"
          f"{' (config='+config+')' if config else ''}"
          f"{' @ '+revision if revision else ''}")
    ds = load_dataset(repo_id, name=config, revision=revision)

    if not isinstance(ds, DatasetDict):
        raise ValueError("Expected a DatasetDict with named splits (train/dev/test/etc.).")

    removed_splits = {}
    kept_splits = {}

    for split_name, split_ds in ds.items():
        n = (len(split_ds) // 100) * 2 # TODO may increase label:plabel ratio if accuracy drops
        take_n = min(max_rows, n)
        print(f"[{split_name}] taking {take_n} of {n}")

        # Shuffle for a representative sample
        shuffled = split_ds.shuffle(seed=seed)

        removed = shuffled.select(range(take_n))
        kept = shuffled.select(range(take_n, n)) if take_n < n else split_ds.select([])

        # Optionally reduce to a column subset
        if keep_cols:
            # Only keep columns that actually exist
            rk = [c for c in keep_cols if c in removed.column_names]
            kk = [c for c in keep_cols if c in kept.column_names]
            if rk:
                removed = removed.remove_columns([c for c in removed.column_names if c not in rk])
            if kk:
                kept = kept.remove_columns([c for c in kept.column_names if c not in kk])

            # Warn about missing columns (once per split)
            missing_r = [c for c in (keep_cols or []) if c not in removed.column_names]
            missing_k = [c for c in (keep_cols or []) if c not in kept.column_names]
            if missing_r or missing_k:
                print(f"  [warn] missing cols in '{split_name}': "
                      f"removed missing={missing_r}, kept missing={missing_k}")

        removed_splits[split_name] = removed
        kept_splits[split_name] = kept

    removed_dd = DatasetDict(removed_splits)
    kept_dd = DatasetDict(kept_splits)

    os.makedirs(removed_out, exist_ok=True)
    os.makedirs(kept_out, exist_ok=True)
    removed_dd.save_to_disk(removed_out)
    kept_dd.save_to_disk(kept_out)
    print(f"Saved removed-only dataset to: {removed_out}")
    print(f"Saved kept-only dataset to:    {kept_out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True, help="HF dataset repo id, e.g. xiaoxl/crisismmd2inf")
    p.add_argument("--config", default=None, help="Optional dataset config name")
    p.add_argument("--revision", default=None, help="Optional branch/tag/commit on the Hub")
    p.add_argument("--removed_out", required=True, help="Directory to save the removed-only DatasetDict")
    p.add_argument("--kept_out", required=True, help="Directory to save the kept-only DatasetDict")
    p.add_argument("--max_rows", type=int, default=15000, help="Max rows to remove per split")
    p.add_argument("--seed", type=int, default=int(datetime.now().timestamp()), help="Shuffle seed")
    p.add_argument(
        "--keep_cols",
        nargs="*",
        default=None,
        help="If set, keep only these columns in both outputs "
             "(e.g., tweet_text event_name label_text tweet_id)",
    )
    args = p.parse_args()

    fetch_and_split_take(
        repo_id=args.repo_id,
        config=args.config,
        revision=args.revision,
        removed_out=args.removed_out,
        kept_out=args.kept_out,
        max_rows=args.max_rows,
        seed=args.seed,
        keep_cols=args.keep_cols,
    )

if __name__ == "__main__":
    main()
