"""
Given a bertweet output folder, find F1 for each run and print a table with:
- rows = event
- columns = {lb/cl}_{set_num}

Folder naming pattern assumption:
  <prefix1>_<prefix2>_<prefix3>_<event_part1>_..._<event_partN>_<lbcl>_<setnum>
  (i.e. event = tokens[3:-2])

Each run folder contains best_cfg.txt with a line like: f1=0.8421
"""

import os
import re
import sys
from collections import defaultdict

def find_best_cfg_scores(base_dir):
    f1_pat = re.compile(r"f1\s*=\s*([0-9]*\.?[0-9]+)")
    table = defaultdict(dict)
    all_cols = set()

    for root, _, files in os.walk(base_dir):
        if "best_cfg.txt" not in files:
            continue

        subfolder = os.path.basename(root)
        parts = subfolder.split("_")

        # event = tokens[3:-2] (joined by '_')
        event_tokens = parts[3:-2]
        event = "_".join(event_tokens) if event_tokens else "(unknown)"

        lbcl = parts[-2] if len(parts) >= 2 else "(unknown)"
        set_num = parts[-1] if len(parts) >= 1 else "(unknown)"
        col_key = f"{lbcl}_{set_num}"

        best_cfg_path = os.path.join(root, "best_cfg.txt")
        try:
            with open(best_cfg_path, "r", encoding="utf-8") as f:
                text = f.read()
            m = f1_pat.search(text)
            if not m:
                continue
            f1 = float(m.group(1))
            table[event][col_key] = f1
            all_cols.add(col_key)
        except Exception as e:
            print(f"Warning: failed reading {best_cfg_path}: {e}", file=sys.stderr)

    return table, sorted(all_cols)

def save_table_as_csv(table, columns, output_path):
    import pandas as pd

    df = pd.DataFrame.from_dict(table, orient="index")
    df = df.reindex(columns=columns)  # ensure consistent column order
    df.index.name = "Event"
    df.to_csv(output_path)
    print(f"Saved table to {output_path}")

def main():
    base_dir = r"..\artifacts\humaid\bertweet5"  # adjust as needed
    table, columns = find_best_cfg_scores(base_dir)
    if not table:
        print("No F1 scores found.")
        return
    save_table_as_csv(table, columns, "bertweet_f1_summary.csv")

if __name__ == "__main__":
    main()
