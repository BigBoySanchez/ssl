# Context: Computing ECE from W&B Run Artifacts (UPDATED & VERIFIED)

## Goal

Compute **Expected Calibration Error (ECE)** for trained models **without retraining**, using **existing Weights & Biases (W&B) prediction artifacts**. This is a **post-hoc analysis** intended for paper-quality reporting and reproducibility.

---

## High-Level Summary (Read First)

* **ECE is computed exclusively from per-example prediction artifacts** (JSONL files).
* **A confidence field (`conf`) is mandatory**. If it is missing, ECE is undefined.
* **No models are rerun**; all calculations rely on stored W&B artifacts.
* **ECE is computed once per (event, label budget, test set)**, even if multiple artifact versions exist.
* Results are aggregated and analyzed externally (Pandas → CSV → Google Sheets).

---

## Prediction Artifact Requirements (Strict)

### ✅ Valid / ECE-Compatible JSONL

Each line must include:

```json
{
  "label": <int>,   // true class
  "pred": <int>,    // predicted class (argmax)
  "conf": <float>   // max softmax probability ∈ [0,1]
}
```

Notes:

* Additional keys (e.g., `guid`, `text`) are ignored.
* `conf` must already be computed during inference (typically via `max(softmax(logits))`).

### ❌ Invalid / Must Be Ignored

```json
{"label": ..., "pred": ...}
```

These files **cannot be repaired post-hoc** and must be skipped entirely.

> **Hard rule:** No `conf` → no ECE.

---

## ECE Definition (What Exactly We Compute)

We compute **standard top-1 Expected Calibration Error**, consistent with:

> *Gupta et al., ISCRAM 2025 – Calibrated Semi-Supervised Models for Disaster Response*

Formal definition:

[
\mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|
]

Where:

* `conf` = predicted confidence (`conf` field)
* `acc` = accuracy indicator `(pred == label)`
* bins are **uniform over [0, 1]**
* **number of bins = 4** (paper setting)

This is **top-1 ECE only**.

---

## Alignment With Other ECE Implementations

The implemented ECE function is mathematically equivalent to the commonly used formulation:

```python
ECE = sum_m |acc(B_m) - conf(B_m)| * (|B_m| / N)
```

Differences checked and validated:

* Bin boundaries: `(lo, hi]` (uniform bins)
* Accuracy: `pred == label`
* Weighting: `count / N`

Minor edge-case difference:

* Some implementations exclude `conf == 0` from the first bin.
* Our implementation **includes `conf == 0` in bin 1**.
* This only affects results if exact zeros appear (rare with softmax).

Conclusion: **methods align**; numerical differences are negligible and explainable.

---

## Canonical ECE Function (Used for All Results)

The following logic is used consistently for all reported numbers:

* Parse JSONL
* Validate keys and confidence range
* Compute per-bin accuracy and mean confidence
* Aggregate weighted absolute differences

Key properties:

* Skips malformed rows (configurable)
* Enforces `0 ≤ conf ≤ 1`
* Uses NumPy for deterministic behavior

This function is the **single source of truth** for ECE.

---

## W&B Artifact Handling (Detailed & Auditable)

### Artifact Downloading

For each selected run:

1. Iterate over `run.logged_artifacts()`
2. Select artifacts whose **name starts with `"preds"`**
3. Download artifacts **without overwriting existing directories**

Artifacts are stored under:

```
wandb-preds/
  ├── preds-<event>-lb<k>-set<j>-seed<s>-vXYZ/
```

Multiple versions (`vXYZ`) may exist for the same logical test set.

---

### Artifact Filtering & De-duplication

For each artifact directory:

* Locate `.jsonl` files
* Inspect the **first non-empty line**:

  * if `"conf"` exists → eligible
  * else → skip

To avoid duplicate computation:

* Extract **set number** from folder name (`set1`, `set2`, `set3`)
* **Compute ECE once per set number**, ignoring additional versions

This guarantees **one ECE per test set**.

---

## Run Selection Strategy

* Iterate over all sweeps in the project
* For each sweep:

  * Select **best run by `dev_macro-F1`**
  * Download prediction artifacts
  * Compute ECE for sets 1, 2, and 3

Stored per sweep:

```json
{
  "sweep_name": "kerala_floods_2018_25lbcl",
  "ece_set1": 0.043,
  "ece_set2": 0.106,
  "ece_set3": 0.029
}
```

---

## Metadata Extraction (Event & Label Budget)

Some runs lack clean config fields. Therefore:

* **Event** and **label budget (lbcl)** are parsed directly from the sweep/run name

Example:

```
kerala_floods_2018_25lbcl
```

Parsed as:

* `event = kerala_floods_2018`
* `lbcl = 25`

Sorting is done by:

```
event → lbcl (numeric)
```

---

## Aggregation & Reporting

### Pandas

* ECE values stored in a DataFrame
* One row per `(event, lbcl)`
* Columns: `ece_set1`, `ece_set2`, `ece_set3`

### Google Sheets

Workflow:

1. Export DataFrame to CSV
2. Upload CSV to Google Sheets
3. Use **Pivot Tables** for aggregation

Pivot configuration:

* Rows: `event`, `lbcl`
* Values: `AVERAGE(ece_set1)`, `AVERAGE(ece_set2)`, `AVERAGE(ece_set3)`
* Optional calculated field:

```
(avg_ece_set1 + avg_ece_set2 + avg_ece_set3) / 3
```

This produces reviewer-ready summary tables.

---

## Explicit Constraints & Validity Guarantees

* Only **top-1 ECE** is computed
* No confidence reconstruction or approximation
* No retraining or re-inference
* Some historical runs are unusable — expected and acceptable

Each reported ECE value is traceable to:

* a specific W&B run
* a specific artifact
* a specific JSONL file
* a fixed, documented computation procedure

---

## Final Checklist (For Verification)

✔ Artifact contains `conf`
✔ Correct bin count (4)
✔ No duplicate set computation
✔ Correct event/lbcl parsing
✔ Deterministic function used everywhere
✔ Aggregation performed *after* computation

If all boxes are checked, the ECE numbers are **valid, reproducible, and defensible**.
