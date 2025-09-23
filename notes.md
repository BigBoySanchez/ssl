# 9/25 Prep

## GOAL
Run VerifyMatch on some crisis datasets...
- crisismmd
- humAID

Got it — let’s “work backwards” from your goal and map the steps.

### 🎯 Goal

Run **VerifyMatch** on **CrisisMMD** and **humAID** datasets.

---

### 🔄 Work Backwards

1. **Final Output (what success looks like):**

   * You have trained/evaluated models with VerifyMatch.
   * You produce results (accuracy, F1, calibration, etc.) on CrisisMMD and humAID.
   * These results are saved (JSON/logs) for analysis.

2. **Before That (evaluation step):**

   * Use `train.py` with `--do_evaluate` to run on test/dev splits.
   * Make sure `--ckpt_path` points to your trained model.
   * Output will be JSON lines with predictions, logits, and confidences.

3. **Before That (training step):**

   * Run `train.py` with `--do_train`.
   * Need labeled (and possibly unlabeled) data paths.
   * Choose task type (likely `SNLI`-style pair classification, since humAID and CrisisMMD are NLI/entailment-like).
   * Save model checkpoint with `--ckpt_path`.

4. **Before That (data preparation):**

   * Convert CrisisMMD and humAID into the format expected by the `Processor` classes in `train.py`:

     * (sentence1, sentence2, label, guid) for NLI-style tasks.
   * Ensure consistent label mapping (entailment/contradiction/neutral or binary labels).
   * Provide `train_path`, `dev_path`, `test_path`.

5. **Before That (environment setup):**

   * Install dependencies: `torch`, `transformers`, `tqdm`, `scikit-learn`, etc.
   * Ensure GPU access (since VerifyMatch relies on BERT/RoBERTa).
   * Organize directories for datasets, checkpoints, outputs.

---

### ✅ Immediate To-Do

Since you’re at planning, your **next concrete step** is:
👉 Check the format of CrisisMMD + humAID and decide which `Processor` (SNLI, RTE, etc.) in `train.py` they align with.

Do you want me to break down **what data format CrisisMMD and humAID are in** and **which processor class fits best**? That’s the key decision before you can start preparing inputs.
