# 9/25 Prep

## GOAL
Run VerifyMatch on some crisis datasets...
- crisismmd
- humAID

DO 15,000 PG LABELS

## GPT Explanation
Awesome — let’s redo the “work backwards” plan and explicitly weave **calibration** into the pipeline you’ve got (notes + `train.py`).

# 🎯 Goal (re-stated)

Run **VerifyMatch** on **CrisisMMD** and **humAID** and produce **calibrated** results (accuracy/F1 + ECE/Brier-style confidence quality) you can trust. Your notes already set the target datasets【】.

---

# 🔄 Work backwards from “done”

## 0) Final deliverables (what “done” looks like)

* Calibrated test outputs for each dataset: JSONL with `true`, `pred`, `logits`, `probs`, `conf`, written by `train.py` evaluate step【】【】.
* A saved **calibration parameter** file (e.g., temperature τ).
* Reported metrics (Accuracy, Macro-F1 from `train.py`; plus your calibration metrics computed from calibrated probs)【】.
* (If using SSL/VerifyMatch) Training-time logs that used **calibrated confidence** to decide high/low buckets for unlabeled data (i.e., the step that softmaxes logits and takes the max confidence)【】.

---

## 1) Calibrated evaluation & reporting

1. **Evaluate → dump raw (uncalibrated) predictions**
   Use `--do_evaluate` to write JSONL with logits/probs/conf; `train.py` does this already for test/dev and prints Accuracy/F1【】【】.

2. **Fit calibration on dev predictions**

   * Train **post-hoc** calibration (e.g., temperature scaling) using the dev JSONL (inputs: `logits`/`true`).
   * Save τ (temperature).

3. **Apply calibration to test predictions**

   * Re-evaluate (or reuse raw test JSONL), apply τ to logits → `softmax(logits/τ)` to get calibrated `probs`, recompute ECE/Brier/coverage-style metrics.
   * Keep both raw and calibrated JSONLs for auditing.

---

## 2) Calibrated VerifyMatch (SSL) training

VerifyMatch decides **high/low-confidence** unlabeled samples using softmax probabilities derived from logits, with optional sharpening `T`【】 and then takes the max as `verifier_prob`【】.
To align that decision with reality, apply your learned **τ** inside this loop:

* Replace:
  `tmp_labels2 = F.softmax(logits2, dim=-1)`
* With (conceptually):
  `tmp_labels2 = F.softmax(logits2 / τ, dim=-1)`
* (Then your existing sharpening with `args.T` still applies)【】.

This ensures the **confidence** used for filtering/mixing unlabeled examples is calibrated before you threshold/split (you can still use your CLI threshold `--th` and sharpening `--T`)【】.

Artifacts you’ll already get from the SSL path (progress files, confidence tracking) continue to work and become more meaningful when confidence is calibrated.

---

## 3) Supervised training + checkpointing (the base model you calibrate)

* Train on your labeled split(s); `train.py` saves the best checkpoint based on dev accuracy【】.
* Later, `--do_evaluate` loads that checkpoint and emits the JSONL outputs【】.

Key flags you’ll use a lot:

* `--ssl` to enable the VerifyMatch path during training (semi-supervised)【】
* `--th`, `--T` to control thresholding and sharpening【】

---

## 4) Data prep (match your datasets to the processors)

Your code expects **pair** inputs (sentence1, sentence2, label\[, guid]) for NLI/QQP-style tasks; the tokenizer builds `[CLS] s1 [SEP] s2 [SEP]` pairs via `encode_pair_inputs`【】【】.
Pick a processor whose label schema fits your data:

* **RTEProcessor** expects labels in `{'entailment','not_entailment'}` which it maps to {1,0}【】【】.
* **QQPProcessor** expects binary labels `'0'/'1'`【】【】.

Format **CrisisMMD**/**humAID** into TSVs compatible with the chosen processor; then point `--train_path/--dev_path/--test_path` to those files. The code will load and batch them through `TextDataset` → processor → `encode_pair_inputs`【】【】.

---

# 🛠️ Concrete commands (templates)

## A) Supervised baseline (per dataset)

Train → best ckpt → dev dump for calibration:

```bash
python train.py \
  --device 0 \
  --model roberta-base \
  --task RTE \
  --train_path data/<ds>/train.tsv \
  --dev_path data/<ds>/dev.tsv \
  --test_path data/<ds>/test.tsv \
  --ckpt_path ckpts/<ds>_rte_roberta.pt \
  --output_path outputs/<ds>_dev_raw.jsonl \
  --do_train --do_evaluate
```

This writes JSONL with `true/pred/conf/logits/probs` for dev/test as used by your evaluation path【】.

Fit temperature τ on dev JSONL, then apply to the test JSONL (using your `calibrate.py`).

## B) SSL with VerifyMatch (calibrated)

```bash
python train.py \
  --device 0 \
  --model roberta-base \
  --task RTE \
  --labeled_train_path data/<ds>/train.tsv \
  --unlabeled_train_path data/<ds>/unlabeled.tsv \
  --dev_path data/<ds>/dev.tsv \
  --test_path data/<ds>/test.tsv \
  --ckpt_path ckpts/<ds>_ssl_ckpt.pt \
  --output_path outputs/<ds>_ssl_test_raw.jsonl \
  --ssl --th 0.7 --T 0.1 \
  --do_train --do_evaluate
```

Then apply τ inside the VerifyMatch loop at the point where `tmp_labels2 = F.softmax(logits2, dim=-1)` is computed (insert `/ τ` before softmax) and keep `--th`/`--T` as knobs【】.

---

# ✅ Minimal checklist to execute

* [ ] **Prep data** into the pair-TSV format your chosen processor expects (RTE/QQP)【】【】
* [ ] **Train baseline** (supervised) → best ckpt saved on dev accuracy【】
* [ ] **Evaluate** to write **dev** & **test** JSONLs with raw `logits/probs/conf`【】
* [ ] **Fit τ on dev**, **apply** to test and to the **VerifyMatch** loop (divide logits by τ before softmax)【】
* [ ] **Report** Accuracy/F1 from `train.py` plus **calibration** metrics from calibrated probs【】

If you want, paste a snippet of your `calibrate.py` flags and I’ll drop in the exact one-liners (fit/apply) and the tiny code change to `train.py` at the **pseudolabel generation** site so VerifyMatch uses calibrated confidence.


## 09/25 Meeting
- for input, just use tweet txt
- try to understand bert last output
- labeled = 200 OR 400 OR 600/class
- unlabeled = rest of data, relabeled by an llm... or n # of pseudo-GENERATED samples

- FOLLOW DONGPING's model
- use gemini for...
- DONGPING REPO

https://github.com/deeplearning-lab-csueb/Fine-tune-Multimodal-LLM-for-CrisisMMD

https://unsloth.ai/blog/gemma3

https://github.com/deeplearning-lab-csueb/Fine-tune-Multimodal-LLM-for-CrisisMMD/blob/main/Anh-New/GPT-Fewshot-Informative.ipynb

what is a zero-shot
gemma3

scikit

make sure ssl is better than supervised

## 10/1 TASKS:
- cvt .arrow datasets into human-readable ones
- DO NOT pass any data to the pseudo-generator, just schema

## 10/1 meeting:
- gpt4-o zero-shot (85% inf, 80% hum) [use gpt 5 now] vs supervised model (BERT; baseline) vs verifymatch <- test set foreach
- get plabel, then run vmatch
- find the best model (hyperparams)
  - 100/class
- get zero-shot v bert v vmatch table by friday, 11:00am (cols = f1, accuracy, precision & recall, weighted average; all in scikit learn)
- ask about cols again

## 10/9

- humaid dataset processed and ready for label-plabel separation 
- separate based on selected, labeled examples
- adapt vmatch/train for the dataset
- adapt bert for the dataset
- HERE -> make/run a script that runs train & eval for each set
- (optional; for easier, more reliable training later) install/use ncat to watch training w/o breaking anything
- GOAL -> make a table for humaid, ideally w/ all lb/cl sets

### 🧩 Data & Experiment Pipeline

1. **humaid only**

   * Join all events
   * Split into **train / dev / test**

2. **labeled_data**

   * Split into **[train | test]**
   * Strip away everything → keep only inputs

3. **union set**
   * use chatgpt5 to add a label column to data
   * join the labeled and pseudolabeled set on tweet text
   

4.  **label, plabel split**   
    * Separate into **labeled** and **unlabeled**
    * use **gold** & **pred** labels repsectively

4. **Modeling**

   * `vmatch(labeled, plabeled, dev, test)`
   * `bert(dev, test, labeled)`
   * `zero_shot(prompt, input-only test)`

5. **Output**
   * Generate CSV: `[id, gold, pred]`
   * `make_table`

## 10/9 NOTES
- cornelia got crisismmd
- focus on humaid
- for text, gpt 4mini is best

- run for 5lb all sets, ideally all labels all sets, then avg
- use both bert and bertweet

https://github.com/deeplearning-lab-csueb/Fine-tune-Multimodal-LLM-for-CrisisMMD/blob/main/ASONAM2025%20(2).pdf

joined row count = split row count on paper + num labeled rows
bertweet expects user + twt link, but opting to leave it out
hf automatically sets train models to eval when running pred