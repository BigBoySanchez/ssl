# CAHSI Disaster AI Training Architecture

## 1. Purpose

Explain the system architecture for this research codebase: components, data flows, experiment pipelines (supervised, zero-shot, and verifymatch), and reproducibility practices.
This document provides the conceptual model for how the repository is organized and how its major components interact.

---

## 2. Conceptual Overview

* Utilize VerifyMatch to create accurate models for diaster response (w/ minimal labeling).
* Three main experiment pathways:

  1. **Supervised fine-tuning** (BERTweet)
  2. **Zero-shot prompting** (GPT)
  3. **Verifymatch (training + calibration + sweeps)**
* Shared utilities for preprocessing, evaluation, and table generation.
* Key artifacts produced:

  * Prediction CSVs
  * Summary / result tables

* Paper PDFs (`2509.16516v2.pdf`, `Gupta_ISCRAM2025.pdf`, `verifymatch_paper.pdf`)

---

## 3. HumAID Experiments
There are only **7** files to focus on.
- `train.py`: this houses the wandb logic and VerifyMatch code (adapted for classification on HumAID)
* `utils/run_humaid.py`: contains functions for simplifying dataset paths
- `requirements.txt`
---
* `sweep_ids.txt` (secret): each row has `sweep_id, event + labels per class`
- `make_sweep.ipynb`: notebook for making sweeps using wandb sdk (currently not needed)
* `make_container.sh` (local hosting only): handles sweep scheduling while doing fine-tuning on a local machine/server
- `colab_cahsi_disaster_train.ipynb` (Colab only): handles sweep scheduling on Google Colab

