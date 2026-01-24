# Hyperparameter Optimization (HPO) for Co-Training

This document details the theoretical and practical framework for the Hyperparameter Optimization (HPO) conducted in our Co-Training experiments (LG-Cotrain + HumAID).

## Theoretical Context

Our implementation operationalizes the Co-Training algorithm described in **[cotrain_paper.pdf]**. The core premise relies on training two classifiers ($\theta_1, \theta_2$) that iteratively label unlabelled data for each other, thereby augmenting the training set with high-confidence pseudo-labels.

### Data Handling & Independence
To satisfy the Co-Training requirement for diverse views (or at least diverse hypotheses), we strictly control the input data distribution:

1.  **Input**: We start with a labeled dataset $D_l$ (defined by `lbcl`, e.g., 50 samples per class).
2.  **Shuffling**: The entire labeled set is shuffled randomly to remove any ordering bias.
3.  **Splitting**: The set is **split evenly** into two disjoint sets, $D_{l1}$ and $D_{l2}$.
    *   $D_{l1} \rightarrow \theta_1$
    *   $D_{l2} \rightarrow \theta_2$
    
    This disjoint training ensures that $\theta_1$ and $\theta_2$ start with different decision boundaries, which is crucial for the "teaching" mechanism to work during the co-training rounds.

### The Optimization Objective
Unlike standard supervised learning where we optimize for convergence on a static set, Co-Training requires hyperparameters that balance **plasticity** (learning from new pseudo-labels) and **stability** (not drifting due to noise).

Our HPO aims to find the sweet spot for these parameters that maximizes the **Macro-F1** score on the hold-out validation set across 3 distinct data folds (Sets 1, 2, and 3).

## PLM as a Hyperparameter

In addition to scalar training parameters, we treat the **Pre-trained Language Model (PLM)** backbone itself as a categorical hyperparameter. 

$$\theta \in \{ \text{BERTweet}, \text{RoBERTa}, \text{DeBERTa}, \dots \}$$

### Theoretical Justification
The choice of PLM provides the **Inductive Bias** for the co-training process. Since co-training relies on each model providing a "view" of the data, the architecture and pre-training corpus determine the quality and distinctiveness of that view.

*   **Domain Alignment**: Models like `BERTweet` are pre-trained on Twitter data, offering a view highly aligned with our social media input (HumAID).
*   **Capacity vs. Generalization**: Models like `RoBERTa-Large` offer higher capacity but may overfit the small $D_{l1}$ more quickly than `RoBERTa-Base`.
*   **Robustness**: Features from `DeBERTa` (disentangled attention) may provide more robust pseudo-labels than standard BERT.

By optimizing the PLM, we are effectively selecting the **prior** that is best suited to bootstrap knowledge from the specific few-shot distribution of $D_l$.

## Tunable Hyperparameters

We utilize a **Bayesian** (or Grid/Random) search via Weights & Biases to tune the following key parameters. These are critical for controlling the dynamics of the Co-Training process:

| Parameter | Distribution | Range / Values | Theoretical Justification |
| :--- | :--- | :--- | :--- |
| **Learning Rate (`lr`)** | Log-Uniform | `1e-5` to `1e-3` | Controls how aggressively models adapt to new pseudo-labels. Too high leads to noise overfitting; too low leads to slow convergence. |
| **Weight Decay** | Log-Uniform | `1e-5` to `1e-2` | Regularization is vital in semi-supervised settings to prevention confirmation bias (overconfidence in own predictions). |
| **Num Epochs** | Integer Uniform | `5` to `20` | Defines the length of the *Initialization* and *Co-Training* rounds. Short epochs prevent overfitting to the small initial labeled set. |
| **Epoch Patience** | Integer Uniform | `3` to `10` | Early stopping criteria. Essential to halt training before the model memorizes the potentially noisy pseudo-labels. |
| **Max Grad Norm** | Uniform | `1.0` to `10.0` | Gradient clipping prevents exploding gradients, ensuring training stability during the volatile co-training updates. |
| **Batch Size** | Integer Uniform | `8` to `64` | Affects the stochasticity of updates. Smaller batches introduce more noise, which can be beneficial or detrimental depending on the stability of pseudo-labels. |
| **Accumulation Steps** | Integer Uniform | `1` to `4` | Simulates larger batch sizes, allowing for stable gradients even with memory constraints. |

> **Note**: These parameters specifically target the *Initialization* and *Co-Training* phases. The *Final Fine-tuning* stage (Stage 3 in the paper) typically uses a fixed, longer epoch count (e.g., 100) to converge on the refined pseudo-labels, as established in our main experimental setup.

## Technical Execution

The HPO process is designed to be robust and reproducible:

1.  **Sweep Generation**: We define the search space in `generate_sweep.py`.
2.  **Cross-Validation**: To ensure statistical significance, every HPO run (a single set of hyperparameters) is evaluated on **3 different data folds** (Set 1, Set 2, Set 3).
3.  **Aggregation**: The final metric reported to the optimizer is the **average** of the metrics from these 3 folds. This prevents overfitting hyperparameters to a specific random split of the few-shot data.

### Connection to Code
*   **Data Splitting**: Implemented in `cotrain/data_utils.py`, ensuring $D_{l1} \cap D_{l2} = \emptyset$.
*   **Model Initialization**: `cotrain/main_bertweet.py` instantiates two independent models (e.g., `BERTweet`, `RoBERTa`) which are then fed $D_{l1}$ and $D_{l2}$ respectively.
