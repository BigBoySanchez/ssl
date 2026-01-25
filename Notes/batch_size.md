# The Beginner's Guide to LLM Batching & Memory

If you are a CS undergrad, the hardest part of Deep Learning isn't the math—it's the **VRAM**. Top-tier models (GPT-4, Llama 3) are so big they won't fit on any single GPU. Here is how they handle batches without crashing.

## 1. The Trinity of Batching
There are three different "batch sizes" you need to keep track of:

1.  **Micro Batch Size (MBS):**
    *   **Definition:** How many items you process in a single "forward pass" on **one** GPU.
    *   **The Constraint:** This is limited by your **Hardware**. If you set this too high, you get the dreaded `torch.cuda.OutOfMemoryError` (OOM).
    *   **CS Analogy:** How many books you can physically carry in your hands at once.

2.  **Gradient Accumulation (GA):**
    *   **Definition:** A loop that runs multiple forward/backward passes and "sums up" the gradients before actually updating the model's weights.
    *   **The Purpose:** It lets you pretend you have a massive GPU when you actually have a small one.
    *   **CS Analogy:** Taking 10 trips to move 100 boxes because you can only carry 10 at a time. After 10 trips, you "finish" the task.

3.  **Global Batch Size (GBS):**
    *   **Definition:** The total number of items the model "sees" before it takes a single step of learning.
    *   **The Formula:** 
        $$GBS = MBS \times GA \times \text{Number of GPUs}$$
    *   **CS Analogy:** The total number of boxes moved in one full work cycle.

---

## 2. How to Pick the Numbers?

### Micro Batch Size (MBS)
*   **Pick the largest value that doesn't OOM.**
*   Why? GPU computation is "throughput-optimized." It's faster to process 8 items at once than to process 1 item eight times.
*   *Note:* It’s usually a power of 2 (2, 4, 8, 16) because of how GPU memory cores (Warps/Threads) are aligned.

### Global Batch Size (GBS)
*   **Pick based on Scaling Laws.** 
*   Research shows that if your batch is too small, the gradients are "noisy" (the model gets confused by outliers). If it's too big, you waste electricity.
*   **SOTA Rule of Thumb:** Start small (e.g., 1M tokens) and increase it as the model gets smarter.

### Accumulation Steps
*   **The Math:** Total desired GBS divided by (MBS * GPUs).
*   If you want a GBS of 128, but you only have 1 GPU and it OOMs at MBS=4:
    *   Set `accumulation_steps = 32`.

---

## 3. Weighted Gradients (The "Importance" Factor)

Sometimes, simply seeing data isn't enough. We "weight" gradients for two main reasons:

### A. Handling Class Imbalance
If 99% of your data is "Normal" and 1% is "Crisis," the model will just learn to predict "Normal" every time. 
*   **The Fix:** You multiply the loss of the "Crisis" samples by a weight (e.g., $10 \times$). 
*   **Result:** When the model gets a "Crisis" sample wrong, the gradient is 10x larger, forcing the optimizer to pay more attention.

### B. Precision Scaling (FP16/BF16)
Computers are bad at tiny numbers. If your gradients are too small ($0.000000123$), they might be rounded to $0$ by the 16-bit float.
*   **The Fix:** Use a **GradScaler**. Multiply the gradients by $65,536$ before doing math, then divide them back down. This keeps them from "falling off" the bottom of the number line.

---

## 4. Pro-Tips for the Lab
*   **Gradient Checkpointing:** If you still OOM with `MBS=1`, turn this on. It deletes intermediate math and re-calculates it later. It's 30% slower but uses ~70% less memory.
*   **Learning Rate Rule:** If you increase your Global Batch Size, you usually need to increase your Learning Rate. A bigger batch means you are more "sure" about the direction, so you can take a bigger step.

---

## 5. Advanced VRAM Management for Co-Training

### Techniques You're Already Using ✓
*   **Manual Garbage Collection:** `gc.collect()` and `torch.cuda.empty_cache()` after deleting models (Lines 522-523 in your script).
*   **Model Lifecycle:** Deleting and re-initializing models between training phases to avoid holding stale references.
*   **Multi-GPU Split:** Using two devices (`device_1`, `device_2`) to physically separate model memory footprints.

### Additional Techniques to Consider

#### A. Gradient Checkpointing (Activation Recomputation)
**What it does:** Trades compute for memory by deleting intermediate activations during the forward pass and recomputing them during backprop.

**How to enable (HuggingFace):**
```python
model.gradient_checkpointing_enable()
```
**When to use:** If you OOM even with `batch_size=1`. This is especially useful for BERTweet/RoBERTa since they have 12 transformer layers.

**Cost:** ~30% slower training, but can reduce activation memory by **70%**.

---

#### B. Mixed Precision Training (FP16/BF16)
**What it does:** Uses 16-bit floats instead of 32-bit, cutting memory usage in half.

**How to enable (PyTorch):**
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

# In your training loop:
with autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**When to use:** If you have modern GPUs (V100, A100, H100) that support Tensor Cores. BERTweet will see ~2x speedup + 50% memory reduction.

**Warning:** Some models are unstable in FP16. Use `torch.bfloat16` instead if you see NaN losses.

---

#### C. DataLoader Optimization
**What to change:**
```python
DataLoader(dataset, batch_size=bs, pin_memory=True, num_workers=4)
```

**Why:**
*   `pin_memory=True`: Speeds up CPU → GPU data transfer by ~20%.
*   `num_workers=4`: Loads the next batch while the GPU is computing, eliminating I/O wait time.

**When to use:** Always. This is free speedup with no downside (unless you're CPU-limited, which is rare).

---

#### D. Optimizer State Sharding (8-bit AdamW)
**What it does:** AdamW stores 2 extra copies of your model's parameters (momentum + variance). For a 110M param model like BERTweet, that's ~1.3GB extra per model.

**How to enable (bitsandbytes):**
```python
import bitsandbytes as bnb

optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
```

**When to use:** If you're training large models (RoBERTa-Large, DeBERTa) and the optimizer state is eating your VRAM.

**Savings:** ~75% reduction in optimizer memory (from 8 bytes/param to 2 bytes/param).

---

#### E. CPU Offloading (Only for Desperate Situations)
**What it does:** Moves model parameters/gradients to CPU RAM when not actively being used.

**How to enable (DeepSpeed):**
```python
# Requires DeepSpeed installed
# Add to your training config:
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  }
}
```

**When to use:** Only if you have a small GPU (8GB) and a large CPU RAM (64GB+). This is **very slow** (10x slower) but lets you train models that wouldn't fit otherwise.

---

### Quick Decision Tree for Your Script

```
Start OOM? 
├─ No → You're good. Focus on hyperparameter tuning.
├─ Yes → Reduce batch_size to 1
    ├─ Still OOM? → Enable Gradient Checkpointing
        ├─ Still OOM? → Switch to 8-bit AdamW
            ├─ Still OOM? → Use Mixed Precision (FP16)
                └─ Still OOM? → Time to get a bigger GPU or use CPU offloading
```

---

### What NOT to Do
*   **Don't spam `torch.cuda.empty_cache()` in your training loop.** It's expensive (~10ms) and PyTorch's caching allocator is smarter than you think. Only call it after major lifecycle events (deleting entire models).
*   **Don't use `del` without `gc.collect()`.** Python's garbage collector is lazy. Always pair them.
*   **Don't use DataParallel (DP).** Use DistributedDataParallel (DDP) instead. DP has ~30% overhead due to single-process bottlenecks.
