# Generative-Recommendation-Pytorch

This repository contains my solution for the **Tencent Advertising Algorithm Competition**.

## Background

The competition organizers provided a **baseline model**, which I think is modified from **SASRec(Self-Attentive Sequential Recommendation)**.  
This repository contains my improvements over that baseline, including model design, architecture changes, and optimization for efficiency.

## What I Did

- **Implemented HSTU model**
  - Inspired by Meta's paper *"Actions Speak Louder Than Words"* and the HSTU architecture [Du et al., 2024].
  - Removed the **softmax** layer and **ffn** layer in self-attention, reducing complexity from **O(n²)** to **O(n)** (while modifying the structure to retain expressive power).
  - Added **Relative Attention Bias (RAB)** to capture relative order in both **position** and **time**, improving generalization.

- **Modified tower architecture**
  - Baseline: dual-tower (user tower + item tower).
  - My version: **three-tower model**:
    - **User Tower**: a small **DNN (Deep Neural Network)** to encode user features.
    - **Item Tower**: a small **DCN (Deep & Cross Network)** to encode positive/negative item samples.
    - **Sequence Tower**: the most complex part; first encodes historical items with the DCN, then integrates user information (position and time embeddings) via **cross-attention**, and finally feeds the sequence into the **HSTU model**.

- **Loss function: switched to InfoNCE**
  - Replaced **BCE (Binary Cross-Entropy)** with **InfoNCE** for contrastive learning.
  - Used **in-batch negatives**, specifically **cross-time negatives**, as negative samples.
  - Potential optimization: first train with cross-time negatives for better generalization, then fine-tune the final 1–2 epochs with task-specific conditions.

- **Efficiency improvements**
  - Rewrote inefficient modules in the baseline.
  - Reduced training time from **~1 hour per epoch → ~15 minutes per epoch**.
  - Reduced infer time from **~1 hour → ~20 minutes**.
