# Week 2 — Contrastive Learning and Embeddings

## Overview

Week 2 covered the training paradigm behind CLIP: contrastive learning via InfoNCE loss, the role of a learnable temperature, and why hard negatives eventually become necessary.

---

## Topics Covered

### InfoNCE Loss

For a batch of N pairs, computes an N×N similarity matrix and applies symmetric cross-entropy:

```python
logits = (image_embeddings @ text_embeddings.T) / temperature.exp()
labels = torch.arange(N)
loss = (cross_entropy(logits, labels) + cross_entropy(logits.T, labels)) / 2
```
**Definition:** For an anchor x with positive x⁺ and K negatives:

L = −log( exp(sim(x, x⁺)/τ) / [ exp(sim(x, x⁺)/τ) + Σₖ₌₁ᴷ exp(sim(x, x⁻ₖ)/τ) ] )

Here **x** is the anchor embedding (e.g. an image), **x⁺** its matching positive (the paired caption), **x⁻ₖ** the K non-matching negatives, **sim(·,·)** the similarity function (cosine similarity, i.e. the dot product of L2-normalized embeddings), and **τ** the temperature. In the batched CLIP setting, every other item in a batch of N pairs is a negative, so K = N − 1, and the loss is applied in both directions and averaged.

**Why InfoNCE, and what it measures.** InfoNCE frames training as instance discrimination — picking the correct positive among many distractors, which maximizes a lower bound on the mutual information between the two modalities. Minimizing it pulls matched pairs together and pushes mismatched pairs apart: exactly the cross-modal alignment this project needs. It also explains the sanity check below — with random embeddings the model can only assign each of N candidates probability 1/N, giving −log(1/N) = log(N) ≈ 4.16 for N = 64.



### Temperature

A learnable scalar, initialized to 0.07, exponentiated during loss computation to stay positive. Controls how sharp/soft the similarity distribution is.

### Why Random Negatives Aren't Enough

Early in training, random negatives are trivially easy to reject. As the model improves, gradient signal from random negatives vanishes, this motivates hard negative mining (implemented in Week 6).

---

## Assignment

Implemented InfoNCE from scratch and verified it against the theoretical random-model baseline: log(batch_size).

---

## Key Results

| Check | Expected | Observed |
|---|---|---|
| Random-model loss (batch=64) | log(64) = 4.1589 | 4.1578 ✅ |

Confirms the loss implementation is correct, a random model should have maximum uncertainty across all 64 candidates. This verification is captured in `Notebooks/SOS_midterm_weeks1-4.ipynb`, Cell 6 (forward pass test) and Cell 8 (loss sanity check). Conceptual foundation for the loss function implemented in Week 4 (`Notebooks/SOS_midterm_weeks1-4.ipynb`, Cell 7 `infonce_loss`) and extended in Week 6 (`Notebooks/SOS_endterm_weeks5-8.ipynb`, Cell 6 `hard_negative_infonce_loss`).

---

## Reflection Questions

**Q1: Why divide by `temperature.exp()` instead of the raw parameter?**
The raw learnable parameter can go negative during training; exponentiating guarantees the divisor is always positive.

**Q2: Why is the loss symmetric (both `logits` and `logits.T`)?**
Training in both directions (image→text and text→image) gives twice the gradient signal and ensures the embedding space works for both retrieval directions, not just one.

---

## Conclusion & Next Steps

Week 2 built the core training mechanism from first principles, which is why Week 4's implementation could be written from scratch rather than copied.

In **Week 3**, the focus shifts to the architecture that produces the embeddings being compared.
