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

Confirms the loss implementation is correct, a random model should have maximum uncertainty across all 64 candidates. This verification is captured in `Notebooks/SOS_midterm_weeks1-4.ipynb`, Cell 6 (forward pass test) and Cell 8 (loss sanity check). This week, conceptual foundation for the loss function implemented in Week 4 (`Notebooks/SOS_midterm_weeks1-4.ipynb`, Cell 7 `infonce_loss`) and extended in Week 6 (`Notebooks/SOS_endterm_weeks5-8.ipynb`, Cell 6 `hard_negative_infonce_loss`).

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
