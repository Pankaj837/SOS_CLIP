# Week 5 — Training Loop and Evaluation


## Overview

Week 5 moved from a verified forward pass (Week 4) to a full training loop with checkpointing, LR scheduling, and Recall@K evaluation, closing the mid-term/end-term gap by running the first complete training pass on Flickr30k.

---

## Topics Covered

### Training Loop

Adam optimizer (lr=1e-4), StepLR scheduler (step_size=3, gamma=0.5), checkpoint saved every epoch to `clip_epoch_{N}.pt`.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
```

### Recall@K Evaluation

Computed on 1,000 held-out samples, both image→text and text→image directions.

### t-SNE

2D projection of image and text embeddings to visually inspect whether the two modalities land in a shared, structured space.

---

## Assignment

Trained the standard `CLIPModel` for 5 epochs on the full 158,915-pair dataset, evaluated Recall@K, and visualized the resulting embedding space with t-SNE.

---

## Key Results

**Training loss per epoch:**

| Epoch | Avg Loss |
|---|---|
| 1 | 3.4269 |
| 2 | 3.0827 |
| 3 | 2.7469 |
| 4 | 2.4139 |
| 5 | 2.1796 |

**Recall@K (Standard Model, 1000 samples):**

| Direction | R@1 | R@5 | R@10 |
|---|---|---|---|
| Image→Text | 0.163 | 0.669 | 0.821 |
| Text→Image | 0.144 | 0.684 | 0.841 |

All numbers pulled directly from `Notebooks/SOS_endterm_weeks5-8.ipynb`, Cell 8 (loss per epoch) and Cell 14 (Recall@K).

---

## Reflection Questions

**Q1: Why does loss decrease smoothly but R@1 stay relatively low (0.163)?**
Loss reflects average ranking quality across the whole batch; R@1 is a strict top-1 metric. With only 30K images and batch size 64 (63 negatives per step), the model gets consistently better at rough discrimination but still struggles with fine-grained top-1 precision, a signal that harder negatives could help (addressed in Week 6).

**Q2: Why checkpoint every epoch rather than just at the end?**
Enables warm-starting later experiments (hard negative fine-tuning in Week 6) from the best intermediate state, and protects against losing progress on long Kaggle sessions.

---

**Notebook:** `Notebooks/SOS_endterm_weeks5-8.ipynb`, Cells 7–8 (device setup, standard training), Cell 12 (`evaluate_recall`), and the Standard Model portions of Cells 14–16 (evaluation run, loss curves, t-SNE).
## Conclusion & Next Steps

Week 5 produced a solid, fully-trained baseline model with real, evaluated numbers.

In **Week 6**, hard negative mining and an intermediate fusion variant are compared against this baseline.
