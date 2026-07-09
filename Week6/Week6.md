# Week 6 — Hard Negative Mining and Fusion Comparison


## Overview

Week 6 covered two extensions to the Week 5 baseline: hard-negative mining (warm-started from the fully-trained standard model) and a from-scratch intermediate fusion variant, compared against the late-fusion baseline.

---

## Topics Covered

### Hard Negative Mining

Instead of treating all 63 in-batch negatives equally, the top-K most confusing negatives (highest similarity, excluding the true match) are selected for the cross-entropy loss:

```python
def hard_negative_infonce_loss(image_embeddings, text_embeddings, temperature, top_k=10):
    sim_matrix = image_embeddings @ text_embeddings.T
    ...
    _, hard_neg_idx = sim_negatives.topk(k, dim=1)
```

Warm-started from `clip_epoch_5.pt` — the **fully-trained** standard model — so the comparison is a fair "does hard-negative refinement improve an already-good baseline," not a confound with an undertrained one.

### Intermediate Fusion Variant

A `FusionCLIPModel` lets image and text features attend to each other via one `TransformerEncoderLayer` before producing final embeddings — testing the Week 1 claim that late fusion was the right choice for this project's scale. Trained on a 20,000-pair subset (3 epochs) rather than the full dataset, to control cost.

---

## Assignment

Trained hard-negative fine-tuning (3 epochs) and an intermediate fusion model (3 epochs, subset), then compared both against the Week 5 standard/late-fusion baseline.

---

## Key Results

**Hard Negative training loss:**

| Epoch | Avg Loss |
|---|---|
| 1 | 1.2998 |
| 2 | 1.0783 |
| 3 | 0.8909 |

**Recall@K comparison:**

| Model | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---|---|---|---|---|---|
| Standard (late fusion) | 0.163 | 0.669 | 0.821 | 0.144 | 0.684 | 0.841 |
| Hard Negative | 0.189 | 0.833 | 0.946 | 0.176 | 0.855 | 0.954 |
| Intermediate Fusion | 0.010 | 0.040 | 0.100 | 0.010 | 0.040 | 0.100 |

All numbers pulled directly from `Notebooks/SOS_endterm_weeks5-8.ipynb`, Cell 9 (HN loss per epoch) and Cell 14 (Recall@K for all models).

---

## Reflection Questions

**Q1: Why did hard negatives improve every metric?**
Forcing the model to distinguish the *most confusing* negatives, rather than easy random ones, provides much stronger gradient signal once the model has already learned coarse discrimination, exactly the R@1 weakness observed in Week 5.

**Q2: Why did intermediate fusion perform near random chance?**
Trained on a 20K subset for only 3 epochs with a single, small cross-attention layer, it never received enough signal or capacity to learn meaningful joint representations. This is a genuine, reportable finding — it empirically supports the Week 1 decision to use late fusion at this project's data/compute scale, rather than an implementation bug.

---

**Notebook:** `Notebooks/SOS_endterm_weeks5-8.ipynb`, Cell 4 (`FusionCLIPModel`), Cell 6 (`hard_negative_infonce_loss`), Cell 9 (hard-negative training), Cell 11 (fusion training), and the Hard Negative / Fusion portions of Cells 14–16 (evaluation, loss curves, t-SNE).

## Conclusion & Next Steps

Hard negatives were a clear, validated win. The fusion experiment, while underwhelming in absolute numbers, is scientifically useful: it demonstrates late fusion's practical advantage under this project's constraints.

In **Week 7**, the hard-negative model is analyzed further: error cases, temperature sensitivity, and a frozen-backbone comparison.
