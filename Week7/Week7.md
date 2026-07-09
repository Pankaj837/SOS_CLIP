# Week 7 — Error Analysis and Improvements

## Overview

Week 7 dug into *why* the hard-negative model still gets things wrong, studied temperature sensitivity, compared a frozen backbone against full fine-tuning, and built a qualitative text-to-image retriever.

---

## Topics Covered

### Error Analysis

For each image, checks whether the top-1 retrieved caption matches the correct one; when it doesn't, logs both captions and their similarity scores.

### Temperature Study

Sweeps fixed temperature values against precomputed embeddings to see how loss responds to sharper/softer similarity distributions.

### Frozen vs. Fine-Tuned Backbone

`model_frozen` freezes both ResNet18 and DistilBERT backbones (only projection heads + temperature train), isolating how much of the model's performance comes from backbone adaptation vs. the shared embedding space alone.

---

## Assignment

Ran error analysis on 500 samples, swept 7 temperature values, trained a frozen-backbone model (3 epochs, full dataset), and built a qualitative retriever showing top-5 images for 3 text queries.

---

## Key Results

**Frozen Backbone Recall@K (1000 samples):**

| Direction | R@1 | R@5 | R@10 |
|---|---|---|---|
| Image→Text | 0.074 | 0.291 | 0.433 |
| Text→Image | 0.067 | 0.277 | 0.411 |

**Sample failure case:**
Image `1000092795.jpg` — correct caption "Two young, White males are outside near many bushes" (similarity 0.920) lost to "Two young guys with shaggy hair look at their hands..." (similarity 0.967) — a case of two highly similar valid captions for the same image, not a true model error.

**Learned temperature (HN model):** −1.3323 → exp = 0.2639

All numbers pulled directly from `Notebooks/SOS_endterm_weeks5-8.ipynb`, Cell 14 (frozen Recall@K), Cell 17 (error analysis output), and Cell 18 (temperature study).

---

## Reflection Questions

**Q1: Why does the frozen backbone underperform so much (R@1 0.074 vs. 0.189 for hard negative)?**
With the backbones frozen, only the projection heads can adapt, the underlying visual and linguistic features stay exactly as pretrained on ImageNet/general text, never specializing to Flickr30k's specific image-caption distribution. This confirms full fine-tuning is doing substantial, necessary work, not just marginal improvement.

**Q2: What do many of the "failure" cases in error analysis actually reveal?**
Several top-1 "errors," like the example above, involve two genuinely plausible captions for the same image (Flickr30k has 5 captions per image, often overlapping in content), meaning some fraction of the "error" rate reflects dataset ambiguity, not model weakness.

---

**Notebook:** `Notebooks/SOS_endterm_weeks5-8.ipynb`, Cell 10 (frozen-backbone training), Cell 17 (error analysis), Cell 18 (temperature study), Cell 19 (qualitative retriever), and the Frozen Backbone portions of Cell 14 (evaluation).

## Conclusion & Next Steps

Week 7 showed that backbone fine-tuning is essential at this scale, and that not all top-1 misses reflect true model failure.

In **Week 8**, all four model variants are consolidated into a final, reproducible comparison.
