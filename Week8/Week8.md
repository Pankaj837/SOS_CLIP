# Week 8 — Final Model, Report and Documentation

## Overview

Week 8 consolidated all four trained model variants : standard, hard negative, frozen backbone, and intermediate fusion, into one final, reproducible comparison and codebase.

---

## Topics Covered

### Final Consolidated Run

All four models trained and evaluated within a single, clean, top-to-bottom notebook: no duplicate cells, no dead code, correct warm-start dependencies between cells.

### Final Codebase

22 cells: imports → both model architectures → dataset → both loss functions → device setup → 4 training loops → 2 evaluation functions → loss curves (all 4 models) → t-SNE (all 4 models) → error analysis → temperature study → qualitative retriever → final summary → model save.

---

## Assignment

Ran the complete, corrected notebook end-to-end and produced one final results table across all four model variants.

---

## Key Results

**Final Recall@K comparison — all models:**

| Model | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---|---|---|---|---|---|
| Standard | 0.163 | 0.669 | 0.821 | 0.144 | 0.684 | 0.841 |
| Hard Negative | 0.189 | 0.833 | 0.946 | 0.176 | 0.855 | 0.954 |
| Frozen Backbone | 0.074 | 0.291 | 0.433 | 0.067 | 0.277 | 0.411 |
| Intermediate Fusion | 0.010 | 0.040 | 0.100 | 0.010 | 0.040 | 0.100 |

**Best overall model: Hard Negative** : best across every metric, in both retrieval directions.

All numbers pulled directly from `Notebooks/SOS_endterm_weeks5-8.ipynb`, Cell 20.

---

## Reflection Questions

**Q1: What does the full 4-way comparison establish, taken together?**
A clear ranking of what matters most for this task at this scale: full backbone fine-tuning >> frozen backbone; late fusion >> undertrained intermediate fusion; and within late fusion, hard-negative refinement gives a further, substantial boost over standard training alone.

**Q2: If given more compute/time, what would be the highest-leverage next experiment?**
Training the intermediate fusion model on the full dataset for the same epoch budget as the standard model, to determine whether its poor result in Week 6 was a genuine architectural limitation or simply an undertrained comparison.

---

**Notebook:** `Notebooks/SOS_endterm_weeks5-8.ipynb`, Cell 20 (final results summary) and Cell 21 (final model save), consolidating outputs from all preceding cells (0–19).

## Conclusion & Next Steps

The project delivers a complete, verified CLIP-inspired pipeline: every architectural decision justified from Week 1's theory, every training choice validated with real Recall@K numbers, and every ablation (hard negatives, frozen backbone, fusion strategy) run and honestly reported, including the ones that didn't work as hoped.

Project complete.
