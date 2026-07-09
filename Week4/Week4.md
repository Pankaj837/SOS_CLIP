# Week 4 — Pipeline Implementation

## Overview

Week 4 turned four weeks of theory into working code: dataset loading, both encoders, the full CLIPModel, and InfoNCE loss all verified end-to-end.

---

## Topics Covered

### Flickr30k Dataset

31,783 images × 5 captions = 158,915 image-caption pairs, loaded via a pipe-separated CSV.

```python
class Flickr30kDataset(Dataset):
    def __init__(self, csv_path, images_path, transform=None):
        self.df = pd.read_csv(csv_path, sep='|')
        self.df.columns = self.df.columns.str.strip()
        self.df['comment'] = self.df['comment'].str.strip()
        ...
```

### Full Forward Pass

Images resized to 224×224, ImageNet-normalized. Batch size 64.

---

## Assignment

Built `ImageEncoder`, `TextEncoder`, `CLIPModel`, and `infonce_loss`, then ran a full forward pass on real data to verify shapes and sanity-check the loss.

---

## Key Results

| Check | Value |
|---|---|
| Total samples | 158,915 |
| Batch shape | [64, 3, 224, 224] |
| Image embeddings | [64, 256] |
| Text embeddings | [64, 256] |
| Temperature (init) | 0.0700 |
| InfoNCE Loss (untrained) | 4.1578 |

- All results printed directly from `Notebooks/SOS_midterm_weeks1-4.ipynb`, Cell 4 (dataset instantiation, total samples), Cell 6 (forward pass test, embedding shapes/temperature), and Cell 8 (loss sanity check).
---

## Reflection Questions

**Q1: Why strip column names/whitespace in the dataset class?**
The raw CSV has leading spaces in column names (` comment_number`, ` comment`) and in caption text, left unstripped, this silently breaks downstream lookups.

**Q2: What does the 4.1578 loss value confirm?**
That the loss implementation is correct, an untrained model with random embeddings should be maximally uncertain across 64 candidates, giving loss ≈ log(64).

---

## Conclusion & Next Steps

Week 4 delivered a verified, working pipeline, dataset, encoders, model, and loss-ready for real training.

In **Week 5**, the training loop runs for real, for the first time.
