## SOS_CLIP
## CS38: Multimodal Machine Learning

This repository documents my week-by-week progress on the CS38 Summer of Science project on Multimodal Machine Learning. It contains concise weekly reports and solved notebooks for Weeks 1–8, along with key results and findings.

### Project Goal

Build a solid, scalable understanding of Multimodal Machine Learning by implementing a complete, modular CLIP-inspired image-text contrastive learning pipeline trained on Flickr30k, progressing from foundational theory to a fully evaluated dual-encoder retrieval system with hard-negative mining, fusion-strategy comparison, and backbone-freezing ablations.

### Repo Layout
```
CS38-Multimodal-ML
├── README.md
├── Week1
│   └── Week1.md
├── Week2
│   └── Week2.md
├── Week3
│   └── Week3.md
├── Week4
│   └── Week4.md
├── Week5
│   └── Week5.md
├── Week6
│   └── Week6.md
├── Week7
│   └── Week7.md
├── Week8
│    └── Week8.md
├── notebooks
    ├── SOS_midterm_weeks1-4.ipynb
    └── SOS_endterm_weeks5-8.ipynb
    
```
Each `WeekX.md` contains:

- Short overview of the week
- Topics covered and why they matter
- Problems/assignments solved
- Key results and figures
- A pointer to the notebook (and relevant cells) implementing that week's work
---
- There are two notebooks : `SOS_midterm_weeks1-4.ipynb` implements and verifies the full pipeline (dataset, encoders, model, InfoNCE loss) covered in Weeks 1–4, and `SOS_endterm_weeks5-8.ipynb` implements the training, evaluation, and all ablations covered in Weeks 5–8. Both are self-contained and can be run independently.**
---
### What to Expect

**Week 1 — Foundations of Multimodal ML**
- Core challenges: representation, alignment, fusion, translation, co-learning
- Fusion strategies: early, late, intermediate, hybrid
- Why late fusion (CLIP-style) was chosen for this project

**Week 2 — Contrastive Learning and Embeddings**
- InfoNCE loss, symmetric cross-entropy, learnable temperature
- Hard negatives: why random negatives stop being useful
- Sanity check: random-model loss ≈ log(batch_size)

**Week 3 — Architecture Deep Dive and Related Work**
- Dual encoder: ResNet18 + DistilBERT with projection heads
- Recall@K as the retrieval metric
- Survey: CLIP, ALIGN, BLIP, Flamingo

**Week 4 — Pipeline Implementation**
- Flickr30k dataset (158,915 image-caption pairs)
- Full CLIPModel forward pass verified: loss 4.1578 ≈ log(64)

**Week 5 — Training Loop and Evaluation**
- 5-epoch standard training run, checkpointing, LR scheduling
- Recall@K evaluation, t-SNE visualization

**Week 6 — Hard Negative Mining and Fusion Comparison**
- Hard-negative InfoNCE, warm-started from the fully-trained standard model
- Intermediate (cross-attention) fusion variant trained and compared against late fusion

**Week 7 — Error Analysis and Improvements**
- Failure case analysis, temperature study, frozen-vs-fine-tuned backbone comparison
- Qualitative retriever: top-5 images for a text query

**Week 8 — Final Model, Report and Documentation**
- Final consolidated results across all 4 model variants
- Clean, modular, reproducible codebase

---

### Dependencies
- torch
- torchvision
- transformers
- pandas
- numpy
- matplotlib
- scikit-learn
- tqdm

---

###  How to Review the Work

1. Open `WeekX/WeekX.md` to read the weekly summary and findings.
2. Open the relevant notebook in `notebooks/` — `SOS_midterm_weeks1-4.ipynb` for Weeks 1–4, `SOS_endterm_weeks5-8.ipynb` for Weeks 5–8 — for the code implementing that week's work. Each `WeekX.md` points to the specific cells covering it.


---

### Status

This repository contains completed work for Weeks 1–8 of the CS38 project.
Submitted as End-Term Report.
