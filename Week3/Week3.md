# Week 3 — Architecture Deep Dive and Related Work

## Overview

Week 3 covered the dual-encoder architecture, projection heads, and Recall@K - the standard retrieval metric, plus a survey of the wider multimodal landscape.

---

## Topics Covered

### Dual Encoder

Two independent networks, one per modality, sharing no weights:

- **ImageEncoder:** ResNet18 (pretrained, classification head removed) → 512-d features → Linear(512, 256)
- **TextEncoder:** DistilBERT → CLS token (768-d) → Linear(768, 256)

Both outputs L2-normalized so dot product = cosine similarity.

### Recall@K

Given a query, what fraction of the time does the correct match appear in the top K results? Computed in both directions (image→text, text→image) at K = 1, 5, 10.

### Related Work

CLIP (Radford et al., 2021) — the central reference. Also surveyed: ALIGN (1.8B noisy pairs), BLIP (contrastive + generative), Flamingo (in-context multimodal LM).

---

## Assignment

Studied why dual encoders (vs. cross-encoders) enable efficient large-scale retrieval, and why Recall@K is the correct metric for a ranking problem rather than a classification one.

---

## Key Results

Conceptual week: architecture decisions validated in Week 4's implementation.

---

## Reflection Questions

**Q1: Why a dual encoder instead of a cross-encoder?**
A cross-encoder must be run fresh for every query-document pair, making large-scale retrieval too slow. A dual encoder lets you precompute all image embeddings once and only run the (cheap) text encoder per query.

**Q2: Why L2-normalize before computing similarity?**
So dot product equals cosine similarity, and all embeddings lie on a unit hypersphere, this is required for InfoNCE to behave correctly.

---

## Conclusion & Next Steps

Every line of Week 4's code has a reason that traces back to this week's architectural reasoning.

In **Week 4**, the pipeline gets implemented and verified end-to-end.
