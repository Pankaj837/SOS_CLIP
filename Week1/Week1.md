# Week 1 — Foundations of Multimodal ML

## Overview

Week 1 built the conceptual foundation for the whole project: what multimodal learning is, why it matters, and how images and text can be brought into a shared representation space.

---

## Topics Covered

### Core Challenges (Baltrušaitis et al., 2017)

- **Representation** — encoding each modality so it can be compared; solved here via separate encoders projecting into a shared 256-d space
- **Alignment** — matching elements across modalities; this project addresses alignment at the global level (whole image ↔ whole caption)
- **Fusion** — how/when to combine modalities (see below)
- **Translation** — generating one modality from another (out of scope)
- **Co-learning** — one modality helping another learn better representations; contrastive image-text training is a form of this

### Fusion Strategies

| Strategy | When combined | Example |
|---|---|---|
| Early (source) | Raw inputs | Rarely works — pixels/tokens structurally incompatible |
| Late (decision) | Final embeddings only | CLIP — used in this project |
| Intermediate (latent) | Middle layers | ViLBERT-style cross-attention |
| Hybrid | Multiple points | Most expressive, most expensive |

**Why late fusion was chosen:** most interpretable, most efficient at inference (embeddings precomputable), and matches the CLIP architecture being replicated.

---

## Assignment

Studied the five core challenges, the four fusion strategies, and real-world multimodal applications (VQA, medical imaging, autonomous driving) to justify architectural choices made in later weeks.

---

## Key Results

This week : conceptual foundation, every later design decision (dual encoder, late fusion, InfoNCE) traces back to this week's reasoning.

---

## Reflection Questions

**Q1: Why late fusion instead of early or intermediate?**
Late fusion keeps the two encoders fully independent until the final similarity computation, which means image embeddings can be precomputed once and reused for any text query, critical for retrieval at scale. Early fusion is structurally awkward (pixels vs. tokens), and intermediate fusion (explored later in Week 6) trades this efficiency for potentially richer cross-modal interaction.

**Q2: What is co-learning, concretely, in this project?**
Text supervision (captions) helps the image encoder learn richer visual features than image labels alone would provide, because captions carry compositional, relational information a single class label cannot.

---

## Conclusion & Next Steps

Week 1 established why every later architectural choice, dual encoder, late fusion, contrastive training, is principled rather than arbitrary.

In **Week 2**, the focus moves to contrastive learning and the InfoNCE loss that drives training.
