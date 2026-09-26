# Dyadic Neural Dynamics — Reproduction

Reproduction of **"Dyadic Neural Dynamics: Extending Representation Learning
to Social Neuroscience"** (arXiv:2509.23479, 2025), which applies **CEBRA** (Contrastive Embedding for Behavioral and
neural Analysis) to dyadic EEG hyperscanning data to learn joint neural
representations of social interaction (speaker/listener roles, autism-quotient
differences between partners).

## Status

**Early scaffolding.** Repo structure is initialized; preprocessing script
for paired speaker/listener EDF recordings is written but not yet run
end-to-end on real data. CEBRA training, label generation, and control/eval
pipelines are not yet implemented.

## Goal

Reproduce the paper's core pipeline:

1. Preprocess paired EEG hyperscanning recordings (speaker + listener)
2. Generate the six behavioral label schemes (V1–V6) described in the paper
3. Train CEBRA to learn 3-D joint embeddings of dyadic neural activity
4. Validate with the paper's control checks (neural shuffle, behavioral
   shuffle, pseudocut injection, leave-one-dyad-out cross-validation,
   channel ablation)
5. Reproduce the reported metrics (goodness-of-fit, KNN-5 decoding accuracy,
   InfoNCE loss, cross-run consistency scores)

## Reference

Glushanina, M., Huang, J., McCleod, M., Ames, B., & Malaia, E. A. (2025).
*Dyadic Neural Dynamics: Extending Representation Learning to Social
Neuroscience.* arXiv:2509.23479.
