# Architecture and Engineering Decisions: Neurodyad

This document details the architectural, neurocomputational, and engineering decisions adopted in the **Neurodyad** project—a reproduction of *"Dyadic Neural Dynamics: Extending Representation Learning to Social Neuroscience"* (arXiv:2509.23479, 2025). Each section explains the selected approach, the scientific and technical rationale, alternatives considered, and the associated trade-offs.

---

## Table of Contents
1. [Dyadic Input Representation & Channel Concatenation](#1-dyadic-input-representation--channel-concatenation)
2. [Continuous EEG Preprocessing Pipeline](#2-continuous-eeg-preprocessing-pipeline)
3. [Synchronous Dyadic Artifact Masking & Pseudocut Injection](#3-synchronous-dyadic-artifact-masking--pseudocut-injection)
4. [Two-Stage Channel-Wise Normalization](#4-two-stage-channel-wise-normalization)
5. [CEBRA Model Architecture & Hyperparameters](#5-cebra-model-architecture--hyperparameters)
6. [Behavioral Label Taxonomy (V1–V6)](#6-behavioral-label-taxonomy-v1v6)
7. [Validation Strategy: Leave-One-Dyad-Out (LODO / N+1)](#7-validation-strategy-leave-one-dyad-out-lodo--n1)
8. [Scientific Sanity Controls & Functional Channel Ablations](#8-scientific-sanity-controls--functional-channel-ablations)
9. [Hybrid Data Ingestion & Execution Resilience](#9-hybrid-data-ingestion--execution-resilience)

---

## 1. Dyadic Input Representation & Channel Concatenation

### The Decision
Instead of computing pairwise cross-subject connectivity metrics (such as Phase Locking Value, Coherence, or Mutual Information) prior to feature learning, we concatenate continuous multichannel EEG signals from both interacting participants along the feature (channel) dimension:
$$X_{\text{joint}} = [X^{(1)}, X^{(2)}] \in \mathbb{R}^{T \times 2C}$$
where $T$ is the number of temporally synchronized timepoints, and $C$ is the number of EEG channels per participant ($C = 64$ or $65$, yielding $2C = 128$ or $130$).

### Rationale
- **End-to-End Inter-Brain Representation Learning**: Precomputed connectivity metrics collapse millisecond-level temporal phase dynamics into windowed scalar summaries. By supplying raw, synchronized voltage traces directly to CEBRA's 1D convolutional temporal encoders, the network learns non-linear inter-brain temporal receptive fields without handcrafted feature bias.
- **Symmetric Dyadic Structure**: Stacking the participants side-by-side preserves the spatial topology of both heads simultaneously across all time offsets.

### Alternatives Considered & Trade-Offs
| Approach | Pros | Cons / Trade-Offs | Decision |
| :--- | :--- | :--- | :--- |
| **Concatenated Channel Matrix ($T \times 2C$)** *(Adopted)* | Preserves temporal resolution; enables end-to-end learning of inter-brain dynamics via 1D convolutions. | Doubled channel dimension increases parameter count in the first projection layer. | **Selected** |
| **Dual-Stream Siamese / Cross-Attention Networks** | Explicitly separates individual encodings before fusion. | Diverges from CEBRA's core architecture; requires substantial custom network surgery without empirical evidence of superior dyadic latent geometry. | Rejected |
| **Precomputed Functional Connectivity (PLV / Coherence)** | Interpretable, lower-dimensional representation. | Discards raw temporal fine structure and non-linear interactions; computationally heavy windowing preprocessing. | Rejected |

---

## 2. Continuous EEG Preprocessing Pipeline

### The Decision
The raw EEG data undergoes a standardized continuous preprocessing sequence:
1. **Bandpass Filtering**: Zero-phase FIR filter between $0.1\,\text{Hz}$ (highpass, removes DC drift/slow sweating artifacts) and $45.0\,\text{Hz}$ (lowpass, removes $50/60\,\text{Hz}$ AC line noise and high-frequency muscle twitch).
2. **Resampling to $250\,\text{Hz}$**: Standardizes recordings across disparate acquisition setups.
3. **Bad Channel Interpolation**: Spherical spline interpolation based on standard montage locations.
4. **Extended-Infomax Independent Component Analysis (ICA)**: Identifies and subtracts ocular (blinks, saccades) and cardiac artifacts without disrupting cortical signal timing.

### Rationale
- **Nyquist-Preserving Downsampling**: At $250\,\text{Hz}$, the Nyquist frequency is $125\,\text{Hz}$, comfortably covering theta ($4\text{--}8\,\text{Hz}$), alpha ($8\text{--}12\,\text{Hz}$), beta ($13\text{--}30\,\text{Hz}$), and lower gamma ($30\text{--}45\,\text{Hz}$) bands where inter-brain neural coupling primarily occurs. Downsampling from $1000\,\text{Hz}$ reduces batch memory footprint by $4\times$, allowing longer temporal batches during contrastive optimization.
- **Extended-Infomax ICA**: EEG artifacts exhibit sub-Gaussian distributions (e.g., periodic cardiac beats) alongside super-Gaussian distributions (blinks). Extended-Infomax detects both forms, unlike standard Infomax or FastICA.

### Trade-Offs
- Filtering down to $45\,\text{Hz}$ discards high-gamma ($>50\,\text{Hz}$) activity, which has high spatial specificity but is notoriously prone to electromyographic (EMG) contamination during naturalistic conversational speech.

---

## 3. Synchronous Dyadic Artifact Masking & Pseudocut Injection

### The Decision
- **Synchronous Masking**: If either participant in a dyad produces a contaminated epoch (e.g., gross head movement or detachment), the time interval is excised from **both** participants synchronously.
- **Pseudocut Injection Control**: To confirm that CEBRA does not artificially exploit segment excision boundaries as contrastive shortcuts, random $50\,\text{ms}$ cuts are artificially injected during validation tests.

### Rationale
- In hyperscanning, inter-brain coherence relies on temporal phase alignment. If one participant's timeline is trimmed asynchronously, relative time stamps desynchronize, destroying true dyadic co-fluctuations.
- Contrastive models can suffer from "shortcut learning"—if discontinuous edge boundaries are introduced by artifact removal, the temporal encoder might cluster artifacts rather than neural dynamics. Pseudocut injection proves that learned latents reflect genuine conversational dynamics, not cut edges.

### Trade-Offs
- Synchronous masking is conservative: noise in Participant 1 discards valid brain activity in Participant 2 for that window, reducing total usable dataset duration.

---

## 4. Two-Stage Channel-Wise Normalization

### The Decision
Each participant's signal is normalized independently before dyadic concatenation using a two-stage transformation:
1. **Channel-wise Z-score**: Center to zero mean and scale to unit variance:
   $$z_{t, c} = \frac{x_{t, c} - \mu_c}{\sigma_c + \epsilon}$$
2. **Channel-wise Min-Max Scaling to $[0, 1]$**:
   $$\tilde{x}_{t, c} = \frac{z_{t, c} - \min(z_c)}{\max(z_c) - \min(z_c) + \epsilon}$$

### Rationale
- **Cross-Subject Impedance Disparity**: Absolute raw microvolt values differ drastically across participants due to skull thickness, hair density, and electrode impedance. Unnormalized concatenation would allow the participant with higher signal amplitude to dominate the contrastive loss gradients.
- **Stability of Contrastive InfoNCE Loss**: CEBRA's contrastive temperature scaling operates optimally when input feature amplitudes are bounded and uniform across dimensions.

### Trade-Offs
- Normalization discards absolute amplitude differences across frequency bands, retaining only relative temporal dynamics and topographical patterns.

---

## 5. CEBRA Model Architecture & Hyperparameters

### The Decision
We adopt the paper's exact architecture and training configuration:
- **Architecture**: `offset10-model` (1D multi-layer convolutional neural network).
- **Time Offsets**: `time_offsets = 10` ($10$ backward/forward context steps).
- **Latent Dimension**: $d = 3$ (output mapped onto the 3D unit sphere $S^2$).
- **Metric**: Cosine distance ($D(u, v) = 1 - \frac{u \cdot v}{\|u\|_2 \|v\|_2}$).
- **Supervision Mode**: `conditional = "time_delta"` (joint temporal and behavioral contrastive learning).
- **Optimization**: Adam optimizer, $\text{learning\_rate} = 3 \times 10^{-4}$, batch size $= 512$, iterations $= 5000$.
- **Temperature Schedule**: Auto-temperature schedule starting at $1.0$ and decaying to $0.1$.

### Rationale
- **Temporal Receptive Field**: At $250\,\text{Hz}$, $10$ time offsets span $40\,\text{ms}$, matching the typical synaptic transmission latency and conversational turn-taking micro-dynamics.
- **3D Spherical Topology ($S^2$)**: High-dimensional embeddings ($d \ge 32$) risk overfitting dyadic idiosyncrasies. A constrained 3D spherical space provides intrinsic regularization, enforces continuous trajectory topology, and enables direct visualization without secondary dimensionality reduction (such as t-SNE or UMAP).
- **Cosine Distance on $S^2$**: Contrastive InfoNCE with cosine distance concentrates representations on the relative angular displacement of states, preventing latent norm explosion.

### Trade-Offs
- A 3D latent space has lower representation capacity than high-dimensional vectors, requiring informative supervision labels to prevent manifold collapse.

---

## 6. Behavioral Label Taxonomy (V1–V6)

### The Decision
Six distinct label regimes are implemented to isolate various facets of dyadic interaction:

| Version | Target Construct | Mathematical Formulation | Scientific Purpose |
| :--- | :--- | :--- | :--- |
| **V1** | Unique Interaction Block | $y_t = 2 \cdot \text{dyad\_id} + [t \ge t_{\text{swap}}]$ | Joint representation of dyad identity and role state. |
| **V2** | Dyad Identification | $y_t = \text{dyad\_id}$ | Tests inter-individual separation vs. common dynamics. |
| **V3** | Conversation Turn Role | $y_t = 0 \text{ (Speaker)} \text{ vs. } 1 \text{ (Listener)}$ | Decodes behavioral roles across arbitrary dyads. |
| **V4** | Absolute AQ Score Disparity | $y_t = |\text{AQ}_1 - \text{AQ}_2|$ | Measures social-cognitive divergence between partners. |
| **V5** | Directional / Signed AQ Offset | $y_t = (\text{AQ}_{\text{speaker}} - \text{AQ}_{\text{listener}}) + \text{offset}$ | Tests asymmetrical social communication dynamics. |
| **V6** | Temporal Partition Sanity Check | $y_t = \lfloor t / (T / 30) \rfloor$ | Control: tests pure chronological time drift encoding. |

### Rationale
- Discretizing continuous behavioral traits (such as Autism-Spectrum Quotient score differences in V4/V5) into positive integer categories allows CEBRA to condition contrastive pairs via discrete categorical distributions while avoiding negative label indexing.
- Implementing V1 through V6 ensures full ablation of behavioral hypotheses: from dyad-specific memory (V1/V2) to generalized social role mechanics (V3) and psychological compatibility (V4/V5).

### Trade-Offs
- Binning continuous AQ scores into discrete categories can lose fine-grained ordinal gradient information.

---

## 7. Validation Strategy: Leave-One-Dyad-Out (LODO / N+1)

### The Decision
Model performance is benchmarked using an $(N-1)$ train / $1$ held-out dyad cross-validation scheme repeated across 5 random seeds:
- **Evaluation Metrics**:
  - **k-Nearest Neighbor (5-NN) Accuracy**: Cosine distance decoding of held-out behavioral states.
  - **Goodness-of-Fit ($R^2$)**: Linear Ridge regression predicting continuous behavioral trajectories from latent coordinates.
  - **Embedding Consistency**: Cross-run alignment matrix comparing latent spaces across seeds.

### Rationale
- Traditional $k$-fold cross-validation randomly splits temporal segments across the same recording session, causing massive temporal leakage (autocorrelation). LODO completely isolates individual dyads, testing whether the learned manifold generalizes to unseen human pairs.
- Multi-seed averaging ensures that reported metrics reflect stable neural manifolds rather than favorable random weight initializations.

---

## 8. Scientific Sanity Controls & Functional Channel Ablations

### The Decision
1. **Permutation Controls**:
   - Neural Shuffle: Scrambles the temporal order of EEG channels ($X_{\text{shuffled}}$).
   - Behavioral Shuffle: Permutes label assignments ($y_{\text{shuffled}}$).
2. **Channel Ablations**:
   - Speech Comprehension Network: Ablates bilateral temporal electrodes (T7, T8, TP7, TP8, CP5, CP6).
   - Executive Function Network: Ablates prefrontal and frontal electrodes (Fp1, Fp2, AF3, AF4, Fz).

### Rationale
- Neural and behavioral permutation controls provide empirical chance baselines. A valid model must drop to chance level ($50\%$ for binary V3, near-zero $R^2$) under shuffled inputs.
- Region-specific channel ablation determines whether decoding accuracy is driven by anatomically relevant brain circuits or diffuse artifacts (e.g. eye muscle tension).

---

## 9. Hybrid Data Ingestion & Execution Resilience

### The Decision
`eeg_loader.py` and `run_experiment.py` support:
1. Multi-format ingestion (`.edf`, `.bdf`, `.fif`, `.set`, `.npy`).
2. Automatic metadata registration via CSV.
3. Fallback to synthetic multi-dyad simulation (`--synthetic`) when fewer than 2 dyads are locally present.

### Rationale
- Real hyperscanning datasets are often multi-gigabyte or subject to data access agreements. The hybrid pipeline ensures that the entire test suite, LODO cross-validation, and visualization routines can be tested and verified locally without blocking development.

