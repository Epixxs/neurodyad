# Dyadic Neural Dynamics — Reproduction

Reproduction of **"Dyadic Neural Dynamics: Extending Representation Learning to Social Neuroscience"** (arXiv:2509.23479, 2025), which applies **CEBRA** (Contrastive Embedding for Behavioral and Neural Analysis) to dyadic EEG hyperscanning data to learn joint neural representations of social interaction (speaker/listener roles, autism-quotient differences between partners).

## Status

**Pipeline Implemented, Model Trained & LODO Validation Verified.**

- **Environment & Hardware**: Configured and verified with Python 3.11, PyTorch (CUDA-enabled with NVIDIA GeForce RTX 3050 Laptop GPU), CEBRA 0.6.0, MNE 1.11.0, and scikit-learn.
- **Core Pipeline Implementation**:
  - **Preprocessing**: Bandpass filtering (0.1–45 Hz), resampling (250 Hz), bad channel interpolation, synchronous dyadic artifact masking, and Extended-Infomax ICA.
  - **Dyadic Pairing & Normalization**: Paired continuous multichannel EEG concatenated along channel dimension with channel-wise z-score standardization and min–max scaling.
  - **Label Schemes (V1–V6)**: Implemented generators for unique interaction blocks (V1), dyad identification (V2), conversation direction / turn-swap (V3), absolute AQ score differences (V4), signed directional AQ differences (V5), and partition sanity checks (V6).
  - **CEBRA Architecture**: Configured with the paper's `offset10-model`, 3D output embedding, cosine distance metric, auto-temperature schedule, and time-delta / behavioral supervision.
  - **Validation & Control Suite**: Neural and behavioral label permutation shuffles, 50 ms pseudocut injection, channel ablation (speech comprehension vs. executive function), and leave-one-dyad-out (N+1) cross-validation.
  - **Evaluation & Visualizations**: 5-NN decoder accuracy, Goodness-of-Fit ($R^2$), cross-run embedding consistency matrices, InfoNCE loss tracking, and 3D spherical scatter plots.
- **Data Ingestion Status**:
  - Continuous dyadic EEG recordings (`Speaker.edf` and `Listener.edf`) placed in `data/raw/` and configured in `data/metadata/dyads_metadata.csv`.
  - Signal integrity verified: 65 channels per participant, 250.0 Hz sampling rate, 75,750 aligned samples (~5.05 minutes).
  - Joint dyadic feature matrix verified and loaded: $X \in \mathbb{R}^{75750 \times 130}$ with matching label vectors.
- **Current Progress & Executed Experiments**:
  - **Model Training**: Successfully trained CEBRA model on the paired dyadic EEG dataset using the V3 label scheme (Speaker vs. Listener). Output embeddings, labels, and 3D latent space visualization (`outputs/latent_space_v3.png`) generated.
  - **LODO (N+1) Cross-Validation**: Successfully executed Leave-One-Dyad-Out cross-validation using the synthetic multi-dyad benchmark (8 mock dyads) across 5 random seeds, verifying the train/test split, 5-NN decoding accuracy, and Ridge regression Goodness-of-Fit ($R^2$) scoring pipeline.
  - **Comprehensive Documentation**: Detailed design decisions and trade-offs documented in [`decision.md`](decision.md); execution flow and lifecycle tracing documented in [`flow.md`](flow.md).

## Project Structure

```
neurodyad/
├── configs/
│   └── cebra_default.yaml          # CEBRA & pipeline hyperparameters
├── data/
│   ├── metadata/
│   │   ├── dyads_metadata.csv      # Active dyad session metadata
│   │   └── dyads_template.csv      # Metadata template
│   └── raw/
│       ├── Speaker.edf             # Participant 1 recording (65 ch, 250 Hz)
│       └── Listener.edf            # Participant 2 recording (65 ch, 250 Hz)
├── outputs/
│   ├── embeddings.npy              # Learned 3D CEBRA latent embeddings
│   ├── labels.npy                  # Aligned behavioral labels
│   └── latent_space_v3.png         # 3D spherical latent space visualization
├── scripts/
│   └── inspect_edf.py              # Header & signal inspection utility
├── src/
│   ├── preprocessing/
│   │   ├── pipeline.py             # Filtering, resampling, bad channel interpolation
│   │   ├── synchronous_mask.py     # Synchronous artifact masking & pseudocut injector
│   │   └── ica_denoise.py          # Extended-Infomax ICA
│   ├── dataset/
│   │   ├── dyad_pairer.py          # Channel concatenation & normalization
│   │   ├── eeg_loader.py           # Multi-format raw loader & metadata mapper
│   │   └── label_generators.py     # V1–V6 behavioral label generators
│   ├── models/
│   │   └── cebra_wrapper.py        # CEBRA model wrapper with paper hyperparameters
│   ├── experiments/
│   │   ├── controls.py             # Neural & behavioral shuffle controls
│   │   ├── ablations.py            # Speech/comprehension & executive channel ablations
│   │   └── leave_one_dyad_out.py   # N+1 cross-validation engine
│   └── evaluation/
│       ├── metrics.py              # 5-NN accuracy, Goodness-of-Fit (R^2), consistency score
│       └── visualizer.py           # 3D latent scatter, consistency heatmaps, loss curves
├── decision.md                     # Architectural decisions, rationale & trade-offs
├── flow.md                         # Code execution flow & lifecycle documentation
├── run_experiment.py               # CLI experiment runner
├── requirements.txt                # Python dependencies
└── README.md
```

## Documentation

- **[`decision.md`](decision.md)**: Detailed breakdown of architectural and neuroscience decisions (channel concatenation, 2-stage normalization, CEBRA hyperparameter choices, V1–V6 label formulations, and control strategies) along with trade-offs.
- **[`flow.md`](flow.md)**: Comprehensive trace of how execution moves through the codebase, from entry points to preprocessing, pairing, model modes, LODO loops, and tensor shape transformations.

## Available Commands

### Inspect Raw EEG Recordings
```bash
python scripts/inspect_edf.py
```

### Verify Data Loading & Pairing
```bash
python run_experiment.py --config configs/cebra_default.yaml --label_version v3
```

### Run Sanity Controls (Neural & Label Permutation)
```bash
python run_experiment.py --run_controls
```

### Train CEBRA Model
```bash
python run_experiment.py --train
```
*Fits the CEBRA model on the paired dyadic EEG dataset and exports embeddings and 3D latent scatter plot to `outputs/`.*

### Run Leave-One-Dyad-Out (N+1) Generalization
```bash
# Run with synthetic multi-dyad test data (8 mock dyads, evaluated across random seeds)
python run_experiment.py --run_lodo --synthetic

# With real multi-dyad recordings (requires >= 2 dyads in data/metadata/dyads_metadata.csv)
python run_experiment.py --run_lodo
```

## Reference

Glushanina, M., Huang, J., McCleod, M., Ames, B., & Malaia, E. A. (2025). *Dyadic Neural Dynamics: Extending Representation Learning to Social Neuroscience.* arXiv:2509.23479.
