import argparse
from pathlib import Path
import yaml
import numpy as np

from src.dataset.dyad_pairer import DyadDataset, DyadMetadata, pair_and_normalize_dyad
from src.dataset.eeg_loader import load_dataset_from_directory
from src.models.cebra_wrapper import DyadicCEBRAModel
from src.evaluation.metrics import evaluate_embedding_quality
from src.evaluation.visualizer import plot_3d_latent_space
from src.experiments.controls import shuffle_neural_data, shuffle_behavioral_labels
from src.experiments.leave_one_dyad_out import LeaveOneDyadOutEvaluator

def build_synthetic_dyad_dataset(n_dyads: int = 8, n_samples: int = 1000, n_channels: int = 64) -> DyadDataset:
    dataset = DyadDataset()
    rng = np.random.default_rng(42)
    for i in range(n_dyads):
        p1 = rng.standard_normal((n_samples, n_channels))
        p2 = rng.standard_normal((n_samples, n_channels))
        joint_neural = pair_and_normalize_dyad(p1, p2)
        meta = DyadMetadata(
            dyad_id=i,
            dyad_name=f"dyad_{i:02d}",
            gender_p1="F" if i % 2 == 0 else "M",
            gender_p2="F" if i < 4 else "M",
            aq_p1=int(rng.integers(1, 10)),
            aq_p2=int(rng.integers(1, 10)),
            role_swap_sample_idx=n_samples // 2,
        )
        dataset.add_dyad(meta, joint_neural)
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Neurodyad CEBRA Representation Learning")
    parser.add_argument("--config", type=str, default="configs/cebra_default.yaml")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--metadata_csv", type=str, default="data/metadata/dyads_metadata.csv")
    parser.add_argument("--label_version", type=str, default="v3", choices=["v1", "v2", "v3", "v4", "v5", "v6"])
    parser.add_argument("--run_controls", action="store_true")
    parser.add_argument("--run_lodo", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Force using synthetic mock dataset (8 dyads) for testing/dry-runs")
    parser.add_argument("--num_seeds", type=int, default=None, help="Number of seeds for LODO evaluation (overrides config)")
    parser.add_argument("--max_iterations", type=int, default=None, help="Max iterations for CEBRA training (overrides config)")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f) or {}

    model_config = config.get("model", {})
    if args.max_iterations is not None:
        model_config["max_iterations"] = args.max_iterations

    print(f"Loaded config: {args.config}")
    print(f"Label scheme: {args.label_version}")

    data_dir = Path(args.data_dir)
    metadata_csv = Path(args.metadata_csv)
    dataset = DyadDataset()
    if not args.synthetic and data_dir.exists() and metadata_csv.exists():
        print(f"Scanning for real recordings in {data_dir} using {metadata_csv}...")
        dataset = load_dataset_from_directory(data_dir, metadata_csv, preprocess=True)

    if len(dataset.dyads) == 0:
        if args.synthetic:
            print("[Notice] Using synthetic mock dataset (8 dyads) as requested by --synthetic.")
        else:
            print("[Notice] Real files not detected in data/raw. Using synthetic mock dataset for testing.")
        dataset = build_synthetic_dyad_dataset()

    X, y = dataset.get_neural_and_labels(label_version=args.label_version)
    print(f"Dataset ready: X={X.shape}, y={y.shape}")

    if args.run_lodo:
        print("\n--- Running Leave-One-Dyad-Out (N+1) Generalization Analysis ---")
        if len(dataset.dyads) < 2:
            print(f"[Notice] Leave-One-Dyad-Out (LODO) cross-validation requires at least 2 dyads (N >= 2) to train on N-1 dyads and test on the held-out dyad.")
            print(f"Currently, only {len(dataset.dyads)} dyad ({list(dataset.dyads.keys())}) is registered in '{metadata_csv}'.")
            print("\nOptions:")
            print("  1. To test the LODO pipeline with synthetic multi-dyad data:")
            print("     python run_experiment.py --run_lodo --synthetic")
            print("  2. To run with real recordings, register additional dyads in:")
            print(f"     {metadata_csv}")
            return
        num_seeds = args.num_seeds if args.num_seeds is not None else config.get("evaluation", {}).get("num_seeds", 5)
        lodo = LeaveOneDyadOutEvaluator(
            dataset=dataset,
            cebra_config=model_config,
            label_version=args.label_version,
            num_seeds=num_seeds,
        )
        results = lodo.run()
        print("\n--- LODO Generalization Summary ---")
        for r in results:
            print(f"Dyad {r['dyad_name']} ({r['gender_group']}): GOF={r['gof_mean']:.4f}+/-{r['gof_std']:.4f}, KNN={r['knn_mean']:.4f}+/-{r['knn_std']:.4f}")
        return

    if args.run_controls:
        print("\n--- Running Sanity Checks ---")
        X_shuffled = shuffle_neural_data(X)
        y_shuffled = shuffle_behavioral_labels(y)
        print(f"Neural shuffle shape: {X_shuffled.shape}")
        print(f"Label shuffle shape: {y_shuffled.shape}")
        return

    if args.train:
        print("\n--- Training CEBRA Model ---")
        cebra_model = DyadicCEBRAModel(model_config)
        embeddings = cebra_model.fit_transform(X, y)
        print(f"Learned 3D embeddings shape: {embeddings.shape}")
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / "embeddings.npy", embeddings)
        np.save(out_dir / "labels.npy", y)
        plot_3d_latent_space(embeddings, y, title=f"CEBRA Latent Space ({args.label_version.upper()})", save_path=str(out_dir / f"latent_space_{args.label_version}.png"))
        print(f"Saved 3D latent space plot to {out_dir / f'latent_space_{args.label_version}.png'}")
        return

    print("\n--- Pipeline Verified ---")
    print(f"Ready for training with CEBRA model architecture: {config.get('model', {}).get('model_architecture')}")

if __name__ == "__main__":
    main()
