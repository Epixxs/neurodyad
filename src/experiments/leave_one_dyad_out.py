from typing import Dict, Any, List, Optional
import numpy as np
from ..dataset.dyad_pairer import DyadDataset
from ..evaluation.metrics import evaluate_embedding_quality
from ..models.cebra_wrapper import DyadicCEBRAModel

class LeaveOneDyadOutEvaluator:
    def __init__(self, dataset: DyadDataset, cebra_config: Optional[Dict[str, Any]] = None, label_version: str = "v3", num_seeds: int = 5):
        self.dataset = dataset
        self.cebra_config = cebra_config or {}
        self.label_version = label_version
        self.num_seeds = num_seeds

    def run(self) -> List[Dict[str, Any]]:
        dyad_ids = list(self.dataset.dyads.keys())
        if len(dyad_ids) < 2:
            raise ValueError(
                f"Leave-One-Dyad-Out (LODO) cross-validation requires at least 2 dyads (N >= 2) to train on N-1 and evaluate on 1. "
                f"Currently, the dataset only contains {len(dyad_ids)} dyad(s): {dyad_ids}."
            )
        results = []
        total_dyads = len(dyad_ids)
        for idx, held_out_id in enumerate(dyad_ids):
            train_ids = [d for d in dyad_ids if d != held_out_id]
            held_out_item = self.dataset.dyads[held_out_id]
            meta = held_out_item["metadata"]
            gender_group = "FF" if (meta.gender_p1 == "F" and meta.gender_p2 == "F") else "Mixed"
            print(f"\n[LODO {idx+1}/{total_dyads}] Holding out {meta.dyad_name} ({gender_group}) | Training on {len(train_ids)} dyad(s)...")
            X_train, y_train = self.dataset.get_neural_and_labels(self.label_version, train_ids)
            X_test, y_test = self.dataset.get_neural_and_labels(self.label_version, [held_out_id])

            seed_metrics = []
            for seed in range(self.num_seeds):
                print(f"  -> Seed {seed+1}/{self.num_seeds} fitting...")
                cfg = dict(self.cebra_config)
                cfg["seed"] = seed
                model = DyadicCEBRAModel(cfg)
                model.fit(X_train, y_train)
                emb_train = model.transform(X_train)
                emb_test = model.transform(X_test)
                metrics = evaluate_embedding_quality(emb_train, y_train, emb_test, y_test)
                seed_metrics.append(metrics)

            gof_vals = [m["goodness_of_fit"] for m in seed_metrics]
            knn_vals = [m["knn_accuracy"] for m in seed_metrics]
            results.append({
                "held_out_dyad_id": held_out_id,
                "dyad_name": meta.dyad_name,
                "gender_group": gender_group,
                "eval_samples": len(y_test),
                "gof_mean": float(np.mean(gof_vals)),
                "gof_std": float(np.std(gof_vals)),
                "knn_mean": float(np.mean(knn_vals)),
                "knn_std": float(np.std(knn_vals)),
            })
        return results
