from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

def normalize_channels(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    z_scored = (x - mean) / (std + eps)
    min_val = np.min(z_scored, axis=0, keepdims=True)
    max_val = np.max(z_scored, axis=0, keepdims=True)
    return (z_scored - min_val) / (max_val - min_val + eps)

def pair_and_normalize_dyad(eeg_p1: np.ndarray, eeg_p2: np.ndarray) -> np.ndarray:
    n_samples = min(len(eeg_p1), len(eeg_p2))
    p1_norm = normalize_channels(eeg_p1[:n_samples])
    p2_norm = normalize_channels(eeg_p2[:n_samples])
    return np.concatenate([p1_norm, p2_norm], axis=1)

@dataclass
class DyadMetadata:
    dyad_id: int
    dyad_name: str
    gender_p1: str
    gender_p2: str
    aq_p1: int
    aq_p2: int
    role_swap_sample_idx: int = -1

class DyadDataset:
    def __init__(self):
        self.dyads: Dict[int, Dict[str, Any]] = {}

    def add_dyad(self, dyad_meta: DyadMetadata, joint_neural_data: np.ndarray) -> None:
        self.dyads[dyad_meta.dyad_id] = {
            "metadata": dyad_meta,
            "neural": joint_neural_data,
        }

    def get_neural_and_labels(self, label_version: str, dyad_ids: Optional[List[int]] = None, aq_shift_offset: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        from .label_generators import get_labels_for_dyad
        target_ids = dyad_ids if dyad_ids is not None else list(self.dyads.keys())
        all_x, all_y = [], []
        for did in target_ids:
            item = self.dyads[did]
            neural = item["neural"]
            meta: DyadMetadata = item["metadata"]
            labels = get_labels_for_dyad(meta, len(neural), label_version, aq_shift_offset)
            all_x.append(neural)
            all_y.append(labels)
        return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)
