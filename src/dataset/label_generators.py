from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .dyad_pairer import DyadMetadata

def generate_v1_labels(dyad_id: int, swap_idx: int, n_samples: int) -> np.ndarray:
    labels = np.zeros(n_samples, dtype=int)
    labels[:swap_idx] = dyad_id * 2
    labels[swap_idx:] = dyad_id * 2 + 1
    return labels

def generate_v2_labels(dyad_id: int, n_samples: int) -> np.ndarray:
    return np.full(n_samples, dyad_id, dtype=int)

def generate_v3_labels(swap_idx: int, n_samples: int) -> np.ndarray:
    labels = np.zeros(n_samples, dtype=int)
    labels[swap_idx:] = 1
    return labels

def generate_v4_labels(aq_p1: int, aq_p2: int, n_samples: int) -> np.ndarray:
    return np.full(n_samples, abs(aq_p1 - aq_p2), dtype=int)

def generate_v5_labels(aq_p1: int, aq_p2: int, swap_idx: int, n_samples: int, shift_offset: int = 10) -> np.ndarray:
    labels = np.zeros(n_samples, dtype=int)
    diff_block1 = (aq_p1 - aq_p2) + shift_offset
    diff_block2 = (aq_p2 - aq_p1) + shift_offset
    labels[:swap_idx] = max(0, diff_block1)
    labels[swap_idx:] = max(0, diff_block2)
    return labels

def generate_v6_labels(n_samples: int, num_bins: int = 30) -> np.ndarray:
    bin_size = max(1, n_samples // num_bins)
    labels = np.zeros(n_samples, dtype=int)
    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < num_bins - 1 else n_samples
        labels[start:end] = i
    return labels

def get_labels_for_dyad(meta: "DyadMetadata", n_samples: int, label_version: str, aq_shift_offset: int = 10) -> np.ndarray:
    ver = label_version.lower()
    swap = meta.role_swap_sample_idx if meta.role_swap_sample_idx > 0 else n_samples // 2
    if ver == "v1":
        return generate_v1_labels(meta.dyad_id, swap, n_samples)
    elif ver == "v2":
        return generate_v2_labels(meta.dyad_id, n_samples)
    elif ver == "v3":
        return generate_v3_labels(swap, n_samples)
    elif ver == "v4":
        return generate_v4_labels(meta.aq_p1, meta.aq_p2, n_samples)
    elif ver == "v5":
        return generate_v5_labels(meta.aq_p1, meta.aq_p2, swap, n_samples, aq_shift_offset)
    elif ver == "v6":
        return generate_v6_labels(n_samples, num_bins=30)
    else:
        raise ValueError(f"Unknown label version: {label_version}")
