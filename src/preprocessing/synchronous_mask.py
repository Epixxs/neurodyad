from typing import Tuple, List
import numpy as np

def apply_synchronous_artifact_mask(eeg_p1: np.ndarray, eeg_p2: np.ndarray, bad_intervals: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = min(len(eeg_p1), len(eeg_p2))
    keep_mask = np.ones(n_samples, dtype=bool)
    for start, end in bad_intervals:
        keep_mask[max(0, start):min(n_samples, end)] = False
    return eeg_p1[:n_samples][keep_mask], eeg_p2[:n_samples][keep_mask]

def inject_pseudocuts(eeg_p1: np.ndarray, eeg_p2: np.ndarray, sfreq: float = 250.0, cut_duration_ms: float = 50.0, num_extra_cuts: int = 5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cut_samples = int(cut_duration_ms * sfreq / 1000.0)
    n_samples = min(len(eeg_p1), len(eeg_p2))
    cut_starts = rng.integers(0, n_samples - cut_samples, size=num_extra_cuts)
    cut_mask = np.ones(n_samples, dtype=bool)
    for start in cut_starts:
        cut_mask[start:start + cut_samples] = False
    return eeg_p1[:n_samples][cut_mask], eeg_p2[:n_samples][cut_mask]
