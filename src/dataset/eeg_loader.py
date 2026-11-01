from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import mne

from .dyad_pairer import DyadDataset, DyadMetadata, pair_and_normalize_dyad
from src.preprocessing.pipeline import preprocess_continuous_eeg

def inspect_eeg_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspects header metadata from an EEG/EDF file"""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in ('.edf', '.bdf'):
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    elif suffix == '.fif':
        raw = mne.io.read_raw_fif(path, preload=False, verbose=False)
    elif suffix == '.set':
        raw = mne.io.read_raw_eeglab(path, preload=False, verbose=False)
    else:
        raw = mne.io._read_raw(path, preload=False, verbose=False)

    return {
        'filename': path.name,
        'n_channels': len(raw.ch_names),
        'ch_names': raw.ch_names,
        'sfreq': raw.info['sfreq'],
        'n_samples': raw.n_times,
        'duration_sec': raw.n_times / raw.info['sfreq']
    }

def load_eeg_file(
    file_path: Union[str, Path],
    preprocess: bool = True,
    target_sfreq: float = 250.0
) -> np.ndarray:
    """Loads a single participant's EEG file and returns a (n_samples, n_channels) array."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == '.npy':
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[0] < arr.shape[1] and arr.shape[0] <= 128:
            arr = arr.T
        return arr
    if suffix in ('.edf', '.bdf'):
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif suffix == '.fif':
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    elif suffix == '.set':
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    else:
        raw = mne.io.read_raw(path, preload=True, verbose=False)
    try:
        raw.pick('eeg', verbose=False)
    except Exception:
        pass
    if preprocess:
        raw = preprocess_continuous_eeg(raw, target_sfreq=target_sfreq)
    data = raw.get_data().T
    return data

def load_dyad_from_files(
    p1_file: Union[str, Path],
    p2_file: Union[str, Path],
    metadata: DyadMetadata,
    preprocess: bool = True
) -> Tuple[DyadMetadata, np.ndarray]:
    eeg_p1 = load_eeg_file(p1_file, preprocess=preprocess)
    eeg_p2 = load_eeg_file(p2_file, preprocess=preprocess)
    joint_eeg = pair_and_normalize_dyad(eeg_p1, eeg_p2)
    if metadata.role_swap_sample_idx <= 0:
        metadata.role_swap_sample_idx = len(joint_eeg) // 2
    return metadata, joint_eeg

def load_dataset_from_directory(
    data_dir: Union[str, Path] = "data/raw",
    metadata_csv: Union[str, Path] = "data/metadata/dyads_metadata.csv",
    preprocess: bool = True
) -> DyadDataset:
    data_dir = Path(data_dir)
    df = pd.read_csv(metadata_csv)
    dataset = DyadDataset()
    for _, row in df.iterrows():
        p1_path = data_dir / row['filename_p1']
        p2_path = data_dir / row['filename_p2']
        if not p1_path.exists() or not p2_path.exists():
            print(f"[Warning] Skipping dyad {row['dyad_name']}: file(s) not found in {data_dir} ({row['filename_p1']}, {row['filename_p2']})")
            continue
        meta = DyadMetadata(
            dyad_id=int(row['dyad_id']),
            dyad_name=str(row['dyad_name']),
            gender_p1=str(row['gender_p1']),
            gender_p2=str(row['gender_p2']),
            aq_p1=int(row['aq_p1']),
            aq_p2=int(row['aq_p2']),
            role_swap_sample_idx=int(row.get('role_swap_sample_idx', -1))
        )
        _, joint_neural = load_dyad_from_files(p1_path, p2_path, metadata=meta, preprocess=preprocess)
        dataset.add_dyad(meta, joint_neural)
        print(f"[Loaded] Dyad {meta.dyad_name}: joint shape {joint_neural.shape} (P1: {p1_path.name}, P2: {p2_path.name})")
    return dataset

