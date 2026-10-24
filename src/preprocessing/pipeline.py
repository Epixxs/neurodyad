from typing import Optional, List
import mne

def preprocess_continuous_eeg(raw: mne.io.BaseRaw, l_freq: float = 0.1, h_freq: float = 45.0, target_sfreq: float = 250.0, bad_channels: Optional[List[str]] = None) -> mne.io.BaseRaw:
    raw_proc = raw.copy().load_data()
    raw_proc.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    if raw_proc.info["sfreq"] != target_sfreq:
        raw_proc.resample(sfreq=target_sfreq, npad="auto")
    if bad_channels:
        raw_proc.info["bads"].extend(bad_channels)
    if len(raw_proc.info["bads"]) > 0:
        raw_proc.interpolate_bads(reset_bads=True, verbose=False)
    return raw_proc
