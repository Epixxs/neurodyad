from enum import Enum
from typing import List, Union
import numpy as np

class ChannelAblationScheme(Enum):
    SPEECH_COMPREHENSION = "speech_comprehension"
    EXECUTIVE_FUNCTION = "executive_function"

DEFAULT_SPEECH_CHANNELS: List[int] = [7, 8, 14, 15, 23, 24, 40, 41]
DEFAULT_EXECUTIVE_CHANNELS: List[int] = [0, 1, 2, 3, 4, 16, 17, 18]

def ablate_eeg_channels(X: np.ndarray, ablation_scheme: Union[ChannelAblationScheme, str, List[int]], channels_per_participant: int = 64) -> np.ndarray:
    X_ablated = X.copy()
    if isinstance(ablation_scheme, list):
        target_indices = ablation_scheme
    elif ablation_scheme in (ChannelAblationScheme.SPEECH_COMPREHENSION, "speech_comprehension", "v3"):
        target_indices = DEFAULT_SPEECH_CHANNELS
    elif ablation_scheme in (ChannelAblationScheme.EXECUTIVE_FUNCTION, "executive_function", "v5"):
        target_indices = DEFAULT_EXECUTIVE_CHANNELS
    else:
        raise ValueError(f"Unknown ablation scheme: {ablation_scheme}")
    for ch_idx in target_indices:
        if ch_idx < channels_per_participant:
            X_ablated[:, ch_idx] = 0.0
            X_ablated[:, ch_idx + channels_per_participant] = 0.0
    return X_ablated
