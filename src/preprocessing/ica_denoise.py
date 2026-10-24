from typing import Optional, List
import mne

def run_extended_infomax_ica(raw: mne.io.BaseRaw, n_components: int = 15, exclude_components: Optional[List[int]] = None, random_state: int = 42) -> mne.io.BaseRaw:
    ica = mne.preprocessing.ICA(n_components=n_components, method="infomax", fit_params=dict(extended=True), random_state=random_state, max_iter="auto")
    ica.fit(raw, verbose=False)
    if exclude_components is not None:
        ica.exclude = exclude_components
        return ica.apply(raw.copy(), verbose=False)
    return raw
