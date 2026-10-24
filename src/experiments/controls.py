import numpy as np

def shuffle_neural_data(X: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return X[rng.permutation(len(X))]

def shuffle_behavioral_labels(y: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return y[rng.permutation(len(y))]
