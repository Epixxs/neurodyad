from typing import Dict, Any, Optional
import numpy as np

class DyadicCEBRAModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            "model_architecture": "offset10-model",
            "batch_size": 512,
            "learning_rate": 3e-4,
            "temperature_mode": "auto",
            "temperature": 1.0,
            "min_temperature": 0.1,
            "max_iterations": 5000,
            "conditional": "time_delta",
            "output_dimension": 3,
            "distance": "cosine",
            "device": "cuda:0",
            "verbose": True,
            "time_offsets": 10,
        }
        if config:
            self.config.update(config)
        self.model = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DyadicCEBRAModel":
        import cebra
        self.model = cebra.CEBRA(**self.config)
        if y is not None and self.config.get("conditional") != "time_delta":
            self.model.fit(X, y)
        else:
            self.model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    @property
    def loss_history(self):
        if self.model is not None and hasattr(self.model, "loss_"):
            return self.model.loss_
        return []
