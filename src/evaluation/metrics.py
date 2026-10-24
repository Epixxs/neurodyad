from typing import Dict, Any, List
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def compute_knn_accuracy(emb_train: np.ndarray, y_train: np.ndarray, emb_test: np.ndarray, y_test: np.ndarray, n_neighbors: int = 5, metric: str = "cosine") -> float:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    knn.fit(emb_train, y_train)
    return float(knn.score(emb_test, y_test))

def compute_goodness_of_fit(emb_train: np.ndarray, y_train: np.ndarray, emb_test: np.ndarray, y_test: np.ndarray) -> float:
    reg = Ridge(alpha=1.0)
    reg.fit(emb_train, y_train)
    preds = reg.predict(emb_test)
    return float(r2_score(y_test, preds))

def compute_consistency_score(embeddings_list: List[np.ndarray]) -> np.ndarray:
    try:
        import cebra
        if hasattr(cebra.sklearn.metrics, "consistency_score"):
            return cebra.sklearn.metrics.consistency_score(embeddings_list)
    except Exception:
        pass
    num_runs = len(embeddings_list)
    matrix = np.zeros((num_runs, num_runs))
    for i in range(num_runs):
        for j in range(num_runs):
            if i == j:
                matrix[i, j] = 100.0
            else:
                e1, e2 = embeddings_list[i], embeddings_list[j]
                norm_e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-8)
                norm_e2 = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-8)
                matrix[i, j] = float(np.clip(np.mean(np.sum(norm_e1 * norm_e2, axis=1)) * 100.0, 0, 100.0))
    return matrix

def evaluate_embedding_quality(emb_train: np.ndarray, y_train: np.ndarray, emb_test: np.ndarray, y_test: np.ndarray, n_neighbors: int = 5) -> Dict[str, float]:
    return {
        "goodness_of_fit": compute_goodness_of_fit(emb_train, y_train, emb_test, y_test),
        "knn_accuracy": compute_knn_accuracy(emb_train, y_train, emb_test, y_test, n_neighbors=n_neighbors)
    }
