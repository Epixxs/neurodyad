from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np

def plot_3d_latent_space(embeddings: np.ndarray, labels: np.ndarray, title: str = "3D CEBRA Latent Space", save_path: Optional[str] = None) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=labels, cmap="viridis", s=4, alpha=0.6)
    plt.colorbar(scatter, label="Behavioral Label", shrink=0.6)
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_consistency_matrix(consistency_matrix: np.ndarray, title: str = "Consistency Scores Across Runs (%)", save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(6, 5))
    im = plt.imshow(consistency_matrix, cmap="Greys", aspect="auto")
    plt.colorbar(im, label="Consistency score (%)")
    for i in range(consistency_matrix.shape[0]):
        for j in range(consistency_matrix.shape[1]):
            plt.text(j, i, f"{consistency_matrix[i, j]:.1f}", ha="center", va="center")
    plt.title(title)
    plt.xlabel("Run Index")
    plt.ylabel("Run Index")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_training_loss(loss_history: List[float], title: str = "CEBRA InfoNCE Training Loss", save_path: Optional[str] = None) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, color="purple", lw=1.5)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("InfoNCE Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
