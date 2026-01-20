import numpy as np


def aggregate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Agregacja embeddingów audio (np. z VGGish) do jednego wektora.

    Args:
        embeddings: tablica shape (N, D)

    Returns:
        wektor cech shape (D,)
    """

    if embeddings.ndim == 1:
        return embeddings

    return np.mean(embeddings, axis=0)
