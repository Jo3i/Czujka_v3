#import numpy as np

#def aggregate_embeddings(embeddings: np.ndarray) -> np.ndarray:
#    """
#    embeddings: (T, 128) lub (128,) – sekwencja embeddingów VGGish
#    return: (128,) – jeden wektor cech
#    """
#    if embeddings.ndim == 1 and embeddings.shape[0] == 128:
#        # już mamy jeden wektor – nic nie robimy
#        return embeddings
#    elif embeddings.ndim == 2 and embeddings.shape[1] == 128:
#        # średnia po czasie
#        return embeddings.mean(axis=0)
#    else:
#        raise ValueError(f"Niepoprawny kształt embeddingów: {embeddings.shape}")
