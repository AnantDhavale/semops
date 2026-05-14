import numpy as np
from semops.base import BaseAdapter
from typing import List


class LocalAdapter(BaseAdapter):
    """
    sentence-transformers adapter.
    Fully local — no API key, no internet required.
    
    Install: pip install semops[local]
    Default model: all-MiniLM-L6-v2 (fast, lean, 384-dim)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install semops[local]")
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [np.array(v, dtype=np.float32) for v in vecs]
