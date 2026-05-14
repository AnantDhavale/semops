import numpy as np
from semops.base import BaseAdapter


class StubAdapter(BaseAdapter):
    """
    Deterministic random adapter for testing.
    Same text always returns same vector.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        seed = sum(ord(c) for c in text)
        rng = np.random.default_rng(seed)
        vec = rng.random(self.dim).astype(np.float32)
        return vec / np.linalg.norm(vec)
