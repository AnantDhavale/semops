from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseAdapter(ABC):
    """
    Every embedding model must implement this.
    semops doesn't care what's under the hood.
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single string into a vector."""
        pass

    def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of strings. Override for batch efficiency."""
        return [self.embed(t) for t in texts]
