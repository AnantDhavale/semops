import numpy as np
from semops.base import BaseAdapter


class OpenAIAdapter(BaseAdapter):
    """OpenAI embeddings adapter."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install semops[openai]")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(input=text, model=self.model)
        return np.array(response.data[0].embedding, dtype=np.float32)

    def embed_many(self, texts: list) -> list:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [np.array(d.embedding, dtype=np.float32) for d in response.data]
