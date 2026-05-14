from typing import List, Sequence, Tuple
import numpy as np
from semops.base import BaseAdapter


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _embed_many_array(texts: Sequence[str], adapter: BaseAdapter) -> np.ndarray:
    if len(texts) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.array([_norm(v) for v in adapter.embed_many(list(texts))], dtype=np.float32)


def embed(text: str, adapter: BaseAdapter) -> np.ndarray:
    """Embed a single string into a vector."""
    return np.array(adapter.embed(text), dtype=np.float32)


def embed_many(texts: Sequence[str], adapter: BaseAdapter) -> List[np.ndarray]:
    """Embed a batch of strings into vectors."""
    return [np.array(v, dtype=np.float32) for v in adapter.embed_many(list(texts))]


def sim(a: str, b: str, adapter: BaseAdapter) -> float:
    """Semantic similarity between two strings. Returns 0-1."""
    va, vb = _norm(adapter.embed(a)), _norm(adapter.embed(b))
    return float(np.clip(np.dot(va, vb), 0, 1))


def diff(a: str, b: str, adapter: BaseAdapter) -> float:
    """Semantic difference. Inverse of sim. Returns 0-1."""
    return 1.0 - sim(a, b, adapter)


def drift(v1: str, v2: str, adapter: BaseAdapter) -> float:
    """
    How much meaning has shifted between two versions of text.
    0 = identical meaning. 1 = completely different.
    """
    return diff(v1, v2, adapter)


def nearest(query: str, corpus: List[str], adapter: BaseAdapter) -> Tuple[str, float]:
    """Find the closest meaning match in a corpus."""
    if len(corpus) == 0:
        raise ValueError("corpus must contain at least one text")
    qv = _norm(adapter.embed(query))
    vecs = adapter.embed_many(corpus)
    scores = [float(np.dot(qv, _norm(v))) for v in vecs]
    idx = int(np.argmax(scores))
    return corpus[idx], scores[idx]


def classify(text: str, labels: List[str], adapter: BaseAdapter) -> Tuple[str, float]:
    """Assign a text to the closest semantic label."""
    if len(labels) == 0:
        raise ValueError("labels must contain at least one label")
    return nearest(text, labels, adapter)


def rank(query: str, corpus: List[str], adapter: BaseAdapter) -> List[Tuple[str, float]]:
    """Rank corpus by semantic relevance to query."""
    if len(corpus) == 0:
        return []
    qv = _norm(adapter.embed(query))
    vecs = adapter.embed_many(corpus)
    scored = [(corpus[i], float(np.dot(qv, _norm(v)))) for i, v in enumerate(vecs)]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def cluster(texts: List[str], adapter: BaseAdapter, k: int = 3) -> List[List[str]]:
    """
    Group texts by meaning using k-means.
    Returns k lists of texts.
    """
    if len(texts) == 0:
        return []
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > len(texts):
        raise ValueError("k cannot exceed the number of texts")

    vecs = _embed_many_array(texts, adapter)

    # simple k-means
    rng = np.random.default_rng(42)
    centroids = vecs[rng.choice(len(vecs), k, replace=False)]

    for _ in range(50):
        dists = np.dot(vecs, centroids.T)
        labels = np.argmax(dists, axis=1)
        new_centroids = np.array([
            vecs[labels == i].mean(axis=0) if (labels == i).any() else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    groups: List[List[str]] = [[] for _ in range(k)]
    for text, label in zip(texts, labels):
        groups[label].append(text)
    return groups


def dedup(texts: List[str], adapter: BaseAdapter, threshold: float = 0.85) -> List[List[str]]:
    """
    Group near-duplicate texts using semantic similarity.
    Returns only groups with at least two items.
    """
    if len(texts) == 0:
        return []
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    vecs = _embed_many_array(texts, adapter)
    sim_matrix = np.clip(np.dot(vecs, vecs.T), 0, 1)

    visited = [False] * len(texts)
    groups: List[List[str]] = []

    for start in range(len(texts)):
        if visited[start]:
            continue

        stack = [start]
        component: List[int] = []
        visited[start] = True

        while stack:
            idx = stack.pop()
            component.append(idx)
            neighbors = np.where(sim_matrix[idx] >= threshold)[0]
            for neighbor in neighbors:
                neighbor_idx = int(neighbor)
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    stack.append(neighbor_idx)

        if len(component) >= 2:
            component.sort()
            groups.append([texts[idx] for idx in component])

    return groups
