from typing import List, Sequence, Tuple
import numpy as np

from semops.base import BaseAdapter
from semops.ops import _norm


def sim_many(query: str, corpus: Sequence[str], adapter: BaseAdapter) -> List[float]:
    """Score one query against many texts."""
    if len(corpus) == 0:
        return []
    qv = _norm(adapter.embed(query))
    vecs = adapter.embed_many(list(corpus))
    return [float(np.clip(np.dot(qv, _norm(v)), 0, 1)) for v in vecs]


def pairwise_sim(
    left_texts: Sequence[str], right_texts: Sequence[str], adapter: BaseAdapter
) -> List[float]:
    """Score two equal-length batches element-by-element."""
    if len(left_texts) != len(right_texts):
        raise ValueError("left_texts and right_texts must be the same length")
    if len(left_texts) == 0:
        return []

    left_vecs = [_norm(v) for v in adapter.embed_many(list(left_texts))]
    right_vecs = [_norm(v) for v in adapter.embed_many(list(right_texts))]
    return [
        float(np.clip(np.dot(left_vecs[i], right_vecs[i]), 0, 1))
        for i in range(len(left_vecs))
    ]


def sim_matrix(
    left_texts: Sequence[str], right_texts: Sequence[str], adapter: BaseAdapter
) -> np.ndarray:
    """Build a similarity matrix between two batches."""
    if len(left_texts) == 0 or len(right_texts) == 0:
        return np.zeros((len(left_texts), len(right_texts)), dtype=np.float32)

    left_vecs = np.array([_norm(v) for v in adapter.embed_many(list(left_texts))])
    right_vecs = np.array([_norm(v) for v in adapter.embed_many(list(right_texts))])
    return np.clip(np.dot(left_vecs, right_vecs.T), 0, 1).astype(np.float32)


def nearest_many(
    queries: Sequence[str], corpus: Sequence[str], adapter: BaseAdapter
) -> List[Tuple[str, float]]:
    """Find the closest corpus match for each query."""
    if len(corpus) == 0:
        raise ValueError("corpus must contain at least one text")
    if len(queries) == 0:
        return []

    corpus_list = list(corpus)
    corpus_vecs = np.array([_norm(v) for v in adapter.embed_many(corpus_list)])
    query_vecs = np.array([_norm(v) for v in adapter.embed_many(list(queries))])
    scores = np.clip(np.dot(query_vecs, corpus_vecs.T), 0, 1)

    matches: List[Tuple[str, float]] = []
    for row in scores:
        idx = int(np.argmax(row))
        matches.append((corpus_list[idx], float(row[idx])))
    return matches


def rank_many(
    queries: Sequence[str],
    corpus: Sequence[str],
    adapter: BaseAdapter,
    top_k: int | None = None,
) -> List[List[Tuple[str, float]]]:
    """Rank the same corpus for multiple queries."""
    if len(queries) == 0:
        return []
    if len(corpus) == 0:
        return [[] for _ in queries]

    corpus_list = list(corpus)
    corpus_vecs = np.array([_norm(v) for v in adapter.embed_many(corpus_list)])
    query_vecs = np.array([_norm(v) for v in adapter.embed_many(list(queries))])
    matrix = np.clip(np.dot(query_vecs, corpus_vecs.T), 0, 1)

    rankings: List[List[Tuple[str, float]]] = []
    for row in matrix:
        order = np.argsort(row)[::-1]
        if top_k is not None:
            order = order[:top_k]
        rankings.append([(corpus_list[idx], float(row[idx])) for idx in order])
    return rankings
