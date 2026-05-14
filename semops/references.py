from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from semops.base import BaseAdapter
from semops.ops import _norm


@dataclass(frozen=True)
class ReferenceScore:
    text: str
    best_reference: str
    best_similarity: float
    centroid_similarity: float
    distance: float


def reference_centroid(references: Sequence[str], adapter: BaseAdapter) -> np.ndarray:
    """Build a single centroid vector from a reference set."""
    if len(references) == 0:
        raise ValueError("references must contain at least one text")
    vecs = np.array([_norm(v) for v in adapter.embed_many(list(references))], dtype=np.float32)
    return _norm(vecs.mean(axis=0))


def match_references(
    text: str, references: Sequence[str], adapter: BaseAdapter, top_k: int = 3
) -> List[Tuple[str, float]]:
    """Find the closest reference examples to a text."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if len(references) == 0:
        raise ValueError("references must contain at least one text")

    query_vec = _norm(adapter.embed(text))
    ref_vecs = adapter.embed_many(list(references))
    scored = [
        (reference, float(np.clip(np.dot(query_vec, _norm(vec)), 0, 1)))
        for reference, vec in zip(references, ref_vecs)
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def reference_distance(
    text: str, references: Sequence[str], adapter: BaseAdapter, mode: str = "nearest"
) -> float:
    """Measure how far a text is from a reference set."""
    scores = score_references([text], references, adapter)
    item = scores[0]
    if mode == "nearest":
        return item.distance
    if mode == "centroid":
        return 1.0 - item.centroid_similarity
    raise ValueError("mode must be 'nearest' or 'centroid'")


def score_references(
    texts: Sequence[str], references: Sequence[str], adapter: BaseAdapter
) -> List[ReferenceScore]:
    """Score a batch of texts against a reference set."""
    if len(references) == 0:
        raise ValueError("references must contain at least one text")
    if len(texts) == 0:
        return []

    reference_list = list(references)
    ref_vecs = np.array([_norm(v) for v in adapter.embed_many(reference_list)], dtype=np.float32)
    ref_centroid = _norm(ref_vecs.mean(axis=0))
    text_vecs = np.array([_norm(v) for v in adapter.embed_many(list(texts))], dtype=np.float32)

    scores: List[ReferenceScore] = []
    for text, vec in zip(texts, text_vecs):
        row = np.clip(np.dot(ref_vecs, vec), 0, 1)
        best_index = int(np.argmax(row))
        best_similarity = float(row[best_index])
        centroid_similarity = float(np.clip(np.dot(ref_centroid, vec), 0, 1))
        scores.append(
            ReferenceScore(
                text=text,
                best_reference=reference_list[best_index],
                best_similarity=best_similarity,
                centroid_similarity=centroid_similarity,
                distance=1.0 - best_similarity,
            )
        )
    return scores
