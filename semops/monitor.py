from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

from semops.base import BaseAdapter
from semops.ops import _norm
from semops.references import score_references, ReferenceScore, reference_centroid


@dataclass(frozen=True)
class ScoredText:
    text: str
    score: float
    flagged: bool | None
    best_reference: str | None = None


@dataclass(frozen=True)
class MonitorReport:
    metric: str
    score_direction: str
    threshold: float | None
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    flagged_count: int
    flagged_ratio: float
    items: List[ScoredText]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _build_report(
    metric: str,
    score_direction: str,
    scores: Sequence[float],
    items: List[ScoredText],
    threshold: float | None,
    metadata: Dict[str, Any] | None = None,
) -> MonitorReport:
    if len(scores) == 0:
        raise ValueError("scores must contain at least one value")

    score_array = np.array(scores, dtype=np.float32)
    flagged_count = sum(1 for item in items if item.flagged)
    return MonitorReport(
        metric=metric,
        score_direction=score_direction,
        threshold=threshold,
        mean_score=float(np.mean(score_array)),
        median_score=float(np.median(score_array)),
        min_score=float(np.min(score_array)),
        max_score=float(np.max(score_array)),
        flagged_count=flagged_count,
        flagged_ratio=flagged_count / len(items),
        items=items,
        metadata=metadata or {},
    )


def _flag_similarity(score: float, threshold: float | None) -> bool | None:
    return None if threshold is None else score < threshold


def _flag_distance(score: float, threshold: float | None) -> bool | None:
    return None if threshold is None else score > threshold


def score_off_topic(
    texts: Sequence[str],
    topic_examples: Sequence[str],
    adapter: BaseAdapter,
    threshold: float | None = None,
) -> MonitorReport:
    """Score whether texts still align to topic examples."""
    reference_scores = score_references(texts, topic_examples, adapter)
    item_scores = [item.best_similarity for item in reference_scores]
    items = [
        ScoredText(
            text=item.text,
            score=item.best_similarity,
            flagged=_flag_similarity(item.best_similarity, threshold),
            best_reference=item.best_reference,
        )
        for item in reference_scores
    ]
    return _build_report(
        metric="off_topic",
        score_direction="higher_is_better",
        scores=item_scores,
        items=items,
        threshold=threshold,
        metadata={"reference_size": len(topic_examples)},
    )


def score_off_brand(
    texts: Sequence[str],
    brand_examples: Sequence[str],
    adapter: BaseAdapter,
    threshold: float | None = None,
) -> MonitorReport:
    """Score whether texts still align to brand voice examples."""
    reference_scores = score_references(texts, brand_examples, adapter)
    item_scores = [item.centroid_similarity for item in reference_scores]
    items = [
        ScoredText(
            text=item.text,
            score=item.centroid_similarity,
            flagged=_flag_similarity(item.centroid_similarity, threshold),
            best_reference=item.best_reference,
        )
        for item in reference_scores
    ]
    return _build_report(
        metric="off_brand",
        score_direction="higher_is_better",
        scores=item_scores,
        items=items,
        threshold=threshold,
        metadata={"reference_size": len(brand_examples)},
    )


def score_policy_distance(
    texts: Sequence[str],
    policy_examples: Sequence[str],
    adapter: BaseAdapter,
    threshold: float | None = None,
) -> MonitorReport:
    """Measure semantic distance from policy examples."""
    reference_scores = score_references(texts, policy_examples, adapter)
    item_scores = [item.distance for item in reference_scores]
    items = [
        ScoredText(
            text=item.text,
            score=item.distance,
            flagged=_flag_distance(item.distance, threshold),
            best_reference=item.best_reference,
        )
        for item in reference_scores
    ]
    return _build_report(
        metric="policy_distance",
        score_direction="lower_is_better",
        scores=item_scores,
        items=items,
        threshold=threshold,
        metadata={"reference_size": len(policy_examples)},
    )


def score_drift(
    baseline_texts: Sequence[str],
    candidate_texts: Sequence[str],
    adapter: BaseAdapter,
    threshold: float | None = None,
) -> MonitorReport:
    """Measure how far a candidate dataset has drifted from a baseline set."""
    if len(baseline_texts) == 0:
        raise ValueError("baseline_texts must contain at least one text")
    if len(candidate_texts) == 0:
        raise ValueError("candidate_texts must contain at least one text")

    baseline_center = reference_centroid(baseline_texts, adapter)
    candidate_vecs = np.array(
        [_norm(v) for v in adapter.embed_many(list(candidate_texts))], dtype=np.float32
    )
    alignment_scores = np.clip(np.dot(candidate_vecs, baseline_center), 0, 1)
    drift_scores = 1.0 - alignment_scores

    items = [
        ScoredText(text=text, score=float(score), flagged=_flag_distance(float(score), threshold))
        for text, score in zip(candidate_texts, drift_scores)
    ]

    candidate_center = _norm(candidate_vecs.mean(axis=0))
    dataset_alignment = float(np.clip(np.dot(candidate_center, baseline_center), 0, 1))

    return _build_report(
        metric="drift",
        score_direction="lower_is_better",
        scores=[float(v) for v in drift_scores],
        items=items,
        threshold=threshold,
        metadata={
            "baseline_size": len(baseline_texts),
            "candidate_size": len(candidate_texts),
            "dataset_alignment": dataset_alignment,
            "dataset_drift": 1.0 - dataset_alignment,
        },
    )


def score_dataset_drift(
    baseline_texts: Sequence[str],
    candidate_texts: Sequence[str],
    adapter: BaseAdapter,
    threshold: float | None = None,
) -> MonitorReport:
    """Alias for score_drift with a more explicit monitoring name."""
    return score_drift(baseline_texts, candidate_texts, adapter, threshold=threshold)
