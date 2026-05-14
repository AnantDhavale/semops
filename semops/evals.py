from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from semops.ops import rank
from semops.base import BaseAdapter


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


@dataclass(frozen=True)
class RetrievalMetrics:
    total_queries: int
    top1_accuracy: float
    mean_reciprocal_rank: float
    hit_rate_at_3: float
    hit_rate_at_5: float


def _as_arrays(scores: Sequence[float], labels: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    if len(scores) == 0:
        raise ValueError("scores and labels must not be empty")
    return np.array(scores, dtype=np.float32), np.array(labels, dtype=bool)


def classification_metrics(
    scores: Sequence[float], labels: Sequence[bool], threshold: float
) -> ThresholdMetrics:
    """Compute binary metrics for a threshold over semantic scores."""
    score_array, label_array = _as_arrays(scores, labels)
    predictions = score_array >= threshold

    tp = int(np.sum(predictions & label_array))
    fp = int(np.sum(predictions & ~label_array))
    tn = int(np.sum(~predictions & ~label_array))
    fn = int(np.sum(~predictions & label_array))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(score_array)

    return ThresholdMetrics(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float(accuracy),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def sweep_thresholds(
    scores: Sequence[float], labels: Sequence[bool], thresholds: Iterable[float] | None = None
) -> List[ThresholdMetrics]:
    """Evaluate many thresholds over the same labeled score set."""
    score_array, label_array = _as_arrays(scores, labels)

    if thresholds is None:
        points = sorted({0.0, 1.0, *[float(v) for v in score_array.tolist()]})
    else:
        points = [float(v) for v in thresholds]

    return [classification_metrics(score_array, label_array, threshold) for threshold in points]


def calibrate_threshold(
    scores: Sequence[float], labels: Sequence[bool], metric: str = "f1"
) -> ThresholdMetrics:
    """Pick the threshold that maximizes a simple metric."""
    metric_name = metric.lower()
    valid_metrics = {"f1", "precision", "recall", "accuracy"}
    if metric_name not in valid_metrics:
        raise ValueError(f"metric must be one of {sorted(valid_metrics)}")

    candidates = sweep_thresholds(scores, labels)
    return max(candidates, key=lambda item: (getattr(item, metric_name), -abs(item.threshold - 0.5)))


def evaluate_retrieval(
    queries: Sequence[str],
    expected_targets: Sequence[str],
    corpus: Sequence[str],
    adapter: BaseAdapter,
) -> RetrievalMetrics:
    """Score how often semantic ranking surfaces the expected target."""
    if len(queries) != len(expected_targets):
        raise ValueError("queries and expected_targets must be the same length")
    if not queries:
        raise ValueError("queries and expected_targets must not be empty")

    reciprocal_ranks: List[float] = []
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0

    corpus_list = list(corpus)
    for query, expected in zip(queries, expected_targets):
        ranked = rank(query, corpus_list, adapter)
        ordered_texts = [text for text, _ in ranked]
        if expected not in ordered_texts:
            reciprocal_ranks.append(0.0)
            continue

        hit_index = ordered_texts.index(expected)
        reciprocal_ranks.append(1.0 / (hit_index + 1))
        if hit_index == 0:
            top1_hits += 1
        if hit_index < 3:
            top3_hits += 1
        if hit_index < 5:
            top5_hits += 1

    total = len(queries)
    return RetrievalMetrics(
        total_queries=total,
        top1_accuracy=top1_hits / total,
        mean_reciprocal_rank=float(np.mean(reciprocal_ranks)),
        hit_rate_at_3=top3_hits / total,
        hit_rate_at_5=top5_hits / total,
    )
