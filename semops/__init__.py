"""
semops — semantic operations for Python.
The NumPy of meaning.
"""

from semops.batch import nearest_many, pairwise_sim, rank_many, sim_many, sim_matrix
from semops.base import BaseAdapter
from semops.evals import (
    RetrievalMetrics,
    ThresholdMetrics,
    calibrate_threshold,
    classification_metrics,
    evaluate_retrieval,
    sweep_thresholds,
)
from semops.exporters import export_csv, export_json, export_parquet, to_rows
from semops.instrumentation import ListSink, SemanticEvent, emit_event, emit_metric, emit_report, make_event
from semops.local_adapter import LocalAdapter
from semops.monitor import (
    MonitorReport,
    ScoredText,
    score_dataset_drift,
    score_drift,
    score_off_brand,
    score_off_topic,
    score_policy_distance,
)
from semops.openai_adapter import OpenAIAdapter
from semops.ops import classify, cluster, dedup, diff, drift, embed, embed_many, nearest, rank, sim
from semops.references import (
    ReferenceScore,
    match_references,
    reference_centroid,
    reference_distance,
    score_references,
)
from semops.stub_adapter import StubAdapter

__version__ = "0.3.0"
__all__ = [
    "BaseAdapter",
    "LocalAdapter",
    "MonitorReport",
    "OpenAIAdapter",
    "ReferenceScore",
    "RetrievalMetrics",
    "ScoredText",
    "SemanticEvent",
    "StubAdapter",
    "ThresholdMetrics",
    "ListSink",
    "calibrate_threshold",
    "classification_metrics",
    "classify",
    "cluster",
    "dedup",
    "diff",
    "drift",
    "embed",
    "embed_many",
    "emit_event",
    "emit_metric",
    "emit_report",
    "evaluate_retrieval",
    "export_csv",
    "export_json",
    "export_parquet",
    "make_event",
    "match_references",
    "nearest",
    "nearest_many",
    "pairwise_sim",
    "rank",
    "rank_many",
    "reference_centroid",
    "reference_distance",
    "score_dataset_drift",
    "score_drift",
    "score_off_brand",
    "score_off_topic",
    "score_policy_distance",
    "score_references",
    "sim",
    "sim_many",
    "sim_matrix",
    "sweep_thresholds",
    "to_rows",
]
