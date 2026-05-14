# semops

**Semantic operations for Python. The NumPy of meaning.**

```bash
pip install semops
pip install semops[openai]
pip install semops[local]
pip install semops[parquet]
```

Run the example scripts from the project root:

```bash
python semops/example_tickets.py
python semops/example_monitoring.py
```

## What it is

`semops` is a local, model-agnostic semantic utility layer for Python.

It gives you reusable primitives for embedding, similarity, ranking, clustering, reference-set scoring, threshold calibration, lightweight evaluation, and local monitoring.

It does not ship cloud infra, hosted APIs, or bundled model credits.

## Features

`semops` solves code-level semantic tasks that teams hit immediately:

- semantic search and relevance ranking
- duplicate detection and record cleanup
- clustering unlabeled text into themes
- routing tickets, queries, or documents by meaning
- checking whether outputs stay on-topic, on-brand, or near policy examples
- evaluating thresholds before hard-coding rules into an app or pipeline
- exporting semantic scores into notebooks, data jobs, or existing observability stacks

## Core API

```python
import semops as so
from semops.adapters.local_adapter import LocalAdapter

adapter = LocalAdapter()

vector = so.embed("refund failed after login", adapter)
score = so.sim("login broken", "unable to sign in", adapter)
delta = so.diff("old copy", "new copy", adapter)
shift = so.drift("before", "after", adapter)
ranked = so.rank("auth issues", ["reset email failed", "crash on boot"], adapter)
match = so.nearest("cannot log in", ["password reset failed", "login is broken"], adapter)
groups = so.cluster(["login failed", "app crashes", "sign in blocked"], adapter, k=2)
```

## Batch APIs

```python
scores = so.sim_many("login issue", tickets, adapter)
matrix = so.sim_matrix(queries, tickets, adapter)
best = so.nearest_many(queries, tickets, adapter)
rankings = so.rank_many(queries, tickets, adapter, top_k=3)
```

## Bring your own model

```python
from semops.base import BaseAdapter
import numpy as np

class MyAdapter(BaseAdapter):
    def embed(self, text: str) -> np.ndarray:
        ...
```

## Reference-set helpers

Use semantic examples as anchors for brand, policy, category, or quality checks.

```python
brand_examples = [
    "Write with clarity and restraint.",
    "Avoid hype and overstatement.",
]

scores = so.score_references(candidate_texts, brand_examples, adapter)
matches = so.match_references("This copy is bold and loud", brand_examples, adapter)
distance = so.reference_distance("Guaranteed forever", brand_examples, adapter)
```

## Local monitoring primitives

These APIs are intentionally local-only. They score datasets but do not persist, schedule, or alert.

```python
drift_report = so.score_dataset_drift(baseline_outputs, candidate_outputs, adapter, threshold=0.35)
topic_report = so.score_off_topic(candidate_outputs, topic_examples, adapter, threshold=0.45)
brand_report = so.score_off_brand(candidate_outputs, brand_examples, adapter, threshold=0.50)
policy_report = so.score_policy_distance(candidate_outputs, policy_examples, adapter, threshold=0.55)
```

## Calibration and evaluation

```python
scores = [0.92, 0.81, 0.34, 0.18]
labels = [True, True, False, False]

metrics = so.classification_metrics(scores, labels, threshold=0.5)
best = so.calibrate_threshold(scores, labels, metric="f1")
curve = so.sweep_thresholds(scores, labels)
```

```python
queries = ["cannot log in", "reset email missing"]
expected = ["login is broken", "password reset email never arrived"]
corpus = [
    "login is broken",
    "password reset email never arrived",
    "app crashes on startup",
]

retrieval = so.evaluate_retrieval(queries, expected, corpus, adapter)
```

## Instrumentation and exporters

`semops` includes a simple local event schema for app-side instrumentation and export helpers for notebooks and pipelines.

```python
sink = so.ListSink()
report = so.score_off_brand(candidate_outputs, brand_examples, adapter, threshold=0.50)
so.emit_report(sink, report, tags={"project": "docs-bot"})

rows = sink.rows()
so.export_json(rows, "semops_events.json")
so.export_csv(rows, "semops_events.csv")
```

Parquet export is optional and requires `semops[parquet]`.

## Adapters

| Adapter | Install | Use case |
|---|---|---|
| `OpenAIAdapter` | `pip install semops[openai]` | Best quality, bring your own key |
| `LocalAdapter` | `pip install semops[local]` | Fully local embeddings |
| `StubAdapter` | core | Deterministic testing |
| `YourAdapter` | - | Bring your own model |

## What it is not

- Not a vector database
- Not a hosted API
- Not a tracing platform
- Not a monitoring dashboard
- Not a RAG framework
- Not free cloud infra or bundled LLM usage

## OSS boundary

Included here:

- semantic primitives and batch helpers
- adapters and adapter interfaces
- local evaluation, calibration, and reference-set scoring
- local-only monitoring reports
- generic event records and file export helpers

Not included here:

- hosted ingestion clients
- dashboards, alerts, or scheduled jobs
- team workflows, review queues, or governance controls
- bundled API credits or managed model access

## Examples

- [example_tickets.py](/Users/Anant/Desktop/semops/semops/example_tickets.py)
- [example_monitoring.py](/Users/Anant/Desktop/semops/semops/example_monitoring.py)

## Contact

If you have any questions, suggestions or find the library useful, please drop me a line at anantdhavale@gmail.com

Also check 'Cerone' on pypi pip install cerone

## License

MIT
