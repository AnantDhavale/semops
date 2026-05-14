# semops

**Semantic operations for Python. The NumPy of meaning.**

`semops` is a local, model-agnostic semantic utility layer for Python.

It gives you reusable primitives for search, classification, deduplication, clustering, reference-set scoring, threshold calibration, lightweight evaluation, and local monitoring.

**Library philosophy:** `semops` computes. User code stores.

## Install

```bash
pip install semops
pip install semops[openai]
pip install semops[local]
```

## Quickstart

```python
import semops as so
from semops.adapters.local_adapter import LocalAdapter

adapter = LocalAdapter()

tickets = [
    "I can't log into my account",
    "Login is broken for me",
    "Password reset email never arrived",
    "Invoice PDF is missing",
]

match, score = so.nearest("Unable to sign in", tickets, adapter)
label, confidence = so.classify(
    "reset email never arrived",
    ["login issue", "password reset", "billing issue"],
    adapter,
)
duplicates = so.dedup(tickets, adapter, threshold=0.80)

print(match, score)
print(label, confidence)
print(duplicates)
```

## What you can build with it

- Semantic search and relevance ranking
- Ticket, query, or document routing by meaning
- Deduplication and record cleanup
- Clustering unlabeled text into themes
- Brand, policy, or topic alignment checks
- Drift checks on outputs, content, or datasets
- Threshold calibration before wiring rules into production

## Core primitives

```python
vector = so.embed(text, adapter)
score = so.sim(a, b, adapter)
difference = so.diff(a, b, adapter)
shift = so.drift(old_text, new_text, adapter)

match, score = so.nearest(query, corpus, adapter)
ranked = so.rank(query, corpus, adapter)
groups = so.cluster(texts, adapter, k=3)

label, score = so.classify(text, labels, adapter)
duplicate_groups = so.dedup(texts, adapter, threshold=0.85)
```

## Batch APIs

```python
scores = so.sim_many(query, corpus, adapter)
matrix = so.sim_matrix(queries, corpus, adapter)
matches = so.nearest_many(queries, corpus, adapter)
rankings = so.rank_many(queries, corpus, adapter, top_k=3)
```

These are useful when you need to score many inputs without writing the loops yourself.

## End-to-end examples

### 1. Search and routing

```python
query = "customer cannot access account"
corpus = [
    "login issue",
    "billing issue",
    "shipping delay",
]

label, score = so.classify(query, corpus, adapter)
print(label, score)
```

### 2. Deduplication

```python
records = [
    "I can't log into my account",
    "Unable to sign in",
    "Invoice PDF is missing",
    "Login is broken for me",
]

duplicate_groups = so.dedup(records, adapter, threshold=0.80)
for group in duplicate_groups:
    print(group)
```

### 3. Brand, policy, and topic scoring

```python
brand_examples = [
    "Write with clarity and restraint.",
    "Avoid hype and overstatement.",
]

topic_examples = [
    "Password resets and login issues",
    "Authentication failures and locked accounts",
]

policy_examples = [
    "Never promise guaranteed uptime.",
    "Do not claim compliance that has not been verified.",
]

candidate_outputs = [
    "Our product definitely guarantees zero downtime forever.",
    "Customers are seeing login failures after password resets.",
]

brand_report = so.score_off_brand(candidate_outputs, brand_examples, adapter, threshold=0.50)
topic_report = so.score_off_topic(candidate_outputs, topic_examples, adapter, threshold=0.45)
policy_report = so.score_policy_distance(candidate_outputs, policy_examples, adapter, threshold=0.55)
```

## Reference-set helpers

Use semantic examples as anchors for brand voice, policies, categories, or quality standards.

```python
scores = so.score_references(candidate_outputs, brand_examples, adapter)
matches = so.match_references("This copy is bold and loud", brand_examples, adapter)
distance = so.reference_distance("Guaranteed forever", brand_examples, adapter)
centroid = so.reference_centroid(brand_examples, adapter)
```

These helpers are useful when you want to ask questions like:
- "How close is this output to our approved examples?"
- "Which example is it most similar to?"
- "How far has this content drifted from the reference set?"

## Local monitoring primitives

`semops` includes local-only monitoring helpers for scoring batches of text.

```python
drift_report = so.score_dataset_drift(baseline_outputs, candidate_outputs, adapter, threshold=0.35)
topic_report = so.score_off_topic(candidate_outputs, topic_examples, adapter, threshold=0.45)
brand_report = so.score_off_brand(candidate_outputs, brand_examples, adapter, threshold=0.50)
policy_report = so.score_policy_distance(candidate_outputs, policy_examples, adapter, threshold=0.55)
```

These return structured reports that you can inspect, export, or feed into your own pipelines.

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

Use these helpers when you need to choose thresholds or compare semantic retrieval quality on your own data.

## Adapters

| Adapter | Install | Use case |
|---|---|---|
| `OpenAIAdapter` | `pip install semops[openai]` | Best quality, bring your own key |
| `LocalAdapter` | `pip install semops[local]` | Fully local embeddings |
| `StubAdapter` | core | Deterministic testing |
| `YourAdapter` | - | Bring your own model |

You can also implement your own adapter with one method:

```python
from semops.base import BaseAdapter
import numpy as np

class MyAdapter(BaseAdapter):
    def embed(self, text: str) -> np.ndarray:
        ...
```

## Examples

Run the example scripts from the project root:

```bash
python semops/example_tickets.py
python semops/example_monitoring.py
```

- [example_tickets.py](/Users/Anant/Desktop/semops/semops/example_tickets.py)
- [example_monitoring.py](/Users/Anant/Desktop/semops/semops/example_monitoring.py)

## OSS boundary

Included here:

- semantic primitives and batch helpers
- adapters and adapter interfaces
- local evaluation, calibration, and reference-set scoring
- local-only monitoring reports

Not included here:

- hosted ingestion clients
- dashboards, alerts, or scheduled jobs
- team workflows, review queues, or governance controls
- bundled API credits or managed model access
- library-owned logging, event sinks, or storage schemas

## Contact

If you have any questions, suggestions or find the library useful, please drop me a line at anantdhavale@gmail.com

Also check `Cerone` on PyPI: `pip install cerone`

## License

MIT
