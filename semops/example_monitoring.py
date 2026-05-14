"""
semops — local monitoring example

Run with:
    pip install semops[local]
    python semops/example_monitoring.py
"""

import semops as so
from semops.adapters.local_adapter import LocalAdapter

adapter = LocalAdapter()

brand_examples = [
    "Write with clarity, restraint, and directness.",
    "Use plain language and avoid hype.",
    "Explain tradeoffs concretely instead of making vague claims.",
]

topic_examples = [
    "Password resets and login issues",
    "Authentication failures and locked accounts",
    "Access problems signing into the app",
]

policy_examples = [
    "Never promise guaranteed uptime.",
    "Avoid collecting secrets in plain text.",
    "Do not claim compliance that has not been verified.",
]

candidate_outputs = [
    "Our product definitely guarantees zero downtime forever.",
    "Customers are seeing login failures after password resets.",
    "We should inspect the authentication flow and email delivery path.",
]

brand_report = so.score_off_brand(candidate_outputs, brand_examples, adapter, threshold=0.50)
topic_report = so.score_off_topic(candidate_outputs, topic_examples, adapter, threshold=0.45)
policy_report = so.score_policy_distance(candidate_outputs, policy_examples, adapter, threshold=0.55)

print("=== Brand alignment ===")
for item in brand_report.items:
    print(f"{item.score:.2f} flagged={item.flagged} :: {item.text}")

print("\n=== Topic alignment ===")
for item in topic_report.items:
    print(f"{item.score:.2f} flagged={item.flagged} :: {item.text}")

print("\n=== Policy distance ===")
for item in policy_report.items:
    print(f"{item.score:.2f} flagged={item.flagged} :: {item.text}")

print("\n=== Dataset summaries ===")
print(brand_report)
print(topic_report)
print(policy_report)
