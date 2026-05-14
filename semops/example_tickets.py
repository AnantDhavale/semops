"""
semops — End-to-end example
Use case: Support Ticket Deduplication & Routing

Real-world problem: 500 support tickets, half are duplicates phrased differently.
semops clusters, deduplicates, and ranks them — no training data, no labels.

Run with:
    pip install semops[local]
    python example_tickets.py
"""

import semops as so
from semops.adapters.local_adapter import LocalAdapter

adapter = LocalAdapter()  # all-MiniLM-L6-v2, downloads once then cached

tickets = [
    "I can't log into my account",
    "Login is broken for me",
    "App crashes on startup",
    "Unable to sign in",
    "The app won't open",
    "Password reset isn't working",
    "My password reset email never arrived",
]

# --- Step 1: Cluster into themes (no labels needed) ---
print("=== STEP 1: Cluster by theme ===")
groups = so.cluster(tickets, adapter, k=3)
for i, group in enumerate(groups):
    print(f"\nTheme {i+1}:")
    for t in group:
        print(f"  - {t}")

# Expected output:
# Theme 1: login issues
# Theme 2: app crashes
# Theme 3: password reset

# --- Step 2: Deduplicate incoming ticket ---
print("\n=== STEP 2: Deduplicate ===")
new_ticket = "I cannot log in to my account"
match, score = so.nearest(new_ticket, tickets, adapter)
print(f"Incoming: '{new_ticket}'")
print(f"Matches:  '{match}' ({score:.0%} similar)")

# Expected: 94% match to "I can't log into my account"

# --- Step 3: Route by category ---
print("\n=== STEP 3: Route to team ===")
ranked = so.rank("login and authentication issues", tickets, adapter)
for text, score in ranked:
    print(f"  {score:.0%}  {text}")

# Expected: login tickets float to top, crash/reset tickets rank lower
