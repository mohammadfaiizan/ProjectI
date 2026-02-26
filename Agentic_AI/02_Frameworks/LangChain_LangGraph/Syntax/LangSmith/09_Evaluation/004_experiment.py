"""
LangSmith: Experiments - Compare multiple targets.

Syntax: evaluate([target_a, target_b], data="...")

Pass list of targets to compare. Results show side-by-side.
"""

from langsmith import evaluate

# results = evaluate(
#     [chain_v1, chain_v2],
#     data="test-dataset",
#     experiment_prefix="ab-test"
# )
