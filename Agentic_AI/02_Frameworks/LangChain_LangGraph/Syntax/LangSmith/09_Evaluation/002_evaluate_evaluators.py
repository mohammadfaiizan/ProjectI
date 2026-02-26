"""
LangSmith: evaluate(..., evaluators=[...]) - Custom evaluators.

Syntax: evaluate(target, data="...", evaluators=[qa, embedding_distance])

Evaluators: list of evaluator functions or string presets.
Presets: "qa", "embedding_distance", "context_qa", etc.
"""

from langsmith import evaluate

# results = evaluate(
#     my_chain,
#     data="dataset",
#     evaluators=["qa", "embedding_distance"]
# )
