"""
LangSmith: create_feedback(correction=...) - Ground truth.

Syntax: client.create_feedback(..., correction={"answer": "correct value"})

Correction: dict - the proper/correct output for this run.
Use for model evaluation and fine-tuning data.
"""

from langsmith import Client

client = Client()
# client.create_feedback(
#     run_id="...",
#     key="correctness",
#     score=0.8,
#     correction={"expected": "Paris", "actual": "Paris"}
# )
