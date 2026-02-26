"""
LangSmith: client.create_feedback() - Add feedback to a run.

Syntax: client.create_feedback(run_id=..., key="...", score=...)

Input: run_id, key (metric name), score (float/int/bool)
Optional: value, comment, correction
"""

from langsmith import Client

client = Client()
# client.create_feedback(
#     run_id="run-uuid",
#     key="helpfulness",
#     score=0.9,
#     comment="Very helpful response"
# )
