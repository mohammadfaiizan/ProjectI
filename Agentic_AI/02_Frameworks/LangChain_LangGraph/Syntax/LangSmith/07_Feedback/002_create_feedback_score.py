"""
LangSmith: create_feedback score - Numeric or boolean rating.

Syntax: score=0.9 (float 0-1) or score=1 (int) or score=True (bool)

Score rates the run on the metric. Use for human eval.
"""

from langsmith import Client

client = Client()
# client.create_feedback(run_id="...", key="accuracy", score=0.95)
# client.create_feedback(run_id="...", key="approved", score=True)
