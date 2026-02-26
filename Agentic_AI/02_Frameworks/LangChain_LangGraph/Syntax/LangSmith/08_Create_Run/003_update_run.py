"""
LangSmith: client.update_run() - Update run with outputs.

Syntax: client.update_run(run_id=..., outputs={...})

Call after run completes to add outputs, end time.
"""

from langsmith import Client

client = Client()
# client.update_run(
#     run_id="run-uuid",
#     outputs={"result": "success"},
#     end_time=datetime.utcnow()
# )
