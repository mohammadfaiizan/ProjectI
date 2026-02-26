"""
LangSmith: client.create_run() - Manually create a run.

Syntax: client.create_run(name="...", inputs={...}, run_type="chain")

For custom instrumentation when @traceable isn't used.
"""

from langsmith import Client

client = Client()
# run = client.create_run(
#     name="custom_operation",
#     inputs={"query": "hello"},
#     run_type="chain",
#     project_name="my-project"
# )
