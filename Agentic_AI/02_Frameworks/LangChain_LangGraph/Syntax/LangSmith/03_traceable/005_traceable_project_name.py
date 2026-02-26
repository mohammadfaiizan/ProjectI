"""
LangSmith: @traceable(project_name="...") - Target project for this run.

Syntax: @traceable(project_name="my-project")

Overrides LANGSMITH_PROJECT for this function's traces.
"""

from langsmith.run_helpers import traceable


@traceable(project_name="analytics-pipeline")
def analyze_data(data: list) -> dict:
    return {"summary": "done"}
