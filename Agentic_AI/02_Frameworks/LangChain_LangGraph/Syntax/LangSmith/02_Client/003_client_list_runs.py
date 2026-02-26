"""
LangSmith: client.list_runs() - List runs for a project.

Syntax: runs = client.list_runs(project_name="...", limit=10)

Input: project_name, limit, filter, etc.
Output: Iterator of Run objects
"""

from langsmith import Client

client = Client()
# for run in client.list_runs(project_name="my-project", limit=100):
#     print(run.id, run.name)
