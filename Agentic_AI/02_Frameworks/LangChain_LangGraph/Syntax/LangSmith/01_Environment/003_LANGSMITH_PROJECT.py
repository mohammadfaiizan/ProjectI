"""
LangSmith: LANGSMITH_PROJECT - Project name for traces.

Syntax: os.environ["LANGCHAIN_PROJECT"] = "my-project"

Traces are grouped by project. Defaults to "default".
"""

import os

os.environ["LANGCHAIN_PROJECT"] = "my-app-production"
# Or set per-run via tracing_context(project_name="...")
