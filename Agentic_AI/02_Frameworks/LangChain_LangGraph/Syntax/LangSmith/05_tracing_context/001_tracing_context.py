"""
LangSmith: tracing_context - Set tracing config for a block.

Syntax: with tracing_context(project_name="..."): ...

Parameters: project_name, tags, metadata, enabled, client
"""

from langsmith.run_helpers import tracing_context


with tracing_context(project_name="staging"):
    # All traces in this block go to "staging" project
    result = "traced"
