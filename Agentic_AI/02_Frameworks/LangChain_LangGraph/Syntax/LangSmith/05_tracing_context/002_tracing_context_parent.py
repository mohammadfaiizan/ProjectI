"""
LangSmith: tracing_context(parent=...) - Distributed tracing.

Syntax: with tracing_context(parent=request.headers): ...

Parent: RunTree, headers dict, or dotted order string.
Propagates trace across services via langsmith-trace headers.
"""

from langsmith.run_helpers import tracing_context

# In service B, receiving request from service A:
# with tracing_context(parent=request.headers):
#     return await handle_request()
