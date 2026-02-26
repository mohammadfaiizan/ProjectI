"""
LangSmith: tracing_context(enabled=False) - Disable tracing.

Syntax: with tracing_context(enabled=False): ...

Temporarily disable tracing for a block.
enabled: bool | 'local' | None
"""

from langsmith.run_helpers import tracing_context


with tracing_context(enabled=False):
    # No traces logged for this block
    pass  # sensitive_operation()
