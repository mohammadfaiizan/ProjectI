"""
LangSmith: @traceable - Decorator to trace any function.

Syntax: @traceable
        def my_func(x): return x

Automatically logs execution as a span in LangSmith.
"""

from langsmith.run_helpers import traceable


@traceable
def my_function(x: float) -> float:
    return x ** 2
# my_function(3) -> traced in LangSmith
