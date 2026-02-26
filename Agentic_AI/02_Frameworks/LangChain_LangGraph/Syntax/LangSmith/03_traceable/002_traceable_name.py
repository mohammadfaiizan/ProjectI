"""
LangSmith: @traceable(name="...") - Custom run name.

Syntax: @traceable(name="CustomName")

Defaults to function name. Override for clearer traces.
"""

from langsmith.run_helpers import traceable


@traceable(name="square_computation")
def square(x: float) -> float:
    return x ** 2
