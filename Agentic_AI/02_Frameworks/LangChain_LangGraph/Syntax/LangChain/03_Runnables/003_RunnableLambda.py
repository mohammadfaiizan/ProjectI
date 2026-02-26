"""
Runnables: RunnableLambda - Wrap a Python function.

Syntax: runnable = RunnableLambda(func)

Input: Passed to func
Output: func(input)

func signature: (x: Any) -> Any
Use: Custom transformation in chain
"""

from langchain_core.runnables import RunnableLambda


def add_metadata(x: dict) -> dict:
    return {**x, "meta": "added"}


runnable = RunnableLambda(add_metadata)
# runnable.invoke({"a": 1}) -> {"a": 1, "meta": "added"}
