"""
LangGraph: TypedDict for state schema - Define state structure.

Syntax: class GraphState(TypedDict):
            key: type
            optional_key: NotRequired[type]

All nodes receive and return state matching this schema.
Use NotRequired for optional keys.
"""

from typing import TypedDict
from typing_extensions import NotRequired


class GraphState(TypedDict):
    messages: list
    counter: int
    result: NotRequired[str]
