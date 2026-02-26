"""
LangGraph: Node return - Partial state update.

Syntax: return {"key": value} or return {**state, "key": value}

Node can return subset of state - only returned keys are updated.
Other keys preserved. Use {**state, "new_key": value} to merge.
"""

from typing import TypedDict


class MyState(TypedDict):
    counter: int
    messages: list


def node(state: MyState) -> dict:
    # Only update counter
    return {"counter": state["counter"] + 1}
    # Or merge: return {**state, "counter": state["counter"] + 1}
