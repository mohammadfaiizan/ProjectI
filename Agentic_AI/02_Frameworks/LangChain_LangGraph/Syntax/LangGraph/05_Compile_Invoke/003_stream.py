"""
LangGraph: app.stream() - Stream state updates as graph runs.

Syntax: for chunk in app.stream(initial_state): ...
        for chunk in app.stream(initial_state, config): ...

Input: Same as invoke
Output: Iterator of (node_name, state_update) tuples
  Each chunk: (str, dict) - node that ran and state delta
"""

from typing import TypedDict


class MyState(TypedDict):
    messages: list


# app = workflow.compile()
# for node_name, state_update in app.stream({"messages": [...]}):
#     print(f"{node_name}: {state_update}")
