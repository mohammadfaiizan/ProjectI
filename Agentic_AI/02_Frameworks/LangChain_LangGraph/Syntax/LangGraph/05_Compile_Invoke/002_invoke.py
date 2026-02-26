"""
LangGraph: app.invoke() - Run graph to completion.

Syntax: result = app.invoke(initial_state)
        result = app.invoke(initial_state, config)

Input:
  initial_state: dict - matches state schema
  config: dict - optional, e.g. {"configurable": {"thread_id": "..."}}

Output: dict - final state after graph completes
"""

from typing import TypedDict
from langchain_core.messages import HumanMessage


class MyState(TypedDict):
    messages: list


# app = workflow.compile()
# result = app.invoke({"messages": [HumanMessage(content="Hello")]})
# result["messages"] -> final messages
