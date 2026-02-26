"""
LangGraph: Annotated with add_messages - Merge messages in state.

Syntax: messages: Annotated[List[BaseMessage], add_messages]

When node returns {"messages": [new_msg]}, add_messages appends/merges
instead of replacing. Use for conversation state.
"""

from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
# Node returns {"messages": [AIMessage(...)]} -> appended to state["messages"]
