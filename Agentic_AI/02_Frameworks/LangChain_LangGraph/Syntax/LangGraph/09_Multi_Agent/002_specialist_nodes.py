"""
LangGraph: Specialist nodes - Different agents for different tasks.

Each specialist: def specialist_node(state): ...
  - Reads task from state["messages"]
  - Returns {"messages": [AIMessage(...)]}
  - Shared state schema across all specialists
"""

from langchain_core.messages import AIMessage, HumanMessage


def coder_node(state):
    messages = state["messages"]
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human:
        return {"messages": [AIMessage(content=f"Code for: {last_human.content}")]}
    return state


def writer_node(state):
    messages = state["messages"]
    last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last_human:
        return {"messages": [AIMessage(content=f"Article: {last_human.content}")]}
    return state
