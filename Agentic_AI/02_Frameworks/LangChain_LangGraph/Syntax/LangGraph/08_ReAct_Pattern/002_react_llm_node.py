"""
LangGraph: ReAct LLM node - Call LLM with tools.

Syntax: def llm_node(state): return {"messages": [llm_with_tools.invoke(state["messages"])]}

LLM must have bind_tools(tools).
Returns AIMessage with optional tool_calls.
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [add]
llm_with_tools = llm.bind_tools(tools)


def llm_node(state):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
