"""
Models: llm.bind_tools() - Enable function/tool calling for agents.

Syntax: llm_with_tools = llm.bind_tools(tools)

Input: tools - List[BaseTool] (from @tool or Tool())
Output: Runnable that can return tool_calls in AIMessage

Used in agents: agent receives messages, may output tool_calls for executor to run
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools([add])
# Used in agent: llm_with_tools receives messages, may return AIMessage with tool_calls
