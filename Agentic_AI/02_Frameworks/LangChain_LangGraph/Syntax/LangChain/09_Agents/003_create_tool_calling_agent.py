"""
Agents: create_tool_calling_agent() - Create agent from llm + tools + prompt.

Syntax: agent = create_tool_calling_agent(llm, tools, prompt)

Input: llm (with tool binding), tools, prompt with MessagesPlaceholder("agent_scratchpad")
Output: Runnable agent
"""

from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [add]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You have tools."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
