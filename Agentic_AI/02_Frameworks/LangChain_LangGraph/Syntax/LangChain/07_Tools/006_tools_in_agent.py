"""
Tools: Passing tools to AgentExecutor.

Syntax: AgentExecutor(agent=agent, tools=[tool1, tool2])

tools: List[BaseTool] - all tools agent can call
Agent decides which tool to use based on tool descriptions.
"""

from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


tools = [add]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You have tools."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
# executor.invoke({"input": "What is 5+7?"})
