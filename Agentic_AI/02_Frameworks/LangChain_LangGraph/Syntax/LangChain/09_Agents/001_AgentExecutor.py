"""
Agents: AgentExecutor - Run agent loop with tools.

Syntax: executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)

Parameters:
  agent: Runnable that outputs AgentAction/AgentFinish
  tools: List[BaseTool]
  verbose: bool - print steps
  handle_parsing_errors: bool | str | Callable
  max_iterations: int - max tool calls
"""

from langchain.agents import AgentExecutor

# executor = AgentExecutor(agent=..., tools=[...])
