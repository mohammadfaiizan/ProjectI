"""
Agents: max_iterations - Limit tool call loops.

Syntax: AgentExecutor(..., max_iterations=5)

Prevents infinite loops. Agent stops after N tool calls even if not done.
Default: 15
"""

from langchain.agents import AgentExecutor

# executor = AgentExecutor(agent=..., tools=[...], max_iterations=5)
# If agent keeps calling tools, stops after 5 and returns best answer
