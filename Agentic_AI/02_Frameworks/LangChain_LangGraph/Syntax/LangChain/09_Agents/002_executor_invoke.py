"""
Agents: executor.invoke() - Run agent with user input.

Syntax: result = executor.invoke({"input": "user question"})

Input: {"input": str} - user question/task
Output: {"output": str, "intermediate_steps": [...]}
  output: Final answer
  intermediate_steps: List of (AgentAction, observation) tuples
"""

from langchain.agents import AgentExecutor

# executor = AgentExecutor(agent=..., tools=[...])
# result = executor.invoke({"input": "What is 7 times 8?"})
# result["output"] -> "56"
# result["intermediate_steps"] -> [(AgentAction(...), "56")]
