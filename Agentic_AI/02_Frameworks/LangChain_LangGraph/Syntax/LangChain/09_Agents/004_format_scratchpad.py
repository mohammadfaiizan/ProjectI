"""
Agents: format_to_openai_function_messages() - Format tool results for agent.

Syntax: format_to_openai_function_messages(intermediate_steps)

Input: intermediate_steps - List of (AgentAction, observation)
Output: List[BaseMessage] - for agent_scratchpad in next prompt

Used in manual ReAct agent setup.
"""

from langchain.agents.format_scratchpad import format_to_openai_function_messages

# agent_scratchpad = format_to_openai_function_messages(intermediate_steps)
# Pass to prompt: {"agent_scratchpad": agent_scratchpad}
