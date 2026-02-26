"""
Agents: handle_parsing_errors - Handle agent output parse failures.

Syntax: AgentExecutor(..., handle_parsing_errors=True)
  or handle_parsing_errors="Custom message"
  or handle_parsing_errors=lambda e: "Fallback"

When agent output fails to parse, return message instead of raising.
"""

from langchain.agents import AgentExecutor


def custom_handler(error: Exception) -> str:
    return "I had trouble understanding. Please rephrase."


# executor = AgentExecutor(..., handle_parsing_errors=custom_handler)
# executor = AgentExecutor(..., handle_parsing_errors=True)  # default message
# executor = AgentExecutor(..., handle_parsing_errors="Sorry, try again.")
