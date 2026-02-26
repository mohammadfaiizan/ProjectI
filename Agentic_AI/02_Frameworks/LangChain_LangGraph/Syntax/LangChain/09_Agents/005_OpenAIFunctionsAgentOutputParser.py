"""
Agents: OpenAIFunctionsAgentOutputParser - Parse LLM output for tool calls.

Syntax: parser = OpenAIFunctionsAgentOutputParser()

Parses AIMessage with tool_calls into AgentAction or AgentFinish.
Used when agent uses LLM with bind_tools().
"""

from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

parser = OpenAIFunctionsAgentOutputParser()
# agent = ... | llm_with_tools | parser
# parser returns AgentAction(tool=..., tool_input=...) or AgentFinish(return_values=...)
