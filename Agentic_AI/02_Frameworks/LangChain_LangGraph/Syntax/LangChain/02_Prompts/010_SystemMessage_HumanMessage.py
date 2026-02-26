"""
Prompts: SystemMessage, HumanMessage, AIMessage - Message types.

Syntax:
  from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
  SystemMessage(content="You are helpful.")
  HumanMessage(content="User input")
  AIMessage(content="Assistant response")

Use in ChatPromptTemplate: ("system", "text") or ("human", "{var}") or ("ai", "text")
Or pass directly: [SystemMessage(...), HumanMessage(...)]
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

system_msg = SystemMessage(content="You are a helpful assistant.")
human_msg = HumanMessage(content="What is 2+2?")
ai_msg = AIMessage(content="4")

# For model: llm.invoke([system_msg, human_msg])
