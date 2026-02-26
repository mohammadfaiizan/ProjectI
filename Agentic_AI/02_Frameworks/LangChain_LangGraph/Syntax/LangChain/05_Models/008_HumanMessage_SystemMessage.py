"""
Models: HumanMessage, SystemMessage, AIMessage - Message types.

Syntax:
  HumanMessage(content="user input")
  SystemMessage(content="You are...")
  AIMessage(content="assistant response")

Pass to llm.invoke([SystemMessage(...), HumanMessage(...)])
"""

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

system = SystemMessage(content="You are a helpful assistant.")
human = HumanMessage(content="What is 2+2?")
ai = AIMessage(content="4")

# llm.invoke([system, human])
