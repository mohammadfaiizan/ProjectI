"""
Models: llm.invoke() - Synchronous call.

Syntax: response = llm.invoke(messages)

Input: str or List[BaseMessage]
  - str -> converted to HumanMessage
  - [SystemMessage(...), HumanMessage(...)] for chat

Output: AIMessage (response.content for text)
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
response = llm.invoke("Say hi in 3 words")
# response.content -> str

response2 = llm.invoke([
    SystemMessage(content="You are helpful."),
    HumanMessage(content="What is 2+2?"),
])
