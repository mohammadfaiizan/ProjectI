"""
Prompts: MessagesPlaceholder - Placeholder for conversation history.

Syntax: MessagesPlaceholder(variable_name="chat_history")

Input: dict must include "chat_history": List[BaseMessage]
  Use for multi-turn chat with memory
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])
# prompt.invoke({"chat_history": [], "input": "Hello"})
# prompt.invoke({"chat_history": [HumanMessage(...), AIMessage(...)], "input": "What did I say?"})
