"""
Prompts: ChatPromptTemplate.from_messages()

Syntax: prompt = ChatPromptTemplate.from_messages([
    ("system", "You are..."),
    ("human", "{input}"),
    ("ai", "response"),  # for few-shot
])

Input: dict with keys matching placeholders in messages
  ("human", "{key}") -> input must have "key"
  ("system", "fixed") -> no input needed
"""

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])
# prompt.invoke({"question": "What is 2+2?"})
# prompt.format_messages(question="What is 2+2?")
