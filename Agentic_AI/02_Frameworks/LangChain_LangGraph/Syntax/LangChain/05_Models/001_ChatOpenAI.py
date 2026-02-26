"""
Models: ChatOpenAI - OpenAI chat model interface.

Syntax: llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

Parameters:
  model: str - gpt-3.5-turbo, gpt-4, etc.
  temperature: float [0, 2] - 0=deterministic, higher=creative
  api_key: str | None - defaults to OPENAI_API_KEY env
"""

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# llm.invoke("Hello")
# llm.stream("Hello")
# llm.batch(["a", "b"])
