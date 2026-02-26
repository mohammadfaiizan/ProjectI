"""
Prompts: ChatPromptTemplate.from_template()

Syntax: prompt = ChatPromptTemplate.from_template("Translate: {text}")

Creates single human message. Same variable rules as PromptTemplate.

Input: dict with keys matching {placeholders}
  Example: {"text": "hello"}
"""

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
# prompt.invoke({"text": "Hello world"})
