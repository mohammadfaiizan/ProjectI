"""
LangSmith: wrap_openai(..., name="...") - Custom trace names.

Syntax: wrap_openai(client, name="my-openai-calls")

Parameters: name, client - customize how traces appear.
"""

from openai import OpenAI
from langsmith import wrap_openai

client = wrap_openai(OpenAI(), name="production-llm")
