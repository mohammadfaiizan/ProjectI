"""
LangSmith: wrap_openai() - Add tracing to OpenAI client.

Syntax: from langsmith import wrap_openai
        client = wrap_openai(OpenAI())

Patches OpenAI client to log traces. Call before using client.
"""

from openai import OpenAI
from langsmith import wrap_openai

client = wrap_openai(OpenAI())
# All client.chat.completions.create() calls are traced
