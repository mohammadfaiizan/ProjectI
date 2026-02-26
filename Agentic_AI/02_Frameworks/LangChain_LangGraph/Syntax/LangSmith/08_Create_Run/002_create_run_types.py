"""
LangSmith: create_run run_type - Run type values.

Syntax: run_type="chain" | "llm" | "tool" | "retriever" | "embedding" | "prompt" | "parser"

Determines how run appears in LangSmith UI.
"""

from langsmith import Client

client = Client()
# client.create_run(name="search", inputs={}, run_type="retriever")
# client.create_run(name="embed", inputs={}, run_type="embedding")
