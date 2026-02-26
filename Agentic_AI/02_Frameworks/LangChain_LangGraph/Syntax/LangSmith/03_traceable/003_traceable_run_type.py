"""
LangSmith: @traceable(run_type="...") - Run type for span.

Syntax: @traceable(run_type="tool")

Types: 'chain', 'llm', 'tool', 'retriever', 'embedding', 'prompt', 'parser'
Default: 'chain'
"""

from langsmith.run_helpers import traceable


@traceable(run_type="tool")
def my_tool(query: str) -> str:
    return f"Result for {query}"


@traceable(run_type="llm")
def llm_call(prompt: str) -> str:
    return "response"
