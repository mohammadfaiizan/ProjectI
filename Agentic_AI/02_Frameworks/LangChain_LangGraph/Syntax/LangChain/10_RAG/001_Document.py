"""
RAG: Document - Text content with metadata.

Syntax: doc = Document(page_content="...", metadata={...})

  page_content: str - the text
  metadata: dict - optional (source, id, etc.)
"""

from langchain_core.documents import Document

doc = Document(
    page_content="LangChain is a framework for LLM applications.",
    metadata={"source": "intro.md", "page": 1},
)
