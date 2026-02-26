"""
RAG: format_docs - Convert retrieved docs to string for prompt.

Syntax: def format_docs(docs): return "\\n\\n".join(d.page_content for d in docs)

Use in chain: retriever | format_docs -> context string
"""

from langchain_core.documents import Document


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# With sources:
def format_docs_with_sources(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source', '')}]\n{d.page_content}" for d in docs
    )
