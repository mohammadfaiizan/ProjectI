"""
RAG: chunk_size and chunk_overlap - TextSplitter parameters.

Syntax:
  chunk_size: int - max characters per chunk
  chunk_overlap: int - overlap between chunks (preserves context)

Larger chunk_size = more context, fewer chunks
Larger chunk_overlap = less information loss at boundaries
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # max 1000 chars per chunk
    chunk_overlap=200,  # 200 chars overlap between chunks
)
