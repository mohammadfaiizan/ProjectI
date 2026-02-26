"""
Embeddings: OpenAIEmbeddings - Text to vector model.

Syntax: embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

Parameters:
  model: str - text-embedding-3-small, text-embedding-3-large, etc.
  api_key: str | None - defaults to OPENAI_API_KEY
"""

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
