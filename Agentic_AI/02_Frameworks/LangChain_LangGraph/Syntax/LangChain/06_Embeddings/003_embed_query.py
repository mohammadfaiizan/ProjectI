"""
Embeddings: embed_query() - Embed single query (for search).

Syntax: vector = embeddings.embed_query(text)

Input: str - query string
Output: List[float] - single vector

Use: Embed user question for similarity search against document vectors
"""

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
query = "What is Python?"
vector = embeddings.embed_query(query)
# vector -> [0.1, -0.2, ...]
