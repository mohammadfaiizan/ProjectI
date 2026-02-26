"""
Embeddings: embed_documents() - Embed multiple texts (for indexing).

Syntax: vectors = embeddings.embed_documents(texts)

Input: List[str] - documents to embed
Output: List[List[float]] - list of vectors
  Each vector shape: (1536,) for text-embedding-3-small
"""

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
docs = ["Python is a language.", "ML uses data."]
vectors = embeddings.embed_documents(docs)
# vectors[0] -> [0.1, -0.2, ...] (1536 dims)
