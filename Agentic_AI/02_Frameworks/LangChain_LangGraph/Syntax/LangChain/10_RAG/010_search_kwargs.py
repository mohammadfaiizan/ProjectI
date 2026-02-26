"""
RAG: search_kwargs - Retriever search parameters.

Syntax: retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

  k: int - number of documents to retrieve (default varies by store)
  score_threshold: float - min similarity (some stores)
  filter: dict - metadata filter (some stores)
"""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
docs = [Document(page_content="Text")]
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
