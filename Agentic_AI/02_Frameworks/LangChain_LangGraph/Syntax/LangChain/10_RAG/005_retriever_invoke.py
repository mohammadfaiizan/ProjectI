"""
RAG: retriever.invoke() - Get similar documents for query.

Syntax: docs = retriever.invoke(query)

Input: str - user question or search query
Output: List[Document] - similar documents (by embedding)
"""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
docs = [
    Document(page_content="Python is a programming language."),
    Document(page_content="RAG combines retrieval with generation."),
]
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
results = retriever.invoke("What is Python?")  # -> List[Document]
