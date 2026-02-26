"""
RAG: vectorstore.as_retriever() - Create retriever from vector store.

Syntax: retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

  search_kwargs: k = number of docs to retrieve
  retriever.invoke(query: str) -> List[Document]
"""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
docs = [Document(page_content="Python is a language.")]
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
# retriever.invoke("programming") -> [Document, ...]
