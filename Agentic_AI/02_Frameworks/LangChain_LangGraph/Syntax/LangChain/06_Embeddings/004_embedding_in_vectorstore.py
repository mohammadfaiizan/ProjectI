"""
Embeddings: Use with VectorStore - VectorStore.from_documents()

Syntax: vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

VectorStore uses embed_documents internally when indexing.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
docs = [Document(page_content="Python is great.", metadata={"id": 1})]
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
