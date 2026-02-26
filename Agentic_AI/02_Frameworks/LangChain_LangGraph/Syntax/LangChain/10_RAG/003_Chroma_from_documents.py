"""
RAG: Chroma.from_documents() - Create vector store from documents.

Syntax: vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="my_collection",
)

Input: documents (List[Document]), embedding (Embeddings)
Output: VectorStore
"""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
docs = [Document(page_content="Python is great.", metadata={"id": 1})]
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="rag_demo",
)
