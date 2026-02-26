"""
RAG: RecursiveCharacterTextSplitter - Split documents into chunks.

Syntax: splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)

Input: List[Document]
Output: List[Document] (smaller chunks)
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)
docs = [Document(page_content="Long text here...")]
chunks = splitter.split_documents(docs)
