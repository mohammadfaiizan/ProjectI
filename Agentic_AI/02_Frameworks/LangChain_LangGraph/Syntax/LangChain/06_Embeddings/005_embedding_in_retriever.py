"""
Embeddings: Retriever uses embed_query internally.

Syntax: retriever = vectorstore.as_retriever()
        results = retriever.invoke("query")

When you invoke retriever with a string, it embeds the query via embed_query
and compares to stored document vectors.
"""

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
docs = [Document(page_content="Python is a programming language.")]
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
# retriever.invoke("programming") -> embeds query, returns similar docs
