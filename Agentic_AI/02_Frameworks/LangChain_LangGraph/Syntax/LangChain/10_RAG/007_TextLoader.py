"""
RAG: TextLoader - Load documents from file.

Syntax: loader = TextLoader("path/to/file.txt")
        docs = loader.load()

Output: List[Document]
  page_content = file content
  metadata = {"source": path}
"""

from langchain_community.document_loaders import TextLoader

# loader = TextLoader("file.txt")
# docs = loader.load()
