"""
Tools module for RAG Chatbot.
Contains document processing and vector store management utilities.
"""

from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional, Dict, Any
import os
import glob


class Document_Processor:
    """Class for loading and processing documents."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def Load_Text_Documents(self, directory: str) -> List[Document]:
        """
        Load all .txt files from a directory.
        
        Args:
            directory: Path to directory containing .txt files
            
        Returns:
            List of Document objects loaded from text files
            
        Raises:
            ValueError: If directory doesn't exist or contains no .txt files
        """
        if not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory}")
        
        txt_files = glob.glob(os.path.join(directory, "*.txt"))
        
        if not txt_files:
            raise ValueError(f"No .txt files found in directory: {directory}")
        
        documents = []
        for file_path in txt_files:
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {len(docs)} document(s) from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Total documents loaded: {len(documents)}")
        return documents
    
    def Load_Web_Page(self, url: str) -> List[Document]:
        """
        Load content from a web page.
        
        Args:
            url: URL of the web page to load
            
        Returns:
            List of Document objects loaded from the web page
            
        Raises:
            ValueError: If URL is invalid or page cannot be loaded
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL format: {url}")
        
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            print(f"Loaded {len(documents)} document(s) from {url}")
            return documents
        except Exception as e:
            raise ValueError(f"Error loading web page {url}: {e}")
    
    def Split_Documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of Document chunks
        """
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} document(s) into {len(chunks)} chunks")
        return chunks


class Vector_Store_Manager:
    """Class for managing ChromaDB vector store operations."""
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embeddings: Embeddings
    ):
        """
        Initialize vector store manager.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embeddings: Embeddings model to use
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = embeddings
        self.vector_store: Optional[Chroma] = None
    
    def Create_Store(
        self,
        documents: List[Document],
        collection_name: Optional[str] = None
    ) -> Chroma:
        """
        Create a new ChromaDB vector store from documents.
        
        Args:
            documents: List of Document objects to add to the store
            collection_name: Optional collection name override
            
        Returns:
            Chroma vector store instance
            
        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Cannot create vector store with empty documents list")
        
        collection = collection_name or self.collection_name
        
        # Remove existing collection if it exists
        if os.path.exists(self.persist_directory):
            try:
                existing_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=collection
                )
                existing_store.delete_collection()
            except Exception:
                pass
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection,
            persist_directory=self.persist_directory
        )
        
        print(f"Created vector store with {len(documents)} documents")
        return self.vector_store
    
    def Get_Retriever(
        self,
        search_kwargs: Optional[Dict[str, Any]] = None
    ) -> BaseRetriever:
        """
        Get a retriever from the vector store.
        
        Args:
            search_kwargs: Optional search parameters (e.g., {"k": 4})
            
        Returns:
            BaseRetriever instance
            
        Raises:
            ValueError: If vector store hasn't been created yet
        """
        if self.vector_store is None:
            # Try to load existing store
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
            except Exception:
                raise ValueError(
                    "Vector store not initialized. Call Create_Store() first."
                )
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        return retriever
    
    def Add_Documents(self, documents: List[Document]) -> None:
        """
        Add more documents to the existing vector store.
        
        Args:
            documents: List of Document objects to add
            
        Raises:
            ValueError: If vector store hasn't been created yet
        """
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. Call Create_Store() first."
            )
        
        if not documents:
            print("No documents to add")
            return
        
        self.vector_store.add_documents(documents)
        print(f"Added {len(documents)} documents to vector store")
    
    def Clear_Store(self) -> None:
        """
        Delete the collection from the vector store.
        
        Raises:
            ValueError: If vector store hasn't been created yet
        """
        if self.vector_store is None:
            try:
                temp_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                temp_store.delete_collection()
                print("Collection deleted")
            except Exception:
                print("No collection found to delete")
            return
        
        try:
            self.vector_store.delete_collection()
            self.vector_store = None
            print("Collection deleted successfully")
        except Exception as e:
            print(f"Error deleting collection: {e}")
