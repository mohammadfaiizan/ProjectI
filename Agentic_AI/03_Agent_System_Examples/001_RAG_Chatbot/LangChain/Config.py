"""
Configuration module for RAG Chatbot.
Contains LLM configuration, vector store settings, and constants.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Optional
import os


class LLM_Config:
    """Configuration class for Language Model and Embeddings setup."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize LLM configuration.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature for model generation (0.0 for deterministic)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (optional, for custom endpoints)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    def Get_LLM(self) -> ChatOpenAI:
        """
        Create and return ChatOpenAI instance.
        
        Returns:
            Configured ChatOpenAI instance
        """
        kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "api_key": self.api_key
        }
        
        if self.base_url:
            kwargs["base_url"] = self.base_url
            
        return ChatOpenAI(**kwargs)
    
    def Get_Embeddings(self) -> OpenAIEmbeddings:
        """
        Create and return OpenAIEmbeddings instance.
        
        Returns:
            Configured OpenAIEmbeddings instance
        """
        kwargs = {
            "api_key": self.api_key
        }
        
        if self.base_url:
            kwargs["base_url"] = self.base_url
            
        return OpenAIEmbeddings(**kwargs)


class Vector_Store_Config:
    """Configuration class for ChromaDB vector store settings."""
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        distance_metric: str = "cosine"
    ):
        """
        Initialize vector store configuration.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            distance_metric: Distance metric for similarity search
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def Get_Collection_Name(self) -> str:
        """Return the collection name."""
        return self.collection_name
    
    def Get_Persist_Directory(self) -> str:
        """Return the persist directory path."""
        return self.persist_directory
    
    def Get_Distance_Metric(self) -> str:
        """Return the distance metric."""
        return self.distance_metric


# Constants for document processing and retrieval
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
