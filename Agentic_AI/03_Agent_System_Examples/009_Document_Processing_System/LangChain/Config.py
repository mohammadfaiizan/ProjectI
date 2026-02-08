"""
Configuration module for Document Processing System.
Contains LLM configuration, processing settings, classification categories,
and storage configuration.
"""

from langchain_openai import ChatOpenAI
from typing import Optional, List
import os


class LLM_Config:
    """Configuration class for Language Model setup."""
    
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


class Processing_Config:
    """Configuration class for document processing settings."""
    
    def __init__(
        self,
        supported_document_types: Optional[List[str]] = None,
        max_document_length: int = 50000
    ):
        """
        Initialize processing configuration.
        
        Args:
            supported_document_types: List of supported document type extensions
            max_document_length: Maximum character length for documents
        """
        self.supported_document_types = supported_document_types or [
            ".txt", ".md", ".pdf", ".docx", ".html"
        ]
        self.max_document_length = max_document_length
    
    def Get_Supported_Types(self) -> List[str]:
        """Return list of supported document types."""
        return self.supported_document_types
    
    def Get_Max_Length(self) -> int:
        """Return maximum document length."""
        return self.max_document_length
    
    def Is_Supported_Type(self, filename: str) -> bool:
        """
        Check if file type is supported.
        
        Args:
            filename: Name of the file to check
            
        Returns:
            True if file type is supported, False otherwise
        """
        file_ext = os.path.splitext(filename)[1].lower()
        return file_ext in self.supported_document_types


class Classification_Config:
    """Configuration class for document classification categories."""
    
    def __init__(
        self,
        document_categories: Optional[List[str]] = None
    ):
        """
        Initialize classification configuration.
        
        Args:
            document_categories: List of document category names
        """
        self.document_categories = document_categories or [
            "invoice",
            "resume",
            "contract",
            "letter",
            "report"
        ]
    
    def Get_Categories(self) -> List[str]:
        """Return list of document categories."""
        return self.document_categories
    
    def Is_Valid_Category(self, category: str) -> bool:
        """
        Check if category is valid.
        
        Args:
            category: Category name to validate
            
        Returns:
            True if category is valid, False otherwise
        """
        return category.lower() in [cat.lower() for cat in self.document_categories]


class Storage_Config:
    """Configuration class for document storage settings."""
    
    def __init__(
        self,
        output_directory: str = "./processed_documents",
        storage_format: str = "json"
    ):
        """
        Initialize storage configuration.
        
        Args:
            output_directory: Directory path for storing processed documents
            storage_format: Format for storing documents (json, yaml, xml)
        """
        self.output_directory = output_directory
        self.storage_format = storage_format.lower()
        
        # Validate storage format
        valid_formats = ["json", "yaml", "xml"]
        if self.storage_format not in valid_formats:
            raise ValueError(
                f"Invalid storage format: {storage_format}. "
                f"Must be one of: {', '.join(valid_formats)}"
            )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)
    
    def Get_Output_Directory(self) -> str:
        """Return output directory path."""
        return self.output_directory
    
    def Get_Storage_Format(self) -> str:
        """Return storage format."""
        return self.storage_format
