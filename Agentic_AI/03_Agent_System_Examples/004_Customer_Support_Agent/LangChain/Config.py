"""
Configuration module for Customer Support Agent.

This module contains all configuration classes and settings for the LangChain-based
customer support system, including LLM configuration, support system settings,
and FAQ data store configuration.
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI


class LLM_Config:
    """
    Configuration class for Language Model setup.
    
    Handles initialization of ChatOpenAI with configurable parameters
    for model selection, temperature, and API key management.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize LLM configuration.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Sampling temperature (0.0 for deterministic)
            api_key: OpenAI API key (if None, reads from environment)
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    def Get_LLM(self) -> ChatOpenAI:
        """
        Create and return a configured ChatOpenAI instance.
        
        Returns:
            Configured ChatOpenAI instance
        """
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            max_tokens=self.max_tokens
        )


class Support_Config:
    """
    Configuration class for Customer Support System settings.
    
    Contains company-specific settings, escalation thresholds,
    and conversation management parameters.
    """
    
    def __init__(
        self,
        company_name: str = "TechStore",
        escalation_threshold: int = 3,
        max_conversation_turns: int = 10,
        auto_escalate_complaints: bool = True,
        support_hours: str = "9 AM - 6 PM EST"
    ):
        """
        Initialize support system configuration.
        
        Args:
            company_name: Name of the company providing support
            escalation_threshold: Number of failed attempts before escalation
            max_conversation_turns: Maximum turns before forcing escalation
            auto_escalate_complaints: Whether to auto-escalate complaint intents
            support_hours: Business hours information
        """
        self.company_name = company_name
        self.escalation_threshold = escalation_threshold
        self.max_conversation_turns = max_conversation_turns
        self.auto_escalate_complaints = auto_escalate_complaints
        self.support_hours = support_hours
    
    def Get_Company_Name(self) -> str:
        """Return the company name."""
        return self.company_name
    
    def Get_Escalation_Threshold(self) -> int:
        """Return the escalation threshold."""
        return self.escalation_threshold
    
    def Get_Max_Conversation_Turns(self) -> int:
        """Return maximum conversation turns."""
        return self.max_conversation_turns
    
    def Should_Auto_Escalate_Complaints(self) -> bool:
        """Return whether complaints should be auto-escalated."""
        return self.auto_escalate_complaints
    
    def Get_Support_Hours(self) -> str:
        """Return support hours information."""
        return self.support_hours


class FAQ_Config:
    """
    Configuration class for FAQ data store settings.
    
    Handles configuration for FAQ storage, retrieval, and vector store setup.
    """
    
    def __init__(
        self,
        use_vector_store: bool = True,
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.7,
        max_faq_results: int = 3,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize FAQ configuration.
        
        Args:
            use_vector_store: Whether to use vector store for semantic search
            embedding_model: Model name for embeddings
            similarity_threshold: Minimum similarity score for FAQ retrieval
            max_faq_results: Maximum number of FAQ results to return
            persist_directory: Directory to persist vector store (None for in-memory)
        """
        self.use_vector_store = use_vector_store
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_faq_results = max_faq_results
        self.persist_directory = persist_directory
    
    def Get_Use_Vector_Store(self) -> bool:
        """Return whether vector store is enabled."""
        return self.use_vector_store
    
    def Get_Embedding_Model(self) -> str:
        """Return embedding model name."""
        return self.embedding_model
    
    def Get_Similarity_Threshold(self) -> float:
        """Return similarity threshold for FAQ retrieval."""
        return self.similarity_threshold
    
    def Get_Max_FAQ_Results(self) -> int:
        """Return maximum FAQ results to return."""
        return self.max_faq_results
    
    def Get_Persist_Directory(self) -> Optional[str]:
        """Return persist directory for vector store."""
        return self.persist_directory
