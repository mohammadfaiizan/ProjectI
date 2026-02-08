"""
Configuration module for Autonomous Web Agent.
Contains LLM configuration, browser settings, and agent parameters.
"""

from langchain_openai import ChatOpenAI
from typing import Optional
import os


class LLM_Config:
    """Configuration class for Language Model setup."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize LLM configuration.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature for model generation (0.0-2.0)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (optional, for custom endpoints)
            max_tokens: Maximum tokens to generate per response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.max_tokens = max_tokens
        
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
        
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
            
        return ChatOpenAI(**kwargs)


class Browser_Config:
    """Configuration class for web browser settings."""
    
    def __init__(
        self,
        timeout: int = 30,
        max_pages: int = 10,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        max_content_length: int = 100000,
        follow_redirects: bool = True,
        verify_ssl: bool = True
    ):
        """
        Initialize browser configuration.
        
        Args:
            timeout: Request timeout in seconds
            max_pages: Maximum number of pages to visit per task
            user_agent: User agent string for HTTP requests
            max_content_length: Maximum content length to process (characters)
            follow_redirects: Whether to follow HTTP redirects
            verify_ssl: Whether to verify SSL certificates
        """
        self.timeout = timeout
        self.max_pages = max_pages
        self.user_agent = user_agent
        self.max_content_length = max_content_length
        self.follow_redirects = follow_redirects
        self.verify_ssl = verify_ssl
    
    def Get_Timeout(self) -> int:
        """Return request timeout in seconds."""
        return self.timeout
    
    def Get_Max_Pages(self) -> int:
        """Return maximum pages to visit."""
        return self.max_pages
    
    def Get_User_Agent(self) -> str:
        """Return user agent string."""
        return self.user_agent
    
    def Get_Max_Content_Length(self) -> int:
        """Return maximum content length."""
        return self.max_content_length
    
    def Get_Follow_Redirects(self) -> bool:
        """Return whether to follow redirects."""
        return self.follow_redirects
    
    def Get_Verify_SSL(self) -> bool:
        """Return whether to verify SSL."""
        return self.verify_ssl


class Agent_Config:
    """Configuration class for agent behavior parameters."""
    
    def __init__(
        self,
        max_iterations: int = 15,
        max_depth: int = 3,
        enable_caching: bool = True,
        enable_link_following: bool = True,
        max_links_per_page: int = 5
    ):
        """
        Initialize agent configuration.
        
        Args:
            max_iterations: Maximum number of agent iterations per task
            max_depth: Maximum depth for following links (0 = no following)
            enable_caching: Whether to cache fetched pages
            enable_link_following: Whether to allow following links
            max_links_per_page: Maximum links to extract per page
        """
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.enable_caching = enable_caching
        self.enable_link_following = enable_link_following
        self.max_links_per_page = max_links_per_page
    
    def Get_Max_Iterations(self) -> int:
        """Return maximum iterations."""
        return self.max_iterations
    
    def Get_Max_Depth(self) -> int:
        """Return maximum link following depth."""
        return self.max_depth
    
    def Get_Enable_Caching(self) -> bool:
        """Return whether caching is enabled."""
        return self.enable_caching
    
    def Get_Enable_Link_Following(self) -> bool:
        """Return whether link following is enabled."""
        return self.enable_link_following
    
    def Get_Max_Links_Per_Page(self) -> int:
        """Return maximum links per page."""
        return self.max_links_per_page
