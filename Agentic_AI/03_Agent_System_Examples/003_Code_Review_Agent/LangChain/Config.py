"""
Configuration module for Code Review Agent.
Contains LLM configuration, review settings, and severity levels.
"""

from langchain_openai import ChatOpenAI
from typing import Optional, Dict, List
from enum import Enum
import os


class Severity(Enum):
    """Severity levels for code review issues."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


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
            temperature: Temperature for model generation (0.0 for deterministic reviews)
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


class Review_Config:
    """Configuration class for code review settings."""
    
    def __init__(
        self,
        severity_levels: Optional[Dict[str, int]] = None,
        max_issues_per_category: Optional[Dict[str, int]] = None,
        enabled_checks: Optional[List[str]] = None
    ):
        """
        Initialize review configuration.
        
        Args:
            severity_levels: Dictionary mapping severity names to numeric values
            max_issues_per_category: Maximum issues to report per category
            enabled_checks: List of enabled check categories
        """
        self.severity_levels = severity_levels or {
            "CRITICAL": 5,
            "HIGH": 4,
            "MEDIUM": 3,
            "LOW": 2,
            "INFO": 1
        }
        
        self.max_issues_per_category = max_issues_per_category or {
            "bug_issues": 20,
            "security_issues": 15,
            "style_issues": 25,
            "performance_issues": 15
        }
        
        self.enabled_checks = enabled_checks or [
            "bugs",
            "security",
            "style",
            "performance"
        ]
    
    def Get_Severity_Value(self, severity: str) -> int:
        """
        Get numeric value for a severity level.
        
        Args:
            severity: Severity level name
            
        Returns:
            Numeric severity value
        """
        return self.severity_levels.get(severity.upper(), 0)
    
    def Get_Max_Issues(self, category: str) -> int:
        """
        Get maximum issues allowed for a category.
        
        Args:
            category: Issue category name
            
        Returns:
            Maximum number of issues
        """
        return self.max_issues_per_category.get(category, 20)
    
    def Is_Check_Enabled(self, check_name: str) -> bool:
        """
        Check if a specific check category is enabled.
        
        Args:
            check_name: Name of the check category
            
        Returns:
            True if enabled, False otherwise
        """
        return check_name.lower() in [c.lower() for c in self.enabled_checks]


# Severity constants for easy access
SEVERITY_CRITICAL = Severity.CRITICAL.value
SEVERITY_HIGH = Severity.HIGH.value
SEVERITY_MEDIUM = Severity.MEDIUM.value
SEVERITY_LOW = Severity.LOW.value
SEVERITY_INFO = Severity.INFO.value
