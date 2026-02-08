"""
Configuration module for Content Generation Pipeline.
Contains LLM configuration, content settings, SEO parameters, and quality thresholds.
"""

from langchain_openai import ChatOpenAI
from typing import Optional, List, Dict
import os


class LLM_Config:
    """Configuration class for Language Model setup."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
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
            max_tokens: Maximum tokens to generate (optional)
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


class Content_Config:
    """Configuration class for content generation parameters."""
    
    def __init__(
        self,
        max_word_count: int = 2000,
        target_audience_levels: List[str] = None,
        tone_options: List[str] = None
    ):
        """
        Initialize content configuration.
        
        Args:
            max_word_count: Maximum word count for generated content
            target_audience_levels: List of available audience levels
            tone_options: List of available tone options
        """
        self.max_word_count = max_word_count
        self.target_audience_levels = target_audience_levels or [
            "beginners",
            "intermediate",
            "advanced",
            "developers",
            "business leaders",
            "general audience"
        ]
        self.tone_options = tone_options or [
            "professional",
            "friendly",
            "technical",
            "casual",
            "formal",
            "conversational"
        ]
    
    def Get_Max_Word_Count(self) -> int:
        """Return maximum word count."""
        return self.max_word_count
    
    def Get_Target_Audience_Levels(self) -> List[str]:
        """Return list of target audience levels."""
        return self.target_audience_levels
    
    def Get_Tone_Options(self) -> List[str]:
        """Return list of tone options."""
        return self.tone_options
    
    def Validate_Audience(self, audience: str) -> bool:
        """Validate if audience is in allowed list."""
        return audience.lower() in [a.lower() for a in self.target_audience_levels]
    
    def Validate_Tone(self, tone: str) -> bool:
        """Validate if tone is in allowed list."""
        return tone.lower() in [t.lower() for t in self.tone_options]


class SEO_Config:
    """Configuration class for SEO optimization parameters."""
    
    def __init__(
        self,
        min_keyword_density: float = 0.01,
        max_title_length: int = 60,
        meta_description_length: int = 160
    ):
        """
        Initialize SEO configuration.
        
        Args:
            min_keyword_density: Minimum keyword density (0.0-1.0)
            max_title_length: Maximum title length in characters
            meta_description_length: Meta description length in characters
        """
        self.min_keyword_density = min_keyword_density
        self.max_title_length = max_title_length
        self.meta_description_length = meta_description_length
    
    def Get_Min_Keyword_Density(self) -> float:
        """Return minimum keyword density."""
        return self.min_keyword_density
    
    def Get_Max_Title_Length(self) -> int:
        """Return maximum title length."""
        return self.max_title_length
    
    def Get_Meta_Description_Length(self) -> int:
        """Return meta description length."""
        return self.meta_description_length


class Quality_Config:
    """Configuration class for quality control parameters."""
    
    def __init__(
        self,
        min_quality_score: float = 0.7,
        max_revision_rounds: int = 3
    ):
        """
        Initialize quality configuration.
        
        Args:
            min_quality_score: Minimum quality score to pass (0.0-1.0)
            max_revision_rounds: Maximum number of revision rounds allowed
        """
        if not 0.0 <= min_quality_score <= 1.0:
            raise ValueError("min_quality_score must be between 0.0 and 1.0")
        
        if max_revision_rounds < 1:
            raise ValueError("max_revision_rounds must be at least 1")
        
        self.min_quality_score = min_quality_score
        self.max_revision_rounds = max_revision_rounds
    
    def Get_Min_Quality_Score(self) -> float:
        """Return minimum quality score threshold."""
        return self.min_quality_score
    
    def Get_Max_Revision_Rounds(self) -> int:
        """Return maximum revision rounds."""
        return self.max_revision_rounds
    
    def Check_Quality_Passed(self, quality_score: float) -> bool:
        """Check if quality score meets threshold."""
        return quality_score >= self.min_quality_score
    
    def Check_Can_Revise(self, revision_count: int) -> bool:
        """Check if more revisions are allowed."""
        return revision_count < self.max_revision_rounds
