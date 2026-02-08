"""
Configuration module for Research Assistant system.
Defines configuration classes for LLM, search, and report generation.
"""

from dataclasses import dataclass
from typing import Optional
from langchain_openai import ChatOpenAI


@dataclass
class LLM_Config:
    """
    Configuration for Language Model setup.
    Manages ChatOpenAI instance creation and parameters.
    """
    
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    def __post_init__(self):
        """Initialize ChatOpenAI instance after dataclass creation."""
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_llm(self) -> ChatOpenAI:
        """Get the configured ChatOpenAI instance."""
        return self.llm


@dataclass
class Search_Config:
    """
    Configuration for web search operations.
    Controls search result limits and provider settings.
    """
    
    max_results_per_query: int = 5
    max_total_results: int = 20
    search_provider: str = "mock"
    timeout_seconds: int = 30
    enable_cache: bool = True
    min_snippet_length: int = 50
    max_snippet_length: int = 500
    
    def validate(self) -> bool:
        """Validate search configuration parameters."""
        if self.max_results_per_query <= 0:
            raise ValueError("max_results_per_query must be positive")
        if self.max_total_results <= 0:
            raise ValueError("max_total_results must be positive")
        if self.max_results_per_query > self.max_total_results:
            raise ValueError("max_results_per_query cannot exceed max_total_results")
        return True


@dataclass
class Report_Config:
    """
    Configuration for report generation.
    Controls report structure, sections, and citation formatting.
    """
    
    max_sections: int = 8
    min_sections: int = 3
    citation_style: str = "APA"
    include_executive_summary: bool = True
    include_methodology: bool = True
    include_conclusion: bool = True
    max_words_per_section: int = 500
    min_sources_per_section: int = 2
    enable_markdown_formatting: bool = True
    
    def get_citation_formatter(self):
        """Get citation formatter based on citation_style."""
        citation_formatters = {
            "APA": self._format_apa_citation,
            "MLA": self._format_mla_citation,
            "Chicago": self._format_chicago_citation
        }
        return citation_formatters.get(self.citation_style, self._format_apa_citation)
    
    def _format_apa_citation(self, author: str, title: str, url: str, date: str = "") -> str:
        """Format citation in APA style."""
        if date:
            return f"{author} ({date}). {title}. Retrieved from {url}"
        return f"{author}. {title}. Retrieved from {url}"
    
    def _format_mla_citation(self, author: str, title: str, url: str, date: str = "") -> str:
        """Format citation in MLA style."""
        if date:
            return f'"{title}." {author}, {date}, {url}.'
        return f'"{title}." {author}, {url}.'
    
    def _format_chicago_citation(self, author: str, title: str, url: str, date: str = "") -> str:
        """Format citation in Chicago style."""
        if date:
            return f"{author}. \"{title}.\" Accessed {date}. {url}."
        return f"{author}. \"{title}.\" {url}."
    
    def validate(self) -> bool:
        """Validate report configuration parameters."""
        if self.max_sections < self.min_sections:
            raise ValueError("max_sections cannot be less than min_sections")
        if self.citation_style not in ["APA", "MLA", "Chicago"]:
            raise ValueError("citation_style must be one of: APA, MLA, Chicago")
        return True
