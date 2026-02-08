"""
Configuration module for Multi-Agent Task Solver.
Defines LLM configurations, agent settings, and routing capabilities.
"""

from typing import Dict, List
from langchain_openai import ChatOpenAI


class LLM_Config:
    """Configuration for Language Models used by different agent roles."""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize LLM configuration.
        
        Args:
            api_key: OpenAI API key (or None to use environment variable)
            base_url: Optional base URL for API (for custom endpoints)
        """
        self.api_key = api_key
        self.base_url = base_url
    
    def get_supervisor_llm(self) -> ChatOpenAI:
        """Get LLM for supervisor agent (low temperature for consistent routing)."""
        return ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_research_llm(self) -> ChatOpenAI:
        """Get LLM for research specialist (moderate temperature for exploration)."""
        return ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_coding_llm(self) -> ChatOpenAI:
        """Get LLM for coding specialist (low temperature for precise code)."""
        return ChatOpenAI(
            model="gpt-4",
            temperature=0.0,
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_writing_llm(self) -> ChatOpenAI:
        """Get LLM for writing specialist (moderate temperature for creativity)."""
        return ChatOpenAI(
            model="gpt-4",
            temperature=0.5,
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_analysis_llm(self) -> ChatOpenAI:
        """Get LLM for analysis specialist (low temperature for accuracy)."""
        return ChatOpenAI(
            model="gpt-4",
            temperature=0.2,
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_aggregator_llm(self) -> ChatOpenAI:
        """Get LLM for aggregator (low temperature for consistent synthesis)."""
        return ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=self.api_key,
            base_url=self.base_url
        )


class Agent_Config:
    """Configuration for agent behavior and capabilities."""
    
    def __init__(
        self,
        max_iterations: int = 10,
        available_specialists: List[str] = None
    ):
        """
        Initialize agent configuration.
        
        Args:
            max_iterations: Maximum number of agent iterations before stopping
            available_specialists: List of available specialist agent types
        """
        self.max_iterations = max_iterations
        self.available_specialists = available_specialists or [
            "research",
            "coding",
            "writing",
            "analysis"
        ]
    
    def is_valid_specialist(self, specialist_type: str) -> bool:
        """Check if a specialist type is available."""
        return specialist_type.lower() in [
            s.lower() for s in self.available_specialists
        ]


class Routing_Config:
    """Configuration for routing tasks to appropriate specialists."""
    
    def __init__(self):
        """Initialize routing configuration with capability mappings."""
        self.capability_mapping = {
            "research": {
                "keywords": ["research", "find", "search", "investigate", "explore", "discover", "analyze trends"],
                "capabilities": [
                    "Web search and information gathering",
                    "Market research and trend analysis",
                    "Technical documentation review",
                    "Comparative analysis of options"
                ]
            },
            "coding": {
                "keywords": ["code", "program", "implement", "create script", "write code", "develop", "build", "example"],
                "capabilities": [
                    "Python code generation",
                    "Script development",
                    "Code examples and snippets",
                    "Software implementation"
                ]
            },
            "writing": {
                "keywords": ["write", "create content", "draft", "article", "report", "summary", "documentation", "explanation"],
                "capabilities": [
                    "Content writing and editing",
                    "Report generation",
                    "Documentation creation",
                    "Technical writing"
                ]
            },
            "analysis": {
                "keywords": ["analyze", "evaluate", "compare", "assess", "review", "examine", "study"],
                "capabilities": [
                    "Data analysis and interpretation",
                    "Comparative evaluation",
                    "Quality assessment",
                    "Statistical analysis"
                ]
            }
        }
    
    def get_specialist_for_task(self, task_description: str) -> str:
        """
        Determine which specialist should handle a task based on keywords.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Name of the specialist type, or "research" as default
        """
        task_lower = task_description.lower()
        scores = {}
        
        for specialist, config in self.capability_mapping.items():
            score = sum(
                1 for keyword in config["keywords"]
                if keyword in task_lower
            )
            scores[specialist] = score
        
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return "research"
    
    def get_capabilities(self, specialist_type: str) -> List[str]:
        """Get list of capabilities for a specialist type."""
        return self.capability_mapping.get(
            specialist_type.lower(),
            {}
        ).get("capabilities", [])
