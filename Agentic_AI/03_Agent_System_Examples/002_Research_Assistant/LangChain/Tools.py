"""
Tools module for Research Assistant system.
Provides web search, content fetching, summarization, and citation tracking tools.
"""

import random
import time
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Mock search results database for demonstration
MOCK_SEARCH_DATABASE = {
    "ai agents software development": [
        {
            "title": "AI Agents Revolutionize Software Development Workflows",
            "url": "https://example.com/ai-agents-dev",
            "snippet": "AI agents are transforming software development by automating code generation, testing, and deployment processes. Recent studies show 40% productivity gains in development teams using AI-powered tools."
        },
        {
            "title": "The Future of Coding: Autonomous AI Development Agents",
            "url": "https://example.com/future-coding",
            "snippet": "Autonomous AI agents can now write, test, and deploy code independently. This paradigm shift enables developers to focus on high-level architecture while agents handle routine implementation tasks."
        },
        {
            "title": "Multi-Agent Systems in Software Engineering",
            "url": "https://example.com/multi-agent-se",
            "snippet": "Multi-agent systems coordinate multiple AI agents to solve complex software engineering problems. Each agent specializes in different aspects like code review, bug detection, or documentation generation."
        }
    ],
    "rag vs fine-tuning enterprise": [
        {
            "title": "RAG vs Fine-Tuning: Choosing the Right Approach for Enterprise LLMs",
            "url": "https://example.com/rag-vs-finetuning",
            "snippet": "Retrieval-Augmented Generation (RAG) provides real-time knowledge access without model retraining, while fine-tuning adapts models to specific domains. Enterprises must evaluate cost, latency, and accuracy trade-offs."
        },
        {
            "title": "Enterprise LLM Deployment Strategies: A Comparative Analysis",
            "url": "https://example.com/enterprise-llm",
            "snippet": "RAG offers flexibility and up-to-date information but requires robust retrieval systems. Fine-tuning provides domain-specific accuracy but needs continuous retraining as knowledge evolves."
        },
        {
            "title": "Cost-Benefit Analysis: RAG and Fine-Tuning for Business Applications",
            "url": "https://example.com/cost-analysis",
            "snippet": "Fine-tuning costs scale with model size and training frequency. RAG systems have lower upfront costs but require infrastructure for vector databases and retrieval pipelines."
        }
    ],
    "multi-agent autonomous driving": [
        {
            "title": "Multi-Agent Reinforcement Learning for Autonomous Vehicles",
            "url": "https://example.com/marl-autonomous",
            "snippet": "Multi-agent systems enable autonomous vehicles to coordinate with other vehicles, pedestrians, and infrastructure. Each agent handles perception, planning, or control tasks independently."
        },
        {
            "title": "Cooperative Multi-Agent Systems in Self-Driving Cars",
            "url": "https://example.com/cooperative-agents",
            "snippet": "Cooperative agents in autonomous driving share information to improve collective decision-making. This approach enhances safety and efficiency in complex traffic scenarios."
        },
        {
            "title": "Distributed Decision-Making in Autonomous Vehicle Fleets",
            "url": "https://example.com/distributed-driving",
            "snippet": "Fleets of autonomous vehicles use multi-agent systems to optimize routes, reduce congestion, and coordinate maneuvers. Each vehicle acts as an independent agent with shared communication protocols."
        }
    ]
}


@tool
def Search_Web(query: str) -> List[Dict[str, str]]:
    """
    Search the web for information related to the query.
    
    Args:
        query: Search query string
        
    Returns:
        List of search results, each containing title, url, and snippet
    """
    query_lower = query.lower()
    
    # Simulate API delay
    time.sleep(0.5)
    
    # Try to find matching results in mock database
    results = []
    for key, value in MOCK_SEARCH_DATABASE.items():
        if any(word in query_lower for word in key.split()):
            results.extend(value)
    
    # Generate generic results if no matches found
    if not results:
        results = [
            {
                "title": f"Research Article: {query}",
                "url": f"https://example.com/research/{query.replace(' ', '-')}",
                "snippet": f"This article discusses various aspects of {query}, including recent developments, key findings, and future implications in the field."
            },
            {
                "title": f"Comprehensive Guide to {query}",
                "url": f"https://example.com/guide/{query.replace(' ', '-')}",
                "snippet": f"A detailed guide covering fundamental concepts, practical applications, and advanced topics related to {query}."
            },
            {
                "title": f"Latest Trends in {query}",
                "url": f"https://example.com/trends/{query.replace(' ', '-')}",
                "snippet": f"Recent trends and innovations in {query}, including industry insights, expert opinions, and emerging technologies."
            }
        ]
    
    # Limit results and add random variation
    results = results[:5]
    random.shuffle(results)
    
    return results


@tool
def Fetch_URL_Content(url: str) -> Dict[str, Any]:
    """
    Fetch and extract content from a given URL.
    
    Args:
        url: URL to fetch content from
        
    Returns:
        Dictionary containing title, content, author, and metadata
    """
    # Simulate network delay
    time.sleep(0.3)
    
    # Extract domain and path for mock content generation
    domain = url.split("//")[-1].split("/")[0] if "//" in url else url.split("/")[0]
    path = url.split("/")[-1] if "/" in url else "article"
    
    # Generate mock article content
    mock_content = f"""
    Article Title: Comprehensive Analysis of {path.replace('-', ' ').title()}
    
    Introduction:
    This article provides an in-depth examination of the topic, covering historical context,
    current state of research, and future directions. The analysis draws from multiple
    authoritative sources and recent publications in the field.
    
    Main Content:
    The subject matter encompasses several key areas that require careful consideration.
    First, we examine the foundational principles and theoretical frameworks that underpin
    current understanding. Second, we analyze practical applications and real-world
    implementations. Third, we discuss challenges, limitations, and areas for future research.
    
    Key Findings:
    Research indicates significant developments in this area, with implications for both
    theoretical understanding and practical applications. Studies have shown consistent
    patterns across multiple domains, suggesting robust underlying mechanisms.
    
    Conclusion:
    The field continues to evolve rapidly, with new insights emerging from interdisciplinary
    research. Future work should focus on addressing current limitations and exploring
    novel applications of these concepts.
    """
    
    return {
        "title": f"Article: {path.replace('-', ' ').title()}",
        "content": mock_content.strip(),
        "author": f"Research Team at {domain}",
        "date": "2024",
        "url": url,
        "word_count": len(mock_content.split())
    }


@tool
def Summarize_Text(text: str, llm: Optional[ChatOpenAI] = None) -> str:
    """
    Summarize a given text using an LLM.
    
    Args:
        text: Text to summarize
        llm: Optional ChatOpenAI instance (will create default if not provided)
        
    Returns:
        Summarized text
    """
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Provide concise, informative summaries that capture key points and main findings."),
        ("human", "Summarize the following text, focusing on main ideas, key findings, and important conclusions:\n\n{text}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"text": text[:3000]})  # Limit input length
    
    return response.content


class Citation_Tracker:
    """
    Tracks sources and generates formatted citations and bibliographies.
    Manages citation numbering and formatting according to specified style.
    """
    
    def __init__(self, citation_style: str = "APA"):
        """
        Initialize citation tracker.
        
        Args:
            citation_style: Citation style (APA, MLA, or Chicago)
        """
        self.citation_style = citation_style
        self.sources: List[Dict[str, Any]] = []
        self.citation_map: Dict[str, int] = {}  # Maps URL to citation number
    
    def add_source(self, title: str, url: str, author: str = "Unknown", date: str = "") -> int:
        """
        Add a source and return its citation number.
        
        Args:
            title: Source title
            url: Source URL
            author: Author name
            date: Publication date
            
        Returns:
            Citation number for this source
        """
        if url in self.citation_map:
            return self.citation_map[url]
        
        citation_num = len(self.sources) + 1
        source = {
            "number": citation_num,
            "title": title,
            "url": url,
            "author": author,
            "date": date
        }
        self.sources.append(source)
        self.citation_map[url] = citation_num
        return citation_num
    
    def format_citation(self, url: str) -> str:
        """
        Format an inline citation for a given URL.
        
        Args:
            url: Source URL
            
        Returns:
            Formatted citation string
        """
        if url not in self.citation_map:
            return "[Citation not found]"
        
        citation_num = self.citation_map[url]
        source = self.sources[citation_num - 1]
        
        if self.citation_style == "APA":
            return f"({source['author']}, {source['date'] or 'n.d.'})"
        elif self.citation_style == "MLA":
            return f"({source['author']} {citation_num})"
        else:  # Chicago
            return f"[{citation_num}]"
    
    def generate_bibliography(self) -> str:
        """
        Generate formatted bibliography of all sources.
        
        Returns:
            Formatted bibliography string
        """
        if not self.sources:
            return "No sources cited."
        
        bibliography_lines = ["\n## References\n"]
        
        for source in self.sources:
            if self.citation_style == "APA":
                citation = f"{source['number']}. {source['author']}. ({source['date'] or 'n.d.'}). {source['title']}. Retrieved from {source['url']}"
            elif self.citation_style == "MLA":
                citation = f"{source['number']}. {source['author']}. \"{source['title']}.\" {source['date'] or 'n.d.'}, {source['url']}."
            else:  # Chicago
                citation = f"{source['number']}. {source['author']}. \"{source['title']}.\" Accessed {source['date'] or 'n.d.'}. {source['url']}."
            
            bibliography_lines.append(citation)
        
        return "\n".join(bibliography_lines)
    
    def get_source_count(self) -> int:
        """Get total number of tracked sources."""
        return len(self.sources)


class Search_Result_Parser:
    """
    Parses and cleans search results for further processing.
    Handles result deduplication, ranking, and content extraction.
    """
    
    def __init__(self, max_results: int = 20):
        """
        Initialize search result parser.
        
        Args:
            max_results: Maximum number of results to keep
        """
        self.max_results = max_results
    
    def parse_results(self, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Parse and clean search results.
        
        Args:
            results: Raw search results
            
        Returns:
            Cleaned and deduplicated results
        """
        seen_urls = set()
        parsed_results = []
        
        for result in results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                cleaned_result = {
                    "title": self._clean_text(result.get("title", "")),
                    "url": url,
                    "snippet": self._clean_text(result.get("snippet", ""))
                }
                parsed_results.append(cleaned_result)
                
                if len(parsed_results) >= self.max_results:
                    break
        
        return parsed_results
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and special characters.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove control characters
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")
        
        return text.strip()
    
    def rank_results(self, results: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
        """
        Rank results by relevance to query.
        
        Args:
            results: Search results to rank
            query: Original search query
            
        Returns:
            Ranked results
        """
        query_words = set(query.lower().split())
        
        def calculate_relevance(result: Dict[str, str]) -> float:
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            
            title_matches = sum(1 for word in query_words if word in title)
            snippet_matches = sum(1 for word in query_words if word in snippet)
            
            # Weight title matches more heavily
            relevance = (title_matches * 2 + snippet_matches) / max(len(query_words), 1)
            return relevance
        
        ranked = sorted(results, key=calculate_relevance, reverse=True)
        return ranked
    
    def extract_key_phrases(self, text: str, max_phrases: int = 5) -> List[str]:
        """
        Extract key phrases from text (simple implementation).
        
        Args:
            text: Text to analyze
            max_phrases: Maximum number of phrases to return
            
        Returns:
            List of key phrases
        """
        # Simple keyword extraction (in production, use NLP libraries)
        words = text.lower().split()
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        significant_words = [w for w in words if w not in common_words and len(w) > 3]
        
        # Return most frequent words as key phrases
        from collections import Counter
        word_freq = Counter(significant_words)
        return [phrase for phrase, _ in word_freq.most_common(max_phrases)]
