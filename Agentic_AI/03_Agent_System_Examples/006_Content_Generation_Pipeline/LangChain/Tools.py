"""
Tools module for Content Generation Pipeline.
Contains research, grammar checking, SEO analysis, readability calculation,
content validation, and SEO analysis utilities.
"""

from langchain_core.tools import tool
from typing import Dict, List, Any, Optional
import re
import math


@tool
def Research_Topic(topic: str) -> Dict[str, Any]:
    """
    Conduct mock research on a given topic and return key facts, statistics, and quotes.
    
    Args:
        topic: The topic to research
        
    Returns:
        Dictionary containing research results with key facts, statistics, and quotes
    """
    topic_lower = topic.lower()
    
    research_data = {
        "topic": topic,
        "key_facts": [],
        "statistics": [],
        "quotes": [],
        "related_topics": []
    }
    
    if "ai agents" in topic_lower or "agent" in topic_lower:
        research_data["key_facts"] = [
            "AI agents are autonomous systems that can perform tasks independently",
            "Modern AI agents use large language models as their reasoning engine",
            "Agent frameworks like LangChain and LangGraph simplify agent development",
            "AI agents can use tools and interact with external systems"
        ]
        research_data["statistics"] = [
            "85% of enterprises plan to deploy AI agents by 2025",
            "AI agent market is expected to reach $50 billion by 2027",
            "Productivity improvements of 30-40% reported with AI agent adoption"
        ]
        research_data["quotes"] = [
            "AI agents represent the next evolution in human-computer interaction",
            "The future of software development will be agent-driven"
        ]
        research_data["related_topics"] = ["LLMs", "RAG", "Tool Use", "Multi-Agent Systems"]
    
    elif "rag" in topic_lower or "retrieval" in topic_lower:
        research_data["key_facts"] = [
            "RAG combines retrieval of external knowledge with generation",
            "RAG systems reduce hallucinations by grounding responses in retrieved documents",
            "Vector databases are commonly used for semantic search in RAG",
            "RAG is particularly effective for domain-specific applications"
        ]
        research_data["statistics"] = [
            "RAG improves answer accuracy by 40-60% compared to standalone LLMs",
            "70% of enterprise AI applications use RAG for knowledge management",
            "RAG reduces training costs by 80% compared to fine-tuning"
        ]
        research_data["quotes"] = [
            "RAG bridges the gap between static knowledge and dynamic generation",
            "Enterprise search is being transformed by RAG technology"
        ]
        research_data["related_topics"] = ["Vector Databases", "Embeddings", "Semantic Search", "Knowledge Graphs"]
    
    elif "langchain" in topic_lower:
        research_data["key_facts"] = [
            "LangChain is a framework for building LLM applications",
            "LangChain provides abstractions for chains, agents, and tools",
            "LangGraph extends LangChain with graph-based stateful workflows",
            "LangChain supports multiple LLM providers and vector stores"
        ]
        research_data["statistics"] = [
            "LangChain has over 100,000 GitHub stars",
            "Used by 50,000+ companies worldwide",
            "LangChain ecosystem includes 200+ integrations"
        ]
        research_data["quotes"] = [
            "LangChain makes it easy to build production-ready LLM applications",
            "The framework democratizes access to advanced AI capabilities"
        ]
        research_data["related_topics"] = ["LangGraph", "Chains", "Agents", "Tools", "RAG"]
    
    else:
        research_data["key_facts"] = [
            f"{topic} is an important topic in modern technology",
            "Understanding {topic} requires both theoretical and practical knowledge",
            "Best practices for {topic} continue to evolve"
        ]
        research_data["statistics"] = [
            "Industry adoption of {topic} is growing rapidly",
            "Research shows significant benefits from implementing {topic}"
        ]
        research_data["quotes"] = [
            f"{topic} represents a significant advancement in the field"
        ]
        research_data["related_topics"] = ["Related Technology", "Best Practices", "Implementation"]
    
    return research_data


@tool
def Check_Grammar(text: str) -> Dict[str, Any]:
    """
    Perform basic grammar and spelling check on text.
    
    Args:
        text: The text to check
        
    Returns:
        Dictionary containing grammar check results with errors and suggestions
    """
    errors = []
    suggestions = []
    
    words = text.split()
    
    common_mistakes = {
        "teh": "the",
        "adn": "and",
        "taht": "that",
        "recieve": "receive",
        "seperate": "separate",
        "occured": "occurred",
        "definately": "definitely"
    }
    
    for i, word in enumerate(words):
        word_lower = word.lower().strip(".,!?;:")
        if word_lower in common_mistakes:
            errors.append({
                "word": word,
                "position": i,
                "suggestion": common_mistakes[word_lower]
            })
            suggestions.append(f"Replace '{word}' with '{common_mistakes[word_lower]}'")
    
    double_spaces = list(re.finditer(r'\s{2,}', text))
    for match in double_spaces:
        errors.append({
            "type": "double_space",
            "position": match.start(),
            "suggestion": "Remove extra space"
        })
    
    sentence_endings = re.findall(r'[.!?]\s+[a-z]', text)
    if sentence_endings:
        for match in sentence_endings:
            errors.append({
                "type": "capitalization",
                "position": text.find(match),
                "suggestion": "Capitalize first letter after sentence ending"
            })
    
    error_count = len(errors)
    is_valid = error_count == 0
    
    return {
        "is_valid": is_valid,
        "error_count": error_count,
        "errors": errors[:10],
        "suggestions": suggestions[:10],
        "word_count": len(words)
    }


@tool
def Analyze_SEO(text: str, target_keyword: str) -> Dict[str, Any]:
    """
    Analyze SEO aspects of text including keyword density and heading structure.
    
    Args:
        text: The text to analyze
        target_keyword: The target keyword to check density for
        
    Returns:
        Dictionary containing SEO analysis results
    """
    text_lower = text.lower()
    keyword_lower = target_keyword.lower()
    
    words = re.findall(r'\b\w+\b', text_lower)
    total_words = len(words)
    
    keyword_count = text_lower.count(keyword_lower)
    keyword_density = keyword_count / total_words if total_words > 0 else 0.0
    
    headings = re.findall(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
    h1_count = len(re.findall(r'^#\s+', text, re.MULTILINE))
    h2_count = len(re.findall(r'^##\s+', text, re.MULTILINE))
    h3_count = len(re.findall(r'^###\s+', text, re.MULTILINE))
    
    title_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
    title = title_match.group(1) if title_match else ""
    title_length = len(title)
    
    meta_description_match = re.search(r'<meta.*description.*content=["\']([^"\']+)["\']', text, re.IGNORECASE)
    if not meta_description_match:
        first_paragraph = re.search(r'^[^\n]+', text, re.MULTILINE)
        meta_description = first_paragraph.group(0) if first_paragraph else ""
    else:
        meta_description = meta_description_match.group(1)
    
    meta_description_length = len(meta_description)
    
    return {
        "keyword": target_keyword,
        "keyword_count": keyword_count,
        "keyword_density": round(keyword_density, 4),
        "total_words": total_words,
        "headings": {
            "h1_count": h1_count,
            "h2_count": h2_count,
            "h3_count": h3_count,
            "total": len(headings),
            "list": headings[:10]
        },
        "title": {
            "text": title,
            "length": title_length,
            "is_optimal": 30 <= title_length <= 60
        },
        "meta_description": {
            "text": meta_description[:160],
            "length": meta_description_length,
            "is_optimal": 120 <= meta_description_length <= 160
        }
    }


@tool
def Calculate_Readability(text: str) -> Dict[str, Any]:
    """
    Calculate Flesch-Kincaid readability score for text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Dictionary containing readability metrics including Flesch-Kincaid score
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    total_sentences = len(sentences)
    
    syllables = 0
    for word in words:
        word_syllables = _Count_Syllables(word)
        syllables += word_syllables
    
    if total_sentences == 0 or total_words == 0:
        return {
            "flesch_kincaid_score": 0.0,
            "reading_level": "Unknown",
            "total_sentences": 0,
            "total_words": 0,
            "total_syllables": 0,
            "avg_sentence_length": 0.0,
            "avg_syllables_per_word": 0.0
        }
    
    avg_sentence_length = total_words / total_sentences
    avg_syllables_per_word = syllables / total_words
    
    flesch_kincaid = (
        206.835 - 
        (1.015 * avg_sentence_length) - 
        (84.6 * avg_syllables_per_word)
    )
    
    flesch_kincaid = max(0.0, min(100.0, flesch_kincaid))
    
    if flesch_kincaid >= 90:
        reading_level = "Very Easy"
    elif flesch_kincaid >= 80:
        reading_level = "Easy"
    elif flesch_kincaid >= 70:
        reading_level = "Fairly Easy"
    elif flesch_kincaid >= 60:
        reading_level = "Standard"
    elif flesch_kincaid >= 50:
        reading_level = "Fairly Difficult"
    elif flesch_kincaid >= 30:
        reading_level = "Difficult"
    else:
        reading_level = "Very Difficult"
    
    return {
        "flesch_kincaid_score": round(flesch_kincaid, 2),
        "reading_level": reading_level,
        "total_sentences": total_sentences,
        "total_words": total_words,
        "total_syllables": syllables,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_syllables_per_word": round(avg_syllables_per_word, 2)
    }


def _Count_Syllables(word: str) -> int:
    """Count syllables in a word."""
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3:
        return 1
    
    vowels = "aeiouy"
    syllable_count = 0
    prev_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel
    
    if word.endswith("e"):
        syllable_count -= 1
    
    if word.endswith("le") and len(word) > 2:
        syllable_count += 1
    
    return max(1, syllable_count)


class Content_Validator:
    """Class for validating content against quality metrics."""
    
    def __init__(
        self,
        min_word_count: int = 500,
        max_word_count: int = 2000,
        min_heading_count: int = 2
    ):
        """
        Initialize content validator.
        
        Args:
            min_word_count: Minimum word count required
            max_word_count: Maximum word count allowed
            min_heading_count: Minimum number of headings required
        """
        self.min_word_count = min_word_count
        self.max_word_count = max_word_count
        self.min_heading_count = min_heading_count
    
    def Validate(self, content: str) -> Dict[str, Any]:
        """
        Validate content against quality metrics.
        
        Args:
            content: The content to validate
            
        Returns:
            Dictionary containing validation results
        """
        words = re.findall(r'\b\w+\b', content)
        word_count = len(words)
        
        headings = re.findall(r'^#{1,6}\s+', content, re.MULTILINE)
        heading_count = len(headings)
        
        paragraphs = re.split(r'\n\s*\n', content)
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        word_count_valid = self.min_word_count <= word_count <= self.max_word_count
        heading_count_valid = heading_count >= self.min_heading_count
        structure_valid = paragraph_count >= 3 and sentence_count >= 5
        
        is_valid = word_count_valid and heading_count_valid and structure_valid
        
        issues = []
        if not word_count_valid:
            if word_count < self.min_word_count:
                issues.append(f"Word count ({word_count}) below minimum ({self.min_word_count})")
            else:
                issues.append(f"Word count ({word_count}) exceeds maximum ({self.max_word_count})")
        
        if not heading_count_valid:
            issues.append(f"Heading count ({heading_count}) below minimum ({self.min_heading_count})")
        
        if not structure_valid:
            issues.append("Content structure is insufficient (needs more paragraphs/sentences)")
        
        return {
            "is_valid": is_valid,
            "word_count": word_count,
            "heading_count": heading_count,
            "paragraph_count": paragraph_count,
            "sentence_count": sentence_count,
            "issues": issues,
            "metrics": {
                "word_count_valid": word_count_valid,
                "heading_count_valid": heading_count_valid,
                "structure_valid": structure_valid
            }
        }
    
    def Calculate_Quality_Score(self, content: str) -> float:
        """
        Calculate overall quality score for content (0.0-1.0).
        
        Args:
            content: The content to score
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        validation = self.Validate(content)
        
        if validation["is_valid"]:
            base_score = 0.7
        else:
            base_score = 0.4
        
        word_count = validation["word_count"]
        word_score = 1.0 if self.min_word_count <= word_count <= self.max_word_count else 0.5
        
        heading_score = min(1.0, validation["heading_count"] / max(1, self.min_heading_count))
        
        structure_score = min(1.0, validation["paragraph_count"] / 5.0)
        
        grammar_result = Check_Grammar(content)
        grammar_score = 1.0 if grammar_result["is_valid"] else max(0.5, 1.0 - (grammar_result["error_count"] / 20.0))
        
        readability_result = Calculate_Readability(content)
        readability_score = min(1.0, readability_result["flesch_kincaid_score"] / 70.0)
        
        quality_score = (
            base_score * 0.3 +
            word_score * 0.2 +
            heading_score * 0.15 +
            structure_score * 0.15 +
            grammar_score * 0.1 +
            readability_score * 0.1
        )
        
        return round(min(1.0, max(0.0, quality_score)), 3)


class SEO_Analyzer:
    """Class for analyzing and optimizing SEO aspects of content."""
    
    def __init__(
        self,
        min_keyword_density: float = 0.01,
        max_keyword_density: float = 0.03
    ):
        """
        Initialize SEO analyzer.
        
        Args:
            min_keyword_density: Minimum keyword density threshold
            max_keyword_density: Maximum keyword density threshold
        """
        self.min_keyword_density = min_keyword_density
        self.max_keyword_density = max_keyword_density
    
    def Analyze(self, content: str, target_keywords: List[str]) -> Dict[str, Any]:
        """
        Analyze SEO aspects of content for multiple keywords.
        
        Args:
            content: The content to analyze
            target_keywords: List of target keywords
            
        Returns:
            Dictionary containing SEO analysis results
        """
        analyses = []
        overall_score = 0.0
        
        for keyword in target_keywords:
            analysis = Analyze_SEO(content, keyword)
            analyses.append(analysis)
            
            keyword_score = 0.0
            if self.min_keyword_density <= analysis["keyword_density"] <= self.max_keyword_density:
                keyword_score += 0.5
            elif analysis["keyword_density"] > 0:
                keyword_score += 0.3
            
            if analysis["headings"]["h1_count"] > 0:
                keyword_score += 0.2
            
            if analysis["title"]["is_optimal"]:
                keyword_score += 0.2
            
            if analysis["meta_description"]["is_optimal"]:
                keyword_score += 0.1
            
            overall_score += keyword_score
        
        overall_score = overall_score / len(target_keywords) if target_keywords else 0.0
        
        return {
            "overall_score": round(overall_score, 3),
            "keyword_analyses": analyses,
            "recommendations": self._Generate_Recommendations(analyses)
        }
    
    def _Generate_Recommendations(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate SEO recommendations based on analyses."""
        recommendations = []
        
        for analysis in analyses:
            keyword = analysis["keyword"]
            density = analysis["keyword_density"]
            
            if density < self.min_keyword_density:
                recommendations.append(
                    f"Increase keyword density for '{keyword}' (current: {density:.2%}, "
                    f"target: {self.min_keyword_density:.2%}-{self.max_keyword_density:.2%})"
                )
            elif density > self.max_keyword_density:
                recommendations.append(
                    f"Reduce keyword density for '{keyword}' (current: {density:.2%}, "
                    f"target: {self.min_keyword_density:.2%}-{self.max_keyword_density:.2%})"
                )
            
            if analysis["headings"]["h1_count"] == 0:
                recommendations.append(f"Add H1 heading containing '{keyword}'")
            
            if not analysis["title"]["is_optimal"]:
                recommendations.append(
                    f"Optimize title length (current: {analysis['title']['length']}, "
                    f"optimal: 30-60 characters)"
                )
            
            if not analysis["meta_description"]["is_optimal"]:
                recommendations.append(
                    f"Optimize meta description length (current: {analysis['meta_description']['length']}, "
                    f"optimal: 120-160 characters)"
                )
        
        return recommendations
    
    def Suggest_Meta_Tags(self, content: str, target_keywords: List[str]) -> Dict[str, str]:
        """
        Suggest meta tags for content.
        
        Args:
            content: The content to generate meta tags for
            target_keywords: List of target keywords
            
        Returns:
            Dictionary containing suggested meta tags
        """
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Content Title"
        
        first_paragraph = re.search(r'^[^\n]+', content, re.MULTILINE)
        description = first_paragraph.group(0) if first_paragraph else ""
        
        if len(description) > 160:
            description = description[:157] + "..."
        
        keywords_str = ", ".join(target_keywords[:5])
        
        return {
            "title": title[:60],
            "description": description,
            "keywords": keywords_str
        }
