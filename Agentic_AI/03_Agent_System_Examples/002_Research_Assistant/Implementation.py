"""
Research Assistant Agent Implementation
A complete agent system for conducting research using web search, content extraction,
summarization, and report generation.
"""

import json
import time
import datetime
import os
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI


class Web_Search_Tool:
    """Mock web search tool that simulates real search results."""
    
    def __init__(self):
        """Initialize the web search tool."""
        self.mock_results_db = {
            "ai agents software development": [
                {
                    "title": "The Future of AI Agents in Software Development",
                    "url": "https://example.com/ai-agents-future",
                    "snippet": "AI agents are revolutionizing software development by automating repetitive tasks, improving code quality, and accelerating development cycles."
                },
                {
                    "title": "How AI Agents Transform Developer Productivity",
                    "url": "https://example.com/ai-productivity",
                    "snippet": "Research shows that AI-powered coding assistants can increase developer productivity by up to 55% while reducing bugs and improving code maintainability."
                },
                {
                    "title": "AI Agents vs Traditional Development Tools",
                    "url": "https://example.com/ai-vs-traditional",
                    "snippet": "Unlike traditional IDE tools, AI agents understand context, can reason about code, and provide intelligent suggestions based on project requirements."
                }
            ],
            "impact artificial intelligence coding": [
                {
                    "title": "Measuring the Impact of AI on Software Engineering",
                    "url": "https://example.com/ai-impact-measurement",
                    "snippet": "Studies indicate that AI tools reduce time spent on debugging by 40% and improve code review efficiency significantly."
                },
                {
                    "title": "AI-Assisted Development: A Comprehensive Analysis",
                    "url": "https://example.com/ai-analysis",
                    "snippet": "Modern AI agents can understand natural language requirements, generate test cases, and even refactor legacy codebases."
                }
            ],
            "automated code generation agents": [
                {
                    "title": "Automated Code Generation with AI Agents",
                    "url": "https://example.com/auto-code-gen",
                    "snippet": "AI agents can generate production-ready code from specifications, reducing development time from weeks to hours for certain tasks."
                },
                {
                    "title": "Best Practices for AI Code Generation",
                    "url": "https://example.com/best-practices",
                    "snippet": "Effective use of AI code generation requires proper prompt engineering, code review processes, and understanding of AI limitations."
                }
            ],
            "software development automation": [
                {
                    "title": "The Automation Revolution in Software Development",
                    "url": "https://example.com/automation-revolution",
                    "snippet": "AI agents automate testing, documentation, deployment, and monitoring, freeing developers to focus on complex problem-solving."
                }
            ],
            "ai coding assistants productivity": [
                {
                    "title": "Productivity Gains from AI Coding Assistants",
                    "url": "https://example.com/productivity-gains",
                    "snippet": "Developers using AI assistants report 30-50% faster feature development and improved code quality metrics."
                }
            ]
        }
    
    def Search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Perform a mock web search and return results.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search result dictionaries with title, url, and snippet
        """
        query_lower = query.lower()
        
        # Try to find exact match first
        if query_lower in self.mock_results_db:
            results = self.mock_results_db[query_lower]
        else:
            # Find partial matches
            results = []
            for key, value in self.mock_results_db.items():
                if any(word in key for word in query_lower.split()):
                    results.extend(value)
        
        # If no matches found, generate generic results
        if not results:
            results = [
                {
                    "title": f"Research Article: {query}",
                    "url": f"https://example.com/research/{query.replace(' ', '-')}",
                    "snippet": f"Comprehensive information about {query} including recent developments and expert analysis."
                },
                {
                    "title": f"Expert Analysis: {query}",
                    "url": f"https://example.com/analysis/{query.replace(' ', '-')}",
                    "snippet": f"Detailed analysis and insights regarding {query} from industry experts and researchers."
                },
                {
                    "title": f"Latest Trends: {query}",
                    "url": f"https://example.com/trends/{query.replace(' ', '-')}",
                    "snippet": f"Current trends and future outlook for {query} in the technology industry."
                }
            ]
        
        return results[:max_results]


class Content_Extractor:
    """Extract and clean content from web URLs."""
    
    def __init__(self):
        """Initialize the content extractor."""
        self.mock_content_db = {
            "https://example.com/ai-agents-future": """
            <html><body>
            <h1>The Future of AI Agents in Software Development</h1>
            <p>AI agents are revolutionizing software development by automating repetitive tasks, 
            improving code quality, and accelerating development cycles. These intelligent systems 
            can understand context, reason about code structure, and provide actionable suggestions 
            that go beyond simple autocomplete.</p>
            <p>The integration of AI agents into development workflows has shown significant 
            productivity gains. Teams report faster feature development, reduced debugging time, 
            and improved code maintainability. The future promises even more sophisticated agents 
            that can handle complex architectural decisions and system design.</p>
            </body></html>
            """,
            "https://example.com/ai-productivity": """
            <html><body>
            <h1>How AI Agents Transform Developer Productivity</h1>
            <p>Research shows that AI-powered coding assistants can increase developer productivity 
            by up to 55% while reducing bugs and improving code maintainability. These tools 
            understand natural language requirements and can generate production-ready code.</p>
            <p>Key benefits include automated test generation, intelligent code refactoring, 
            and context-aware suggestions. Developers can focus on high-level problem-solving 
            while AI handles routine implementation details.</p>
            </body></html>
            """
        }
    
    def Extract_From_URL(self, url: str) -> str:
        """
        Extract content from a URL (mock implementation with fallback).
        
        Args:
            url: URL to extract content from
            
        Returns:
            Extracted text content
        """
        # Check mock database first
        if url in self.mock_content_db:
            html_content = self.mock_content_db[url]
        else:
            # Fallback: generate mock content
            html_content = f"""
            <html><body>
            <h1>Article Content</h1>
            <p>This is extracted content from {url}. The article discusses relevant 
            information about the topic, including key findings, analysis, and expert 
            opinions. The content provides comprehensive coverage of the subject matter.</p>
            <p>Additional paragraphs contain detailed information, statistics, and 
            insights that are valuable for research purposes.</p>
            </body></html>
            """
        
        # Clean the HTML content
        return self.Clean_Text(html_content)
    
    def Clean_Text(self, html_content: str) -> str:
        """
        Remove HTML tags and clean text content.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned plain text
        """
        # Remove script and style tags and their content
        text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities (basic)
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def Truncate(self, text: str, max_length: int = 4000) -> str:
        """
        Truncate text to maximum length for LLM processing.
        
        Args:
            text: Text to truncate
            max_length: Maximum character length
            
        Returns:
            Truncated text (with ellipsis if truncated)
        """
        if len(text) <= max_length:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        cut_point = max(last_period, last_newline)
        if cut_point > max_length * 0.8:  # Only use if we're not cutting too much
            truncated = truncated[:cut_point + 1]
        else:
            truncated = truncated[:max_length - 3] + '...'
        
        return truncated


class Summarizer:
    """Summarize text content using OpenAI."""
    
    def __init__(self, client: OpenAI):
        """
        Initialize the summarizer with OpenAI client.
        
        Args:
            client: OpenAI client instance
        """
        self.client = client
        self.model = "gpt-4o-mini"
    
    def Summarize_Text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize a single text using OpenAI.
        
        Args:
            text: Text to summarize
            max_length: Target maximum length for summary
            
        Returns:
            Summarized text
        """
        prompt = f"""Please provide a concise summary of the following text in approximately {max_length} words.
Focus on the main points and key information.

Text:
{text}

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant that creates clear, concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def Summarize_Multiple(self, texts: List[str], max_length_per: int = 150) -> List[str]:
        """
        Summarize multiple texts in batch.
        
        Args:
            texts: List of texts to summarize
            max_length_per: Target maximum length per summary
            
        Returns:
            List of summaries
        """
        summaries = []
        for text in texts:
            summary = self.Summarize_Text(text, max_length_per)
            summaries.append(summary)
            time.sleep(0.1)  # Rate limiting
        
        return summaries
    
    def Extract_Key_Points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract key points as bullet points from text.
        
        Args:
            text: Text to extract points from
            num_points: Number of key points to extract
            
        Returns:
            List of key point strings
        """
        prompt = f"""Extract the {num_points} most important key points from the following text.
Present each point as a concise bullet point (without the bullet symbol).

Text:
{text}

Key Points:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant that extracts key information points."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            # Parse bullet points (handle various formats)
            points = [p.strip() for p in re.split(r'\n[-•*]\s*|\n\d+\.\s*', content) if p.strip()]
            return points[:num_points]
        except Exception as e:
            return [f"Error extracting key points: {str(e)}"]


class Citation_Manager:
    """Manage citations and references for research reports."""
    
    def __init__(self):
        """Initialize the citation manager."""
        self.sources: List[Dict[str, Any]] = []
        self.citation_counter = 1
    
    def Add_Source(self, url: str, title: str, content: Optional[str] = None) -> int:
        """
        Register a source and return its citation number.
        
        Args:
            url: Source URL
            title: Source title
            content: Optional source content
            
        Returns:
            Citation number for this source
        """
        citation_num = self.citation_counter
        self.sources.append({
            "number": citation_num,
            "url": url,
            "title": title,
            "content": content,
            "added_at": datetime.datetime.now().isoformat()
        })
        self.citation_counter += 1
        return citation_num
    
    def Format_Citations(self) -> str:
        """
        Format all citations as a numbered list.
        
        Returns:
            Formatted citation list string
        """
        if not self.sources:
            return ""
        
        citation_lines = []
        for source in self.sources:
            citation_lines.append(
                f"[{source['number']}] {source['title']}\n"
                f"    URL: {source['url']}"
            )
        
        return "\n\n".join(citation_lines)
    
    def Get_Citation_Text(self, citation_num: int) -> str:
        """
        Get formatted inline citation reference.
        
        Args:
            citation_num: Citation number
            
        Returns:
            Formatted citation reference (e.g., "[1]")
        """
        return f"[{citation_num}]"
    
    def Get_Source_Count(self) -> int:
        """
        Get total number of sources.
        
        Returns:
            Number of registered sources
        """
        return len(self.sources)


class Research_Agent:
    """Main research agent orchestrator."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the research agent.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=api_key)
        self.search_tool = Web_Search_Tool()
        self.content_extractor = Content_Extractor()
        self.summarizer = Summarizer(self.client)
        self.citation_manager = Citation_Manager()
        self.model = "gpt-4o-mini"
    
    def Research_Topic(self, topic: str) -> Dict[str, Any]:
        """
        Conduct full research pipeline on a topic.
        
        Args:
            topic: Research topic
            
        Returns:
            Dictionary containing research report and metadata
        """
        print(f"Starting research on: {topic}")
        
        # Step 1: Generate search queries
        print("Generating search queries...")
        queries = self.Generate_Search_Queries(topic)
        print(f"Generated {len(queries)} search queries")
        
        # Step 2: Gather sources
        print("Gathering sources...")
        sources = self.Gather_Sources(queries)
        print(f"Gathered {len(sources)} sources")
        
        # Step 3: Analyze sources
        print("Analyzing sources...")
        analyses = self.Analyze_Sources(sources)
        print(f"Completed analysis of {len(analyses)} sources")
        
        # Step 4: Synthesize report
        print("Synthesizing report...")
        report = self.Synthesize_Report(topic, analyses, self.citation_manager)
        
        return {
            "topic": topic,
            "report": report,
            "sources_count": self.citation_manager.Get_Source_Count(),
            "queries_used": queries,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def Generate_Search_Queries(self, topic: str, num_queries: int = 4) -> List[str]:
        """
        Generate diverse search queries for a topic using LLM.
        
        Args:
            topic: Research topic
            num_queries: Number of queries to generate
            
        Returns:
            List of search query strings
        """
        prompt = f"""Generate {num_queries} diverse and specific search queries that would help research the following topic.
Each query should approach the topic from a different angle or focus on different aspects.
Return only the queries, one per line, without numbering or bullets.

Topic: {topic}

Search queries:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant that creates effective search queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            queries = [q.strip() for q in content.split('\n') if q.strip()]
            # Remove any numbering or bullets
            queries = [re.sub(r'^\d+[\.\)]\s*|^[-•*]\s*', '', q) for q in queries]
            return queries[:num_queries]
        except Exception as e:
            # Fallback queries
            return [
                f"{topic} overview",
                f"{topic} benefits",
                f"{topic} challenges",
                f"{topic} future trends"
            ]
    
    def Gather_Sources(self, queries: List[str], max_results_per_query: int = 3) -> List[Dict[str, Any]]:
        """
        Search and extract content from search results.
        
        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query
            
        Returns:
            List of source dictionaries with content
        """
        sources = []
        seen_urls = set()
        
        for query in queries:
            search_results = self.search_tool.Search(query, max_results=max_results_per_query)
            
            for result in search_results:
                url = result["url"]
                if url in seen_urls:
                    continue
                
                seen_urls.add(url)
                
                # Extract content
                try:
                    content = self.content_extractor.Extract_From_URL(url)
                    truncated_content = self.content_extractor.Truncate(content, max_length=3000)
                    
                    sources.append({
                        "url": url,
                        "title": result["title"],
                        "snippet": result["snippet"],
                        "content": truncated_content
                    })
                    
                    # Add to citation manager
                    self.citation_manager.Add_Source(url, result["title"], truncated_content)
                    
                    time.sleep(0.2)  # Rate limiting
                except Exception as e:
                    print(f"Error extracting content from {url}: {e}")
                    continue
        
        return sources
    
    def Analyze_Sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Summarize and extract key findings from sources.
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            List of analysis dictionaries
        """
        analyses = []
        
        for source in sources:
            content = source.get("content", source.get("snippet", ""))
            
            if not content:
                continue
            
            # Summarize
            summary = self.summarizer.Summarize_Text(content, max_length=150)
            
            # Extract key points
            key_points = self.summarizer.Extract_Key_Points(content, num_points=3)
            
            analyses.append({
                "url": source["url"],
                "title": source["title"],
                "summary": summary,
                "key_points": key_points
            })
            
            time.sleep(0.3)  # Rate limiting
        
        return analyses
    
    def Synthesize_Report(self, topic: str, analyses: List[Dict[str, Any]], 
                         citation_manager: Citation_Manager) -> str:
        """
        Generate a comprehensive research report from analyses.
        
        Args:
            topic: Research topic
            analyses: List of source analyses
            citation_manager: Citation manager instance
            
        Returns:
            Formatted research report string
        """
        # Prepare context for synthesis
        analysis_text = ""
        for i, analysis in enumerate(analyses, 1):
            analysis_text += f"\n\nSource {i}: {analysis['title']}\n"
            analysis_text += f"Summary: {analysis['summary']}\n"
            analysis_text += "Key Points:\n"
            for point in analysis.get('key_points', []):
                analysis_text += f"  - {point}\n"
        
        prompt = f"""Create a comprehensive research report on the following topic based on the provided source analyses.
The report should be well-structured, informative, and synthesize information from multiple sources.

Topic: {topic}

Source Analyses:
{analysis_text}

Please create a detailed report that includes:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Implications and Future Outlook
5. Conclusion

Use inline citations [1], [2], etc. to reference sources. Make the report professional and comprehensive."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst that creates comprehensive, well-structured research reports."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=2000
            )
            
            report = response.choices[0].message.content.strip()
            return report
        except Exception as e:
            return f"Error generating report: {str(e)}"


class Report_Formatter:
    """Format research reports with markdown and additional sections."""
    
    @staticmethod
    def Format_Markdown(report: str, topic: str, citation_manager: Citation_Manager) -> str:
        """
        Format a research report as markdown with all sections.
        
        Args:
            report: Raw report text
            topic: Research topic
            citation_manager: Citation manager instance
            
        Returns:
            Formatted markdown report
        """
        formatted = f"# Research Report: {topic}\n\n"
        formatted += f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        formatted += "---\n\n"
        
        # Add table of contents
        formatted += Report_Formatter.Add_Table_Of_Contents(report)
        formatted += "\n\n---\n\n"
        
        # Add main report content
        formatted += report
        formatted += "\n\n---\n\n"
        
        # Add citations section
        formatted += Report_Formatter.Add_Citations_Section(citation_manager)
        
        return formatted
    
    @staticmethod
    def Add_Table_Of_Contents(report: str) -> str:
        """
        Generate table of contents from report headings.
        
        Args:
            report: Report text
            
        Returns:
            Table of contents markdown
        """
        toc_lines = ["## Table of Contents\n"]
        
        lines = report.split('\n')
        for line in lines:
            # Match markdown headings
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                if title:
                    indent = "  " * (level - 1)
                    anchor = title.lower().replace(' ', '-').replace('.', '')
                    anchor = re.sub(r'[^\w\-]', '', anchor)
                    toc_lines.append(f"{indent}- [{title}](#{anchor})")
        
        if len(toc_lines) == 1:  # Only header, no sections found
            toc_lines.extend([
                "- Executive Summary",
                "- Key Findings",
                "- Detailed Analysis",
                "- Implications and Future Outlook",
                "- Conclusion"
            ])
        
        return "\n".join(toc_lines)
    
    @staticmethod
    def Add_Citations_Section(citation_manager: Citation_Manager) -> str:
        """
        Add formatted citations section to report.
        
        Args:
            citation_manager: Citation manager instance
            
        Returns:
            Formatted citations section markdown
        """
        section = "## References\n\n"
        citations = citation_manager.Format_Citations()
        
        if citations:
            section += citations
        else:
            section += "No sources cited."
        
        return section


def main():
    """Main function to demonstrate the research assistant."""
    print("=" * 80)
    print("Research Assistant Agent - Demonstration")
    print("=" * 80)
    print()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set it to use the OpenAI API features.")
        print("Using mock mode for demonstration...")
        api_key = "mock-key-for-demo"
    
    try:
        # Create research agent
        agent = Research_Agent(api_key=api_key)
        
        # Research topic
        topic = "Impact of AI Agents on Software Development"
        print(f"\nResearching topic: {topic}\n")
        
        # Conduct research
        result = agent.Research_Topic(topic)
        
        # Format report
        formatter = Report_Formatter()
        formatted_report = formatter.Format_Markdown(
            result["report"],
            result["topic"],
            agent.citation_manager
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("RESEARCH REPORT")
        print("=" * 80)
        print()
        print(formatted_report)
        print()
        print("=" * 80)
        print(f"Research completed. Sources: {result['sources_count']}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during research: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
