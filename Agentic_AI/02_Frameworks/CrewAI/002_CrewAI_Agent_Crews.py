"""
CrewAI Basic Examples: Agent and Crew Configurations

This module demonstrates fundamental CrewAI patterns including:
1. Basic Crew with sequential process
2. Hierarchical Crew with manager agent
3. Custom Tools implementation
4. Task Dependencies
5. Output handling (JSON, files, Pydantic models)
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from pydantic import BaseModel
from typing import List, Dict
import json
import os


# ============================================================================
# CUSTOM TOOLS
# ============================================================================

@tool
def search_web(query: str) -> str:
    """
    Search the web for information about a given query.
    
    Args:
        query: The search query string
        
    Returns:
        Search results as a formatted string
    """
    # In a real implementation, this would call a search API
    return f"Search results for '{query}': Found relevant information about the topic."


@tool
def calculate_statistics(numbers: List[float]) -> Dict[str, float]:
    """
    Calculate statistical measures for a list of numbers.
    
    Args:
        numbers: List of numeric values
        
    Returns:
        Dictionary containing mean, median, min, max
    """
    if not numbers:
        return {"mean": 0, "median": 0, "min": 0, "max": 0}
    
    sorted_nums = sorted(numbers)
    n = len(numbers)
    
    mean = sum(numbers) / n
    median = sorted_nums[n // 2] if n % 2 == 1 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    min_val = min(numbers)
    max_val = max(numbers)
    
    return {
        "mean": mean,
        "median": median,
        "min": min_val,
        "max": max_val
    }


@tool
def read_file_content(filepath: str) -> str:
    """
    Read content from a file.
    
    Args:
        filepath: Path to the file to read
        
    Returns:
        File contents as string
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file_content(filepath: str, content: str) -> str:
    """
    Write content to a file.
    
    Args:
        filepath: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Success message
    """
    try:
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote content to {filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class ResearchFindings(BaseModel):
    """Structured model for research findings"""
    topic: str
    key_points: List[str]
    sources: List[str]
    summary: str
    confidence_level: float


class ArticleStructure(BaseModel):
    """Structured model for article outline"""
    title: str
    introduction: str
    main_sections: List[str]
    conclusion: str
    word_count_estimate: int


# ============================================================================
# EXAMPLE 1: BASIC CREW WITH SEQUENTIAL PROCESS
# ============================================================================

def basic_sequential_crew():
    """
    Demonstrates a basic crew with two agents working sequentially.
    Agent 1: Researcher - conducts research on a topic
    Agent 2: Writer - writes an article based on research
    
    Returns:
        Crew execution result
    """
    # Define agents
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Conduct thorough research on given topics and provide accurate, well-sourced information",
        backstory="""You are an experienced researcher with expertise in academic sources,
                   data analysis, and fact-checking. You have a PhD in Information Science
                   and have worked for major research institutions. You always verify
                   information from multiple sources before presenting findings.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_web]
    )
    
    writer = Agent(
        role="Technical Content Writer",
        goal="Write clear, engaging, and well-structured articles based on research findings",
        backstory="""You are a skilled technical writer with over 10 years of experience
                   creating content for technical audiences. You excel at translating complex
                   information into clear, readable prose. Your articles are always well-organized,
                   engaging, and factually accurate.""",
        verbose=True,
        allow_delegation=False
    )
    
    # Define tasks
    research_task = Task(
        description="""Research the topic of 'Artificial Intelligence in Healthcare'.
                      Focus on recent developments, practical applications, and future prospects.
                      Provide comprehensive findings with key points and sources.""",
        expected_output="A detailed research report with key findings, trends, and implications",
        agent=researcher
    )
    
    writing_task = Task(
        description="""Write a comprehensive article based on the research findings.
                      The article should be well-structured with an introduction, main sections,
                      and conclusion. Aim for approximately 1000 words.""",
        expected_output="A complete article ready for publication",
        agent=writer,
        context=[research_task]
    )
    
    # Create crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute crew
    result = crew.kickoff(inputs={"topic": "Artificial Intelligence in Healthcare"})
    return result


# ============================================================================
# EXAMPLE 2: HIERARCHICAL CREW WITH MANAGER AGENT
# ============================================================================

def hierarchical_crew_with_manager():
    """
    Demonstrates a hierarchical crew with a manager agent coordinating specialists.
    Manager: Project Manager - coordinates and reviews work
    Specialist 1: Market Researcher - conducts market research
    Specialist 2: Financial Analyst - analyzes financial data
    Specialist 3: Strategy Consultant - develops strategic recommendations
    
    Returns:
        Crew execution result
    """
    # Manager agent
    manager = Agent(
        role="Project Manager",
        goal="Coordinate team members, review their work, and ensure high-quality deliverables",
        backstory="""You are an experienced project manager with expertise in coordinating
                   cross-functional teams. You excel at breaking down complex projects into
                   manageable tasks, assigning them to the right specialists, and ensuring
                   quality through thorough review processes.""",
        verbose=True,
        allow_delegation=True
    )
    
    # Specialist agents
    market_researcher = Agent(
        role="Market Research Specialist",
        goal="Conduct comprehensive market research and provide actionable insights",
        backstory="""You are a market research expert with 8 years of experience analyzing
                   market trends, competitor landscapes, and consumer behavior. You specialize
                   in both quantitative and qualitative research methods.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_web]
    )
    
    financial_analyst = Agent(
        role="Financial Analyst",
        goal="Analyze financial data and provide financial insights and projections",
        backstory="""You are a certified financial analyst with expertise in financial modeling,
                   valuation, and risk analysis. You have worked for top consulting firms and
                   have a strong track record of accurate financial forecasting.""",
        verbose=True,
        allow_delegation=False,
        tools=[calculate_statistics]
    )
    
    strategy_consultant = Agent(
        role="Strategy Consultant",
        goal="Develop strategic recommendations based on research and analysis",
        backstory="""You are a senior strategy consultant with experience advising Fortune 500
                   companies. You excel at synthesizing complex information from multiple sources
                   and developing actionable strategic recommendations.""",
        verbose=True,
        allow_delegation=False
    )
    
    # Define tasks
    market_research_task = Task(
        description="""Research the market for electric vehicles, including market size,
                      growth trends, key competitors, and consumer preferences.""",
        expected_output="Comprehensive market research report with key insights",
        agent=market_researcher
    )
    
    financial_analysis_task = Task(
        description="""Analyze financial data for the electric vehicle market, including
                      revenue projections, cost structures, and investment requirements.""",
        expected_output="Financial analysis report with projections and recommendations",
        agent=financial_analyst
    )
    
    strategy_task = Task(
        description="""Develop strategic recommendations for entering the electric vehicle
                      market based on the market research and financial analysis. Provide
                      actionable steps and risk mitigation strategies.""",
        expected_output="Strategic plan with recommendations and implementation roadmap",
        agent=strategy_consultant,
        context=[market_research_task, financial_analysis_task]
    )
    
    # Create hierarchical crew
    crew = Crew(
        agents=[manager, market_researcher, financial_analyst, strategy_consultant],
        tasks=[market_research_task, financial_analysis_task, strategy_task],
        process=Process.hierarchical,
        verbose=True
    )
    
    # Execute crew
    result = crew.kickoff(inputs={"project": "Electric Vehicle Market Entry Strategy"})
    return result


# ============================================================================
# EXAMPLE 3: CREW WITH CUSTOM TOOLS
# ============================================================================

def crew_with_custom_tools():
    """
    Demonstrates a crew where agents use custom tools to extend their capabilities.
    Data Analyst: Uses calculation tools to analyze data
    Report Writer: Uses file tools to read and write reports
    
    Returns:
        Crew execution result
    """
    # Agent with custom calculation tools
    data_analyst = Agent(
        role="Data Analyst",
        goal="Analyze datasets and provide statistical insights",
        backstory="""You are a data analyst specializing in statistical analysis and
                   data interpretation. You use various analytical tools to extract
                   meaningful insights from data.""",
        verbose=True,
        tools=[calculate_statistics]
    )
    
    # Agent with file manipulation tools
    report_writer = Agent(
        role="Report Writer",
        goal="Create and manage report files based on analysis results",
        backstory="""You are a technical writer who creates comprehensive reports
                   based on data analysis. You excel at organizing information and
                   writing clear, structured reports.""",
        verbose=True,
        tools=[read_file_content, write_file_content]
    )
    
    # Define tasks
    analysis_task = Task(
        description="""Analyze the following dataset: [10.5, 12.3, 9.8, 11.2, 13.1, 10.9, 12.7].
                      Calculate statistical measures including mean, median, min, and max.
                      Provide insights about the data distribution.""",
        expected_output="Statistical analysis with key metrics and insights",
        agent=data_analyst
    )
    
    report_task = Task(
        description="""Create a comprehensive report file based on the statistical analysis.
                      The report should include an introduction, methodology, results, and
                      conclusions. Save the report to 'analysis_report.txt'.""",
        expected_output="Report file saved successfully",
        agent=report_writer,
        context=[analysis_task]
    )
    
    # Create crew
    crew = Crew(
        agents=[data_analyst, report_writer],
        tasks=[analysis_task, report_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute crew
    result = crew.kickoff()
    return result


# ============================================================================
# EXAMPLE 4: TASK DEPENDENCIES AND CONTEXT PASSING
# ============================================================================

def crew_with_task_dependencies():
    """
    Demonstrates complex task dependencies where tasks build upon previous outputs.
    Task 1: Research - gathers information
    Task 2: Analysis - analyzes research (depends on Task 1)
    Task 3: Writing - writes content (depends on Task 1 and Task 2)
    Task 4: Review - reviews final content (depends on Task 3)
    
    Returns:
        Crew execution result
    """
    # Define agents
    researcher = Agent(
        role="Research Specialist",
        goal="Gather comprehensive information on specified topics",
        backstory="""You are a research specialist with expertise in information gathering
                   and source verification. You excel at finding relevant, accurate information
                   from multiple sources.""",
        verbose=True,
        tools=[search_web]
    )
    
    analyst = Agent(
        role="Data Analyst",
        goal="Analyze research data and extract key insights",
        backstory="""You are an analyst who specializes in processing and interpreting
                   research data. You identify patterns, trends, and key insights from
                   raw information.""",
        verbose=True
    )
    
    writer = Agent(
        role="Content Writer",
        goal="Create well-written content based on research and analysis",
        backstory="""You are a skilled writer who creates engaging content based on
                   research and analysis. You excel at synthesizing information into
                   clear, readable prose.""",
        verbose=True
    )
    
    reviewer = Agent(
        role="Content Reviewer",
        goal="Review and improve written content for quality and accuracy",
        backstory="""You are an experienced editor and reviewer with a keen eye for
                   detail. You ensure content is accurate, well-structured, and meets
                   quality standards.""",
        verbose=True
    )
    
    # Define tasks with dependencies
    research_task = Task(
        description="""Research the topic of 'Sustainable Energy Solutions'.
                      Gather information about solar, wind, and hydroelectric power.
                      Include recent developments and cost-effectiveness data.""",
        expected_output="Comprehensive research data with sources",
        agent=researcher
    )
    
    analysis_task = Task(
        description="""Analyze the research data on sustainable energy solutions.
                      Identify key trends, compare different energy sources, and
                      highlight the most promising solutions.""",
        expected_output="Analysis report with key insights and comparisons",
        agent=analyst,
        context=[research_task]  # Depends on research_task output
    )
    
    writing_task = Task(
        description="""Write a comprehensive article about sustainable energy solutions
                      based on the research and analysis. The article should be informative,
                      well-structured, and approximately 1500 words.""",
        expected_output="Complete article ready for review",
        agent=writer,
        context=[research_task, analysis_task]  # Depends on both previous tasks
    )
    
    review_task = Task(
        description="""Review the written article for accuracy, clarity, and quality.
                      Check facts against the research data, improve structure if needed,
                      and ensure the article meets publication standards.""",
        expected_output="Reviewed and improved article",
        agent=reviewer,
        context=[writing_task]  # Depends on writing_task output
    )
    
    # Create crew
    crew = Crew(
        agents=[researcher, analyst, writer, reviewer],
        tasks=[research_task, analysis_task, writing_task, review_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute crew
    result = crew.kickoff()
    return result


# ============================================================================
# EXAMPLE 5: JSON OUTPUT HANDLING
# ============================================================================

def crew_with_json_output():
    """
    Demonstrates tasks configured to output structured JSON data.
    Research task outputs JSON with structured findings.
    
    Returns:
        Crew execution result
    """
    researcher = Agent(
        role="Research Analyst",
        goal="Conduct research and provide structured findings",
        backstory="""You are a research analyst who specializes in structured data
                   collection and analysis. You excel at organizing information into
                   clear, structured formats.""",
        verbose=True,
        tools=[search_web]
    )
    
    research_task = Task(
        description="""Research the topic of 'Quantum Computing Applications'.
                      Provide findings in a structured format with:
                      - Key points (list)
                      - Sources (list)
                      - Summary (text)
                      - Confidence level (0-1)""",
        expected_output="""JSON object with structure:
                         {
                           "topic": "string",
                           "key_points": ["string"],
                           "sources": ["string"],
                           "summary": "string",
                           "confidence_level": 0.0-1.0
                         }""",
        agent=researcher,
        output_json=True  # Enable JSON output
    )
    
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return result


# ============================================================================
# EXAMPLE 6: FILE OUTPUT HANDLING
# ============================================================================

def crew_with_file_output():
    """
    Demonstrates tasks configured to save output directly to files.
    Writing task saves output to a specified file.
    
    Returns:
        Crew execution result
    """
    writer = Agent(
        role="Documentation Writer",
        goal="Create documentation files",
        backstory="""You are a technical writer specializing in creating comprehensive
                   documentation. You excel at writing clear, well-organized documentation
                   that is easy to follow.""",
        verbose=True
    )
    
    documentation_task = Task(
        description="""Create comprehensive documentation for a Python API.
                      Include sections on:
                      - Installation
                      - Quick Start Guide
                      - API Reference
                      - Examples
                      - Troubleshooting""",
        expected_output="Complete documentation in markdown format",
        agent=writer,
        output_file="api_documentation.md"  # Save to file
    )
    
    crew = Crew(
        agents=[writer],
        tasks=[documentation_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return result


# ============================================================================
# EXAMPLE 7: PYDANTIC MODEL OUTPUT
# ============================================================================

def crew_with_pydantic_output():
    """
    Demonstrates tasks using Pydantic models for structured output validation.
    Research task outputs validated ResearchFindings model.
    Writing task outputs validated ArticleStructure model.
    
    Returns:
        Crew execution result
    """
    researcher = Agent(
        role="Research Specialist",
        goal="Conduct research and provide structured findings",
        backstory="""You are a research specialist who provides well-structured,
                   validated research findings. You ensure all data is accurate and
                   properly formatted.""",
        verbose=True,
        tools=[search_web]
    )
    
    writer = Agent(
        role="Article Writer",
        goal="Create structured article outlines",
        backstory="""You are a writer who creates well-structured article outlines
                   before writing. You excel at planning content structure.""",
        verbose=True
    )
    
    research_task = Task(
        description="""Research 'Machine Learning in Finance' and provide structured
                      findings including topic, key points, sources, summary, and
                      confidence level.""",
        expected_output=ResearchFindings,  # Use Pydantic model
        agent=researcher
    )
    
    outline_task = Task(
        description="""Create a structured outline for an article about machine learning
                      in finance based on the research findings. Include title, introduction,
                      main sections, conclusion, and word count estimate.""",
        expected_output=ArticleStructure,  # Use Pydantic model
        agent=writer,
        context=[research_task]
    )
    
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, outline_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to demonstrate various CrewAI patterns.
    Uncomment the example you want to run.
    """
    print("CrewAI Basic Examples")
    print("=" * 50)
    
    # Example 1: Basic Sequential Crew
    # print("\n1. Basic Sequential Crew")
    # print("-" * 50)
    # result1 = basic_sequential_crew()
    # print(f"\nResult: {result1}")
    
    # Example 2: Hierarchical Crew
    # print("\n2. Hierarchical Crew with Manager")
    # print("-" * 50)
    # result2 = hierarchical_crew_with_manager()
    # print(f"\nResult: {result2}")
    
    # Example 3: Crew with Custom Tools
    # print("\n3. Crew with Custom Tools")
    # print("-" * 50)
    # result3 = crew_with_custom_tools()
    # print(f"\nResult: {result3}")
    
    # Example 4: Task Dependencies
    # print("\n4. Crew with Task Dependencies")
    # print("-" * 50)
    # result4 = crew_with_task_dependencies()
    # print(f"\nResult: {result4}")
    
    # Example 5: JSON Output
    # print("\n5. Crew with JSON Output")
    # print("-" * 50)
    # result5 = crew_with_json_output()
    # print(f"\nResult: {result5}")
    
    # Example 6: File Output
    # print("\n6. Crew with File Output")
    # print("-" * 50)
    # result6 = crew_with_file_output()
    # print(f"\nResult: {result6}")
    
    # Example 7: Pydantic Model Output
    # print("\n7. Crew with Pydantic Model Output")
    # print("-" * 50)
    # result7 = crew_with_pydantic_output()
    # print(f"\nResult: {result7}")
    
    print("\nNote: Uncomment examples in main() to run them.")
    print("Make sure to configure your LLM API keys before running.")


if __name__ == "__main__":
    main()
