"""
Main module for Research Assistant system.
Provides entry point and demo functionality.
"""

import os
from typing import Optional
from Config import LLM_Config, Search_Config, Report_Config
from Agent import Research_Graph


def Setup_Research_System(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_results: int = 20,
    citation_style: str = "APA"
) -> Research_Graph:
    """
    Set up and configure the research assistant system.
    
    Args:
        model_name: LLM model name
        temperature: LLM temperature setting
        api_key: OpenAI API key (uses environment variable if not provided)
        base_url: Optional base URL for API (for custom endpoints)
        max_results: Maximum search results to collect
        citation_style: Citation style (APA, MLA, or Chicago)
        
    Returns:
        Configured Research_Graph instance
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    # Create configurations
    llm_config = LLM_Config(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )
    
    search_config = Search_Config(
        max_results_per_query=5,
        max_total_results=max_results
    )
    search_config.validate()
    
    report_config = Report_Config(
        citation_style=citation_style,
        max_sections=8,
        min_sections=3
    )
    report_config.validate()
    
    # Create and build research graph
    research_graph = Research_Graph(
        llm_config=llm_config,
        search_config=search_config,
        report_config=report_config
    )
    
    research_graph.Build_Graph()
    
    return research_graph


def Run_Research(topic: str, research_graph: Optional[Research_Graph] = None) -> str:
    """
    Execute full research pipeline for a given topic.
    
    Args:
        topic: Research topic to investigate
        research_graph: Optional pre-configured research graph
        
    Returns:
        Generated research report
    """
    if research_graph is None:
        research_graph = Setup_Research_System()
    
    print(f"\n{'='*60}")
    print(f"Starting Research: {topic}")
    print(f"{'='*60}\n")
    
    # Run research
    final_state = research_graph.Run(topic, min_sources=5)
    
    # Extract report
    report = final_state.get("report", "No report generated.")
    num_sources = final_state.get("citations", None)
    if num_sources:
        source_count = num_sources.get_source_count()
        print(f"\nResearch completed with {source_count} sources.")
    
    return report


def Run_Demo():
    """
    Run interactive demo where user can enter topics for research.
    """
    print("\n" + "="*60)
    print("Research Assistant Demo")
    print("="*60)
    print("\nThis demo allows you to research topics using AI-powered search,")
    print("content analysis, and report generation.")
    print("\nType 'quit' or 'exit' to end the demo.\n")
    
    # Setup system
    try:
        research_graph = Setup_Research_System()
        print("Research system initialized successfully.\n")
    except Exception as e:
        print(f"Error initializing research system: {e}")
        print("Please check your API key configuration.")
        return
    
    # Interactive loop
    while True:
        try:
            topic = input("Enter research topic: ").strip()
            
            if not topic:
                print("Please enter a valid topic.")
                continue
            
            if topic.lower() in ["quit", "exit", "q"]:
                print("\nExiting demo. Goodbye!")
                break
            
            # Run research
            report = Run_Research(topic, research_graph)
            
            # Display report
            print("\n" + "="*60)
            print("RESEARCH REPORT")
            print("="*60)
            print(report)
            print("="*60 + "\n")
            
            # Ask if user wants to continue
            continue_research = input("Research another topic? (y/n): ").strip().lower()
            if continue_research not in ["y", "yes"]:
                print("\nExiting demo. Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\nExiting demo. Goodbye!")
            break
        except Exception as e:
            print(f"\nError during research: {e}")
            print("Please try again with a different topic.\n")


if __name__ == "__main__":
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the research assistant.")
        print("\nYou can still run the demo, but API calls will fail.")
        print("Set the key with: export OPENAI_API_KEY='your-key-here'")
        print("\nProceeding with demo anyway...\n")
    
    # Run demo
    Run_Demo()
