"""
Main module for Autonomous Web Agent.
Provides setup and execution functions for web agent tasks.
"""

from Config import LLM_Config, Browser_Config, Agent_Config
from Agent import Web_Agent_Graph
from typing import Optional, Dict, Any


def Setup_Web_Agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.3,
    timeout: int = 30,
    max_pages: int = 10,
    max_iterations: int = 15,
    max_depth: int = 3,
    api_key: Optional[str] = None
) -> Web_Agent_Graph:
    """
    Setup and configure a web agent instance.
    
    Args:
        model_name: LLM model name
        temperature: LLM temperature
        timeout: Request timeout in seconds
        max_pages: Maximum pages to visit
        max_iterations: Maximum agent iterations
        max_depth: Maximum link following depth
        api_key: Optional API key override
        
    Returns:
        Configured Web_Agent_Graph instance
    """
    llm_config = LLM_Config(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key
    )
    
    browser_config = Browser_Config(
        timeout=timeout,
        max_pages=max_pages
    )
    
    agent_config = Agent_Config(
        max_iterations=max_iterations,
        max_depth=max_depth
    )
    
    agent = Web_Agent_Graph(
        llm_config=llm_config,
        browser_config=browser_config,
        agent_config=agent_config
    )
    
    return agent


def Execute_Web_Task(
    task: str,
    start_url: Optional[str] = None,
    agent: Optional[Web_Agent_Graph] = None,
    **agent_kwargs
) -> Dict[str, Any]:
    """
    Execute a web task using the agent.
    
    Args:
        task: Task description
        start_url: Optional starting URL
        agent: Optional pre-configured agent instance
        **agent_kwargs: Additional arguments for Setup_Web_Agent if agent not provided
        
    Returns:
        Dictionary with result and metadata
    """
    if agent is None:
        agent = Setup_Web_Agent(**agent_kwargs)
    
    print(f"Executing task: {task}")
    if start_url:
        print(f"Starting URL: {start_url}")
    print("-" * 60)
    
    result = agent.Execute_Task(task=task, start_url=start_url)
    
    print("\n" + "=" * 60)
    print("TASK COMPLETED")
    print("=" * 60)
    print(f"\nResult:\n{result['result']}")
    print(f"\nVisited {len(result['visited_urls'])} pages")
    print(f"Completed in {result['iterations']} iterations")
    
    if result['visited_urls']:
        print("\nVisited URLs:")
        for i, url in enumerate(result['visited_urls'], 1):
            print(f"  {i}. {url}")
    
    return result


def Run_Demo():
    """
    Run an interactive demo allowing user to enter tasks.
    """
    print("=" * 60)
    print("Autonomous Web Agent - Interactive Demo")
    print("=" * 60)
    print("\nEnter a task for the agent to complete.")
    print("You can optionally provide a starting URL.")
    print("Type 'quit' to exit.\n")
    
    agent = Setup_Web_Agent()
    
    while True:
        try:
            task = input("\nTask: ").strip()
            
            if task.lower() in ['quit', 'exit', 'q']:
                print("Exiting demo.")
                break
            
            if not task:
                print("Please enter a task.")
                continue
            
            start_url = input("Starting URL (optional, press Enter to skip): ").strip()
            start_url = start_url if start_url else None
            
            print("\nProcessing...")
            result = Execute_Web_Task(
                task=task,
                start_url=start_url,
                agent=agent
            )
            
            print("\n" + "-" * 60)
            response = input("\nRun another task? (y/n): ").strip().lower()
            if response != 'y':
                break
                
        except KeyboardInterrupt:
            print("\n\nExiting demo.")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    Run_Demo()
