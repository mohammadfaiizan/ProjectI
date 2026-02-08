"""
Main entry point for Multi-Agent Task Solver.
Provides setup and execution functions for interactive use.
"""

import os
from typing import Optional
from .Config import LLM_Config, Agent_Config, Routing_Config
from .Agent import Multi_Agent_Graph


def Setup_Solver(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_iterations: int = 10
) -> Multi_Agent_Graph:
    """
    Setup and initialize the multi-agent solver.
    
    Args:
        api_key: OpenAI API key (or None to use OPENAI_API_KEY env var)
        base_url: Optional base URL for API (for custom endpoints)
        max_iterations: Maximum number of iterations before stopping
        
    Returns:
        Configured Multi_Agent_Graph instance
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key is None:
        raise ValueError(
            "API key must be provided either as argument or OPENAI_API_KEY environment variable"
        )
    
    llm_config = LLM_Config(api_key=api_key, base_url=base_url)
    agent_config = Agent_Config(max_iterations=max_iterations)
    routing_config = Routing_Config()
    
    solver = Multi_Agent_Graph(
        llm_config=llm_config,
        agent_config=agent_config,
        routing_config=routing_config
    )
    
    solver.Build_Graph()
    
    return solver


def Solve_Task(task_description: str, solver: Optional[Multi_Agent_Graph] = None) -> str:
    """
    Solve a task using the multi-agent solver.
    
    Args:
        task_description: Description of the task to solve
        solver: Optional pre-configured solver (if None, creates new one)
        
    Returns:
        Final aggregated result from the solver
    """
    if solver is None:
        solver = Setup_Solver()
    
    print(f"Solving task: {task_description}")
    print("Processing with multi-agent system...")
    
    result = solver.Solve(task_description)
    
    return result


def Run_Demo():
    """
    Run interactive demo allowing user to input tasks.
    """
    print("=" * 60)
    print("Multi-Agent Task Solver Demo")
    print("=" * 60)
    print()
    
    try:
        solver = Setup_Solver()
        print("Solver initialized successfully.")
        print()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set OPENAI_API_KEY environment variable or provide API key.")
        return
    
    print("Enter tasks to solve (or 'quit' to exit):")
    print()
    
    while True:
        task = input("Task: ").strip()
        
        if task.lower() in ["quit", "exit", "q"]:
            print("Exiting demo.")
            break
        
        if not task:
            print("Please enter a valid task description.")
            continue
        
        print()
        print("-" * 60)
        result = Solve_Task(task, solver)
        print("-" * 60)
        print()
        print("Result:")
        print(result)
        print()
        print("=" * 60)
        print()


if __name__ == "__main__":
    Run_Demo()


# Example usage:
# from LangChain.Main import Setup_Solver, Solve_Task
# solver = Setup_Solver(api_key="your-api-key")
# result = Solve_Task("Your task description here", solver)
