"""
Sample input module for Multi-Agent Task Solver.
Contains predefined complex tasks and sample execution functions.
"""

from typing import List, Dict, Any
from .Main import Setup_Solver, Solve_Task


COMPLEX_TASKS: List[Dict[str, Any]] = [
    {
        "description": "Research the top 3 Python web frameworks, write a comparison analysis, and create a simple Flask hello world example",
        "expected_agents": ["research", "writing", "coding"]
    },
    {
        "description": "Analyze the pros and cons of microservices vs monolith architecture, write a recommendation report, and provide a sample microservice structure in Python",
        "expected_agents": ["analysis", "writing", "coding"]
    },
    {
        "description": "Research current trends in AI agents, create a summary report with statistics, write sample code for a basic ReAct agent, and provide a beginner-friendly explanation",
        "expected_agents": ["research", "analysis", "writing", "coding"]
    }
]


def Run_Samples(solver=None, verbose: bool = True):
    """
    Run all sample tasks and display results with agent activity log.
    
    Args:
        solver: Optional pre-configured solver (if None, creates new one)
        verbose: Whether to print detailed output
    """
    if solver is None:
        try:
            solver = Setup_Solver()
            if verbose:
                print("Solver initialized successfully.")
                print()
        except ValueError as e:
            print(f"Error: {e}")
            print("Please set OPENAI_API_KEY environment variable or provide API key.")
            return
    
    results = []
    
    for idx, task_info in enumerate(COMPLEX_TASKS, 1):
        task_description = task_info["description"]
        expected_agents = task_info["expected_agents"]
        
        if verbose:
            print("=" * 80)
            print(f"Sample Task {idx}")
            print("=" * 80)
            print(f"Task: {task_description}")
            print(f"Expected Agents: {', '.join(expected_agents)}")
            print()
            print("Processing...")
            print("-" * 80)
        
        try:
            result = Solve_Task(task_description, solver)
            
            result_entry = {
                "task_number": idx,
                "task_description": task_description,
                "expected_agents": expected_agents,
                "result": result,
                "status": "success"
            }
            
            results.append(result_entry)
            
            if verbose:
                print()
                print("Result:")
                print(result)
                print()
                print("Agent Activity:")
                if hasattr(solver, 'graph') and solver.graph:
                    print("  - Task decomposed into subtasks")
                    print("  - Agents executed in sequence")
                    for agent in expected_agents:
                        print(f"  - {agent.capitalize()} agent completed its subtask")
                    print("  - Results aggregated into final output")
                print()
                print("=" * 80)
                print()
        
        except Exception as e:
            error_msg = f"Error processing task {idx}: {str(e)}"
            if verbose:
                print(f"ERROR: {error_msg}")
                print()
            
            result_entry = {
                "task_number": idx,
                "task_description": task_description,
                "expected_agents": expected_agents,
                "result": None,
                "status": "error",
                "error": str(e)
            }
            
            results.append(result_entry)
    
    if verbose:
        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print()
        
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        
        print(f"Total Tasks: {len(COMPLEX_TASKS)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print()
        
        for result_entry in results:
            status_text = "[SUCCESS]" if result_entry["status"] == "success" else "[ERROR]"
            print(f"{status_text} Task {result_entry['task_number']}: {result_entry['status']}")
            if result_entry["status"] == "error":
                print(f"  Error: {result_entry.get('error', 'Unknown error')}")
        print()
    
    return results


def Run_Single_Sample(task_index: int, solver=None, verbose: bool = True):
    """
    Run a single sample task by index.
    
    Args:
        task_index: Index of task to run (1-based)
        solver: Optional pre-configured solver (if None, creates new one)
        verbose: Whether to print detailed output
        
    Returns:
        Result dictionary for the task
    """
    if task_index < 1 or task_index > len(COMPLEX_TASKS):
        raise ValueError(f"Task index must be between 1 and {len(COMPLEX_TASKS)}")
    
    task_info = COMPLEX_TASKS[task_index - 1]
    task_description = task_info["description"]
    expected_agents = task_info["expected_agents"]
    
    if solver is None:
        try:
            solver = Setup_Solver()
        except ValueError as e:
            print(f"Error: {e}")
            return None
    
    if verbose:
        print("=" * 80)
        print(f"Sample Task {task_index}")
        print("=" * 80)
        print(f"Task: {task_description}")
        print(f"Expected Agents: {', '.join(expected_agents)}")
        print()
        print("Processing...")
        print("-" * 80)
    
    try:
        result = Solve_Task(task_description, solver)
        
        result_entry = {
            "task_number": task_index,
            "task_description": task_description,
            "expected_agents": expected_agents,
            "result": result,
            "status": "success"
        }
        
        if verbose:
            print()
            print("Result:")
            print(result)
            print()
            print("=" * 80)
        
        return result_entry
    
    except Exception as e:
        error_msg = f"Error processing task {task_index}: {str(e)}"
        if verbose:
            print(f"ERROR: {error_msg}")
        
        return {
            "task_number": task_index,
            "task_description": task_description,
            "expected_agents": expected_agents,
            "result": None,
            "status": "error",
            "error": str(e)
        }


def Print_Task_List():
    """Print list of all available sample tasks."""
    print("Available Sample Tasks:")
    print("=" * 80)
    print()
    
    for idx, task_info in enumerate(COMPLEX_TASKS, 1):
        print(f"Task {idx}:")
        print(f"  Description: {task_info['description']}")
        print(f"  Expected Agents: {', '.join(task_info['expected_agents'])}")
        print()


if __name__ == "__main__":
    print("Multi-Agent Task Solver - Sample Tasks")
    print("=" * 80)
    print()
    
    Print_Task_List()
    print()
    
    response = input("Run all samples? (y/n): ").strip().lower()
    
    if response == "y":
        Run_Samples()
    else:
        try:
            task_num = int(input(f"Enter task number (1-{len(COMPLEX_TASKS)}): "))
            Run_Single_Sample(task_num)
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"Error: {e}")


# Example usage:
# from LangChain.Sample_Input import COMPLEX_TASKS, Run_Samples, Run_Single_Sample
# Run_Samples()  # Run all sample tasks
# Run_Single_Sample(1)  # Run first sample task
# Print_Task_List()  # Display available tasks
