"""
Sample Input module for Autonomous Web Agent.
Contains predefined tasks and sample execution functions.
"""

from Main import Setup_Web_Agent, Execute_Web_Task
from typing import List, Dict, Any, Optional


WEB_TASKS: List[Dict[str, Any]] = [
    {
        "task": "Find information about Python 3.12 new features",
        "start_url": "https://docs.python.org",
        "description": "Extract key features from Python documentation",
        "expected_output_type": "summary"
    },
    {
        "task": "Research the latest trends in AI agent frameworks",
        "start_url": None,
        "description": "Search and summarize AI framework landscape",
        "expected_output_type": "list"
    },
    {
        "task": "Extract pricing information from a SaaS comparison page",
        "start_url": "https://example.com/pricing",
        "description": "Find and compare pricing tiers",
        "expected_output_type": "table"
    }
]


def Run_Samples():
    """
    Execute all sample tasks with mock content and print results.
    """
    print("=" * 80)
    print("Autonomous Web Agent - Sample Tasks Execution")
    print("=" * 80)
    
    agent = Setup_Web_Agent(
        model_name="gpt-4o-mini",
        temperature=0.3,
        max_iterations=10,
        max_depth=2
    )
    
    results = []
    
    for i, task_config in enumerate(WEB_TASKS, 1):
        print(f"\n{'=' * 80}")
        print(f"SAMPLE TASK {i}/{len(WEB_TASKS)}")
        print(f"{'=' * 80}")
        print(f"Task: {task_config['task']}")
        print(f"Description: {task_config['description']}")
        print(f"Start URL: {task_config['start_url'] or 'None (will search)'}")
        print(f"Expected Output Type: {task_config['expected_output_type']}")
        print("-" * 80)
        
        try:
            result = Execute_Web_Task(
                task=task_config['task'],
                start_url=task_config['start_url'],
                agent=agent
            )
            
            results.append({
                "task_config": task_config,
                "result": result,
                "success": True
            })
            
            print(f"\n✓ Task {i} completed successfully")
            
        except Exception as e:
            print(f"\n✗ Task {i} failed with error: {str(e)}")
            results.append({
                "task_config": task_config,
                "result": None,
                "success": False,
                "error": str(e)
            })
        
        print("\n" + "-" * 80)
    
    # Summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\nTotal Tasks: {len(WEB_TASKS)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print("\nSuccessful Tasks:")
        for i, result in enumerate(results, 1):
            if result['success']:
                task_config = result['task_config']
                execution_result = result['result']
                print(f"\n  {i}. {task_config['task']}")
                print(f"     Visited {len(execution_result['visited_urls'])} pages")
                print(f"     Completed in {execution_result['iterations']} iterations")
                print(f"     Result Preview: {execution_result['result'][:150]}...")
    
    if failed > 0:
        print("\nFailed Tasks:")
        for i, result in enumerate(results, 1):
            if not result['success']:
                task_config = result['task_config']
                print(f"\n  {i}. {task_config['task']}")
                print(f"     Error: {result.get('error', 'Unknown error')}")
    
    return results


def Run_Single_Sample(task_index: int):
    """
    Run a single sample task by index.
    
    Args:
        task_index: Index of task in WEB_TASKS (1-based)
    """
    if task_index < 1 or task_index > len(WEB_TASKS):
        print(f"Invalid task index. Must be between 1 and {len(WEB_TASKS)}")
        return
    
    task_config = WEB_TASKS[task_index - 1]
    
    print("=" * 80)
    print(f"Running Sample Task {task_index}")
    print("=" * 80)
    print(f"Task: {task_config['task']}")
    print(f"Description: {task_config['description']}")
    print(f"Start URL: {task_config['start_url'] or 'None'}")
    print("-" * 80)
    
    agent = Setup_Web_Agent(
        model_name="gpt-4o-mini",
        temperature=0.3,
        max_iterations=10,
        max_depth=2
    )
    
    result = Execute_Web_Task(
        task=task_config['task'],
        start_url=task_config['start_url'],
        agent=agent
    )
    
    return result


def Get_Task_Description(task_index: int) -> Optional[Dict[str, Any]]:
    """
    Get description of a sample task.
    
    Args:
        task_index: Index of task in WEB_TASKS (1-based)
        
    Returns:
        Task configuration dictionary or None if invalid index
    """
    if task_index < 1 or task_index > len(WEB_TASKS):
        return None
    
    return WEB_TASKS[task_index - 1]


def List_All_Tasks():
    """
    List all available sample tasks.
    """
    print("=" * 80)
    print("Available Sample Tasks")
    print("=" * 80)
    
    for i, task_config in enumerate(WEB_TASKS, 1):
        print(f"\n{i}. {task_config['task']}")
        print(f"   Description: {task_config['description']}")
        print(f"   Start URL: {task_config['start_url'] or 'None'}")
        print(f"   Expected Output: {task_config['expected_output_type']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "list":
            List_All_Tasks()
        elif command == "run":
            if len(sys.argv) > 2:
                try:
                    task_index = int(sys.argv[2])
                    Run_Single_Sample(task_index)
                except ValueError:
                    print("Invalid task index. Must be a number.")
            else:
                Run_Samples()
        else:
            print("Usage:")
            print("  python Sample_Input.py list          - List all tasks")
            print("  python Sample_Input.py run           - Run all tasks")
            print("  python Sample_Input.py run <index>   - Run specific task")
    else:
        Run_Samples()
