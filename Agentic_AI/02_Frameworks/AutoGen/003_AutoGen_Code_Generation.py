"""
AutoGen Code Generation Examples

This module demonstrates code generation and execution workflows using AutoGen:
1. Code Generation Agent: Agent that writes Python code
2. Code Execution with Feedback: Execute code and feed errors back to agent
3. Data Analysis Workflow: Generate pandas code to analyze CSV data
4. Multi-Agent Code Review: Coder, reviewer, and tester agents collaborate
5. Math Problem Solving: Solve math problems using code execution
6. Safe Code Execution: Docker-based or local sandbox execution
"""

import os
import json
from typing import Dict, Any, List
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager


# ============================================================================
# Configuration
# ============================================================================

def get_llm_config() -> Dict[str, Any]:
    """
    Get LLM configuration for agents.
    Modify this to use your preferred LLM provider and API keys.
    """
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    config_list = [
        {
            "model": "gpt-4",
            "api_key": api_key,
            "base_url": None,
            "api_type": "open_ai"
        }
    ]
    
    return {
        "config_list": config_list,
        "temperature": 0.7,
        "timeout": 120,
        "max_tokens": 2000
    }


# ============================================================================
# Example 1: Code Generation Agent
# ============================================================================

def example_code_generation_agent():
    """
    Basic code generation: Agent writes Python code based on requirements.
    """
    print("\n" + "="*80)
    print("Example 1: Code Generation Agent")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    # Create code generation agent
    coder = AssistantAgent(
        name="coder",
        system_message="You are an expert Python programmer. You write clean, well-documented code. "
                      "Always provide complete, runnable code blocks with proper error handling. "
                      "Include comments explaining key logic.",
        llm_config=llm_config
    )
    
    # Create UserProxyAgent for code execution
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    task = """
    Write a Python class called 'BankAccount' with the following features:
    1. Initialize with account holder name and initial balance
    2. Methods: deposit(amount), withdraw(amount), get_balance()
    3. Prevent withdrawals that would result in negative balance
    4. Include proper error handling
    
    Then create an instance, deposit 1000, withdraw 300, and display the balance.
    """
    
    print(f"Task: {task}\n")
    print("Starting code generation...\n")
    
    user_proxy.initiate_chat(
        coder,
        message=task
    )
    
    print("\n" + "-"*80)
    print("Code Generation Example Complete")
    print("-"*80 + "\n")


# ============================================================================
# Example 2: Code Execution with Feedback
# ============================================================================

def example_code_execution_with_feedback():
    """
    Generate code, execute it, and feed errors back to agent for iterative improvement.
    """
    print("\n" + "="*80)
    print("Example 2: Code Execution with Feedback")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    coder = AssistantAgent(
        name="coder",
        system_message="You are a Python programmer. When code execution fails, "
                      "analyze the error message and fix the code. Iterate until the code works correctly.",
        llm_config=llm_config
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=15,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    task = """
    Write a Python function that reads a JSON file, parses it, and returns the data.
    The function should handle file not found errors and JSON parsing errors gracefully.
    
    Then test it by:
    1. Creating a sample JSON file with data: {"name": "Test", "value": 42}
    2. Reading and parsing it
    3. Printing the parsed data
    
    If there are any errors, fix them and try again.
    """
    
    print(f"Task: {task}\n")
    print("Starting code generation with iterative feedback...\n")
    
    user_proxy.initiate_chat(
        coder,
        message=task
    )
    
    print("\n" + "-"*80)
    print("Code Execution with Feedback Example Complete")
    print("-"*80 + "\n")


# ============================================================================
# Example 3: Data Analysis Workflow
# ============================================================================

def example_data_analysis_workflow():
    """
    Agent generates pandas code to analyze CSV data.
    Creates sample data, analyzes it, and visualizes results.
    """
    print("\n" + "="*80)
    print("Example 3: Data Analysis Workflow")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    data_analyst = AssistantAgent(
        name="data_analyst",
        system_message="You are a data analyst expert. You use pandas, numpy, and matplotlib "
                      "to analyze data. Write complete, executable code that includes "
                      "data loading, cleaning, analysis, and visualization.",
        llm_config=llm_config
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=20,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    # First, create sample CSV data
    print("Creating sample CSV data...\n")
    
    sample_data_code = """
import csv
import os

# Create sample sales data
data = [
    ["Date", "Product", "Sales", "Region"],
    ["2024-01-01", "Product A", 1500, "North"],
    ["2024-01-02", "Product B", 2300, "South"],
    ["2024-01-03", "Product A", 1800, "North"],
    ["2024-01-04", "Product C", 1200, "East"],
    ["2024-01-05", "Product B", 2500, "South"],
    ["2024-01-06", "Product A", 1600, "West"],
    ["2024-01-07", "Product C", 1400, "East"],
    ["2024-01-08", "Product B", 2200, "North"],
]

os.makedirs("coding", exist_ok=True)
with open("coding/sales_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("Sample CSV file created: coding/sales_data.csv")
"""
    
    exec(sample_data_code)
    
    task = """
    Analyze the CSV file 'coding/sales_data.csv' which contains sales data.
    
    Perform the following analysis:
    1. Load the CSV file using pandas
    2. Display basic statistics (mean, sum, etc.) for sales
    3. Group by Product and calculate total sales per product
    4. Group by Region and calculate total sales per region
    5. Find the product with the highest total sales
    6. Display the results in a clear format
    
    Write complete, executable code.
    """
    
    print(f"Task: {task}\n")
    print("Starting data analysis...\n")
    
    user_proxy.initiate_chat(
        data_analyst,
        message=task
    )
    
    print("\n" + "-"*80)
    print("Data Analysis Workflow Example Complete")
    print("-"*80 + "\n")


# ============================================================================
# Example 4: Multi-Agent Code Review
# ============================================================================

def example_multi_agent_code_review():
    """
    Multiple agents collaborate: coder writes code, reviewer reviews it, tester suggests tests.
    """
    print("\n" + "="*80)
    print("Example 4: Multi-Agent Code Review")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    # Create specialized agents
    coder = AssistantAgent(
        name="coder",
        system_message="You are a Python developer. Write clean, efficient code following best practices. "
                      "Respond to feedback from reviewers and testers.",
        llm_config=llm_config
    )
    
    reviewer = AssistantAgent(
        name="reviewer",
        system_message="You are a code reviewer. Analyze code for: "
                      "- Bugs and logic errors\n"
                      "- Performance issues\n"
                      "- Code style and readability\n"
                      "- Best practices\n"
                      "Provide constructive, specific feedback.",
        llm_config=llm_config
    )
    
    tester = AssistantAgent(
        name="tester",
        system_message="You are a QA tester. Think about: "
                      "- Edge cases\n"
                      "- Error scenarios\n"
                      "- Test coverage\n"
                      "- Integration testing\n"
                      "Suggest comprehensive test cases.",
        llm_config=llm_config
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=20,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    # Create GroupChat
    groupchat = GroupChat(
        agents=[coder, reviewer, tester, user_proxy],
        messages=[],
        max_round=15
    )
    
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
        system_message="You manage a code review session. Coordinate between coder, reviewer, and tester. "
                      "Ensure the code is written, reviewed, and tested thoroughly."
    )
    
    task = """
    Develop a Python class for a 'ShoppingCart' with the following requirements:
    
    1. Add items with name, price, and quantity
    2. Remove items
    3. Calculate total price
    4. Apply discount percentage
    5. Display cart contents
    
    Workflow:
    - Coder: Write the initial implementation
    - Reviewer: Review the code and provide feedback
    - Tester: Suggest test cases
    - Coder: Incorporate feedback and improve the code
    - User_proxy: Execute the final code and tests
    
    Make sure the code is production-ready.
    """
    
    print(f"Task: {task}\n")
    print("Starting multi-agent code review...\n")
    
    user_proxy.initiate_chat(
        manager,
        message=task
    )
    
    print("\n" + "-"*80)
    print("Multi-Agent Code Review Example Complete")
    print("-"*80 + "\n")


# ============================================================================
# Example 5: Math Problem Solving
# ============================================================================

def example_math_problem_solving():
    """
    Agent solves mathematical problems by generating and executing code.
    """
    print("\n" + "="*80)
    print("Example 5: Math Problem Solving")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    math_solver = AssistantAgent(
        name="math_solver",
        system_message="You are a mathematical problem solver. When given a math problem, "
                      "write Python code to solve it. Use appropriate libraries (numpy, scipy, etc.) "
                      "when needed. Show your work step by step.",
        llm_config=llm_config
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=15,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    problems = [
        """
        Problem 1: Find all prime numbers between 1 and 100.
        Write code to generate and display them.
        """,
        """
        Problem 2: Calculate the factorial of 20.
        Use both iterative and recursive approaches, then compare performance.
        """,
        """
        Problem 3: Solve the quadratic equation: x^2 - 5x + 6 = 0
        Find all real solutions.
        """,
        """
        Problem 4: Calculate the first 10 numbers in the Fibonacci sequence.
        Then find the sum of all even Fibonacci numbers below 1000.
        """
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\n--- Problem {i} ---\n")
        print(f"{problem}\n")
        print("Solving...\n")
        
        user_proxy.initiate_chat(
            math_solver,
            message=problem,
            clear_history=(i == 1)  # Clear history only for first problem
        )
        
        print("\n" + "-"*40 + "\n")
    
    print("\n" + "-"*80)
    print("Math Problem Solving Example Complete")
    print("-"*80 + "\n")


# ============================================================================
# Example 6: Safe Code Execution
# ============================================================================

def example_safe_code_execution():
    """
    Demonstrate safe code execution using Docker or local sandbox.
    Configure execution environment for security.
    """
    print("\n" + "="*80)
    print("Example 6: Safe Code Execution")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    coder = AssistantAgent(
        name="coder",
        system_message="You are a Python programmer. Write safe, secure code. "
                      "Avoid dangerous operations like file system access outside work_dir, "
                      "network calls, or system modifications.",
        llm_config=llm_config
    )
    
    # Configuration 1: Local execution (faster, less isolated)
    print("--- Configuration 1: Local Execution ---\n")
    
    user_proxy_local = UserProxyAgent(
        name="user_proxy_local",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,
            "timeout": 60
        }
    )
    
    task_local = """
    Write a Python function that:
    1. Creates a list of 100 random integers between 1 and 1000
    2. Sorts the list
    3. Finds the median value
    4. Saves the sorted list to a file in the work directory
    
    Execute the code and show results.
    """
    
    print(f"Task: {task_local}\n")
    print("Running with local execution...\n")
    
    user_proxy_local.initiate_chat(
        coder,
        message=task_local
    )
    
    print("\n" + "-"*40 + "\n")
    
    # Configuration 2: Docker execution (safer, isolated)
    print("--- Configuration 2: Docker Execution (Commented) ---\n")
    print("""
    To use Docker execution, uncomment and configure:
    
    user_proxy_docker = UserProxyAgent(
        name="user_proxy_docker",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": True,
            "docker_image": "python:3.11",
            "timeout": 60
        }
    )
    
    Docker execution provides:
    - Complete isolation from host system
    - Reproducible environments
    - Enhanced security for untrusted code
    - Clean state for each execution
    """)
    
    # Example of Docker configuration (not executed)
    docker_config_example = """
    # Docker execution configuration example
    docker_config = {
        "work_dir": "coding",
        "use_docker": True,
        "docker_image": "python:3.11",
        "timeout": 60,
        "docker_timeout": 300  # Timeout for Docker operations
    }
    
    user_proxy_docker = UserProxyAgent(
        name="user_proxy_docker",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=docker_config
    )
    """
    
    print(docker_config_example)
    
    print("\n" + "-"*80)
    print("Safe Code Execution Example Complete")
    print("-"*80 + "\n")


# ============================================================================
# Example 7: Advanced Code Generation Workflow
# ============================================================================

def example_advanced_code_generation_workflow():
    """
    Complex workflow: Generate code, test it, refactor it, and document it.
    """
    print("\n" + "="*80)
    print("Example 7: Advanced Code Generation Workflow")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    # Create workflow agents
    architect = AssistantAgent(
        name="architect",
        system_message="You are a software architect. Design the overall structure and API. "
                      "Think about scalability, maintainability, and best practices.",
        llm_config=llm_config
    )
    
    implementer = AssistantAgent(
        name="implementer",
        system_message="You are a developer. Implement code based on architectural designs. "
                      "Write clean, efficient, well-documented code.",
        llm_config=llm_config
    )
    
    tester = AssistantAgent(
        name="tester",
        system_message="You are a test engineer. Write comprehensive unit tests. "
                      "Ensure good test coverage and edge case handling.",
        llm_config=llm_config
    )
    
    refactorer = AssistantAgent(
        name="refactorer",
        system_message="You are a code refactoring specialist. Improve code quality: "
                      "- Optimize performance\n"
                      "- Improve readability\n"
                      "- Reduce complexity\n"
                      "- Apply design patterns where appropriate",
        llm_config=llm_config
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=25,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    groupchat = GroupChat(
        agents=[architect, implementer, tester, refactorer, user_proxy],
        messages=[],
        max_round=20
    )
    
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
        system_message="You manage a software development workflow. Coordinate: "
                      "1. Architect designs the solution\n"
                      "2. Implementer writes the code\n"
                      "3. Tester writes and runs tests\n"
                      "4. Refactorer improves the code\n"
                      "5. User_proxy executes code and tests\n"
                      "Iterate until the solution is complete and tested."
    )
    
    task = """
    Develop a complete Python module for a 'TaskManager' system:
    
    Requirements:
    1. Task class with: id, title, description, priority, status, due_date
    2. TaskManager class with methods:
       - add_task(task)
       - remove_task(task_id)
       - update_task(task_id, updates)
       - get_tasks_by_priority(priority)
       - get_overdue_tasks()
       - save_to_file(filename)
       - load_from_file(filename)
    
    Workflow:
    1. Architect: Design the class structure and API
    2. Implementer: Write the implementation
    3. Tester: Write unit tests
    4. Refactorer: Optimize and improve the code
    5. User_proxy: Execute tests and verify functionality
    
    Make it production-ready with proper error handling and documentation.
    """
    
    print(f"Task: {task}\n")
    print("Starting advanced code generation workflow...\n")
    
    user_proxy.initiate_chat(
        manager,
        message=task
    )
    
    print("\n" + "-"*80)
    print("Advanced Code Generation Workflow Example Complete")
    print("-"*80 + "\n")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Run all code generation examples.
    Comment out examples you don't want to run.
    """
    print("\n" + "#"*80)
    print("# AutoGen Code Generation Examples")
    print("#"*80)
    
    # Run examples
    try:
        example_code_generation_agent()
    except Exception as e:
        print(f"Error in example_code_generation_agent: {e}\n")
    
    try:
        example_code_execution_with_feedback()
    except Exception as e:
        print(f"Error in example_code_execution_with_feedback: {e}\n")
    
    try:
        example_data_analysis_workflow()
    except Exception as e:
        print(f"Error in example_data_analysis_workflow: {e}\n")
    
    try:
        example_multi_agent_code_review()
    except Exception as e:
        print(f"Error in example_multi_agent_code_review: {e}\n")
    
    try:
        example_math_problem_solving()
    except Exception as e:
        print(f"Error in example_math_problem_solving: {e}\n")
    
    try:
        example_safe_code_execution()
    except Exception as e:
        print(f"Error in example_safe_code_execution: {e}\n")
    
    try:
        example_advanced_code_generation_workflow()
    except Exception as e:
        print(f"Error in example_advanced_code_generation_workflow: {e}\n")
    
    print("\n" + "#"*80)
    print("# All Code Generation Examples Complete")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
