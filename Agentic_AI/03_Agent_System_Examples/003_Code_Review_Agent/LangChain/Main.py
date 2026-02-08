"""
Main module for Code Review Agent.
Entry point for running code reviews.
"""

from Config import LLM_Config, Review_Config
from Agent import Code_Review_Graph
from typing import Optional
import os


def Setup_Review_System(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    api_key: Optional[str] = None
) -> Code_Review_Graph:
    """
    Setup and configure the code review system.
    
    Args:
        model_name: Name of the LLM model to use
        temperature: Temperature for LLM (0.0 for deterministic reviews)
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        
    Returns:
        Configured Code_Review_Graph instance
    """
    llm_config = LLM_Config(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key
    )
    
    review_config = Review_Config()
    llm = llm_config.Get_LLM()
    
    review_graph = Code_Review_Graph(llm=llm, review_config=review_config)
    
    return review_graph


def Review_Code(code_string: str, review_system: Optional[Code_Review_Graph] = None) -> dict:
    """
    Review a code string and return detailed report.
    
    Args:
        code_string: Python source code to review
        review_system: Optional pre-configured review system
        
    Returns:
        Dictionary containing review results
    """
    if review_system is None:
        review_system = Setup_Review_System()
    
    result = review_system.Review(code_string)
    
    print("=" * 80)
    print("CODE REVIEW REPORT")
    print("=" * 80)
    print(f"\n{result['summary']}\n")
    
    all_issues = (
        result.get("bug_issues", []) +
        result.get("security_issues", []) +
        result.get("style_issues", []) +
        result.get("performance_issues", [])
    )
    
    if all_issues:
        print("\n" + "=" * 80)
        print("DETAILED ISSUES")
        print("=" * 80)
        
        for i, issue in enumerate(all_issues, 1):
            print(f"\n[{i}] {issue.get('severity', 'UNKNOWN')} - {issue.get('category', 'unknown').upper()}")
            print(f"    Description: {issue.get('description', 'N/A')}")
            if issue.get('line_number'):
                print(f"    Line: {issue['line_number']}")
            if issue.get('suggestion'):
                print(f"    Suggestion: {issue['suggestion']}")
    
    print("\n" + "=" * 80)
    
    return result


def Review_File(file_path: str, review_system: Optional[Code_Review_Graph] = None) -> dict:
    """
    Review a Python file and return detailed report.
    
    Args:
        file_path: Path to Python file to review
        review_system: Optional pre-configured review system
        
    Returns:
        Dictionary containing review results
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        code_content = f.read()
    
    print(f"\nReviewing file: {file_path}\n")
    
    return Review_Code(code_content, review_system)


def Run_Demo():
    """
    Run a demonstration of the code review system with sample code.
    """
    print("Code Review Agent - Demo Mode")
    print("=" * 80)
    
    sample_code = """
def calculate_total(items):
    total = 0
    for i in range(len(items)):
        total = total + items[i]
    return total

def process_user_input(user_input):
    result = eval(user_input)
    return result

def authenticate(username, password):
    if username == "admin" and password == "password123":
        return True
    return False
"""
    
    print("\nSample code to review:")
    print("-" * 80)
    print(sample_code)
    print("-" * 80)
    
    try:
        review_system = Setup_Review_System()
        result = Review_Code(sample_code, review_system)
        
        print("\nDemo completed successfully!")
        return result
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        print("Make sure OPENAI_API_KEY environment variable is set.")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            Review_File(file_path)
        except Exception as e:
            print(f"Error reviewing file: {str(e)}")
    else:
        Run_Demo()
