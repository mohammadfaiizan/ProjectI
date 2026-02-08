"""
Main module for Data Analysis Agent.

This module provides the main entry point and high-level functions for
setting up and using the data analysis system.
"""

import os
import json
import pandas as pd
from typing import Optional, Dict, Any

from Config import LLM_Config, Analysis_Config, Execution_Config
from Agent import Data_Analysis_Graph
from Tools import Load_CSV_Data


# ============================================================================
# System Setup
# ============================================================================

def Setup_Analysis_System(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_rows_to_display: int = 100,
    chart_output_directory: str = "./charts",
    timeout_seconds: int = 30,
    max_retries: int = 3
) -> Data_Analysis_Graph:
    """
    Setup and configure the data analysis system.
    
    Args:
        model_name: OpenAI model name to use
        temperature: LLM temperature (0.0 for deterministic code generation)
        max_rows_to_display: Maximum rows to display in results
        chart_output_directory: Directory for saving charts
        timeout_seconds: Code execution timeout
        max_retries: Maximum retry attempts for failed executions
    
    Returns:
        Configured Data_Analysis_Graph instance
    """
    # Initialize configurations
    llm_config = LLM_Config(
        model_name=model_name,
        temperature=temperature
    )
    
    analysis_config = Analysis_Config(
        max_rows_to_display=max_rows_to_display,
        chart_output_directory=chart_output_directory
    )
    
    execution_config = Execution_Config(
        timeout_seconds=timeout_seconds,
        max_retries=max_retries
    )
    
    # Create and return the analysis graph
    analysis_graph = Data_Analysis_Graph(
        llm_config=llm_config,
        analysis_config=analysis_config,
        execution_config=execution_config
    )
    
    return analysis_graph


# ============================================================================
# Analysis Functions
# ============================================================================

def Analyze_Data(csv_path: str, question: str, analysis_graph: Optional[Data_Analysis_Graph] = None) -> Dict[str, Any]:
    """
    Analyze data from a CSV file by asking a question.
    
    Args:
        csv_path: Path to the CSV file to analyze
        question: Natural language question about the data
        analysis_graph: Optional pre-configured analysis graph (creates new one if None)
    
    Returns:
        Dictionary containing analysis results:
        - interpretation: Natural language answer
        - code: Generated pandas code
        - execution_result: Execution results
        - schema: Data schema
        - error: Error message if any
    """
    # Setup analysis system if not provided
    if analysis_graph is None:
        analysis_graph = Setup_Analysis_System()
    
    # Load CSV data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    load_result = Load_CSV_Data.invoke({"file_path_or_content": csv_path})
    
    if not load_result.get("success"):
        raise ValueError(f"Failed to load CSV: {load_result.get('error')}")
    
    data_json = load_result.get("dataframe_json")
    
    if not data_json:
        raise ValueError("Failed to convert data to JSON")
    
    # Ask question
    result = analysis_graph.Ask(question=question, data=data_json)
    
    return result


# ============================================================================
# Demo Function
# ============================================================================

def Run_Demo():
    """
    Run an interactive demo of the data analysis agent.
    
    Allows users to load a CSV file and ask questions interactively.
    """
    print("=" * 70)
    print("Data Analysis Agent - Interactive Demo")
    print("=" * 70)
    print()
    
    # Setup analysis system
    print("Setting up analysis system...")
    analysis_graph = Setup_Analysis_System()
    print("System ready!")
    print()
    
    # Get CSV file path
    csv_path = input("Enter path to CSV file (or press Enter for sample data): ").strip()
    
    if not csv_path:
        print("Using sample data...")
        from Sample_Input import Generate_Sample_Dataset
        df = Generate_Sample_Dataset()
        csv_path = "sample_data.csv"
        df.to_csv(csv_path, index=False)
        print(f"Sample data saved to {csv_path}")
        print()
    
    # Load data
    print(f"Loading data from {csv_path}...")
    load_result = Load_CSV_Data.invoke({"file_path_or_content": csv_path})
    
    if not load_result.get("success"):
        print(f"Error loading CSV: {load_result.get('error')}")
        return
    
    data_json = load_result.get("dataframe_json")
    schema = load_result.get("schema")
    
    print(f"Data loaded successfully!")
    print(f"  Rows: {load_result.get('row_count', 0)}")
    print(f"  Columns: {load_result.get('column_count', 0)}")
    print(f"  Column names: {', '.join(schema.get('columns', []))}")
    print()
    
    # Interactive question loop
    print("Enter questions about the data (type 'quit' or 'exit' to stop):")
    print()
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print()
        print("Analyzing...")
        
        try:
            result = analysis_graph.Ask(question=question, data=data_json)
            
            print()
            print("-" * 70)
            print("RESULT:")
            print("-" * 70)
            print(result.get("interpretation", "No interpretation available"))
            print()
            
            if result.get("code"):
                print("Generated Code:")
                print("-" * 70)
                print(result["code"])
                print()
            
            if result.get("error"):
                print(f"Warning: {result['error']}")
                print()
            
            if result.get("retry_count", 0) > 0:
                print(f"Note: Code was retried {result['retry_count']} time(s)")
                print()
        
        except Exception as e:
            print(f"Error: {str(e)}")
            print()
        
        print("=" * 70)
        print()
    
    # Cleanup sample file if created
    if csv_path == "sample_data.csv" and os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except:
            pass


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    Run_Demo()
