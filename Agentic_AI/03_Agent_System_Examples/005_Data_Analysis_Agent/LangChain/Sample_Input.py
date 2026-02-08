"""
Sample Input module for Data Analysis Agent.

This module provides sample datasets and questions for testing and
demonstrating the data analysis agent capabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

from Config import LLM_Config, Analysis_Config, Execution_Config
from Agent import Data_Analysis_Graph
from Tools import Load_CSV_Data


# ============================================================================
# Sample Dataset Generation
# ============================================================================

def Generate_Sample_Dataset() -> pd.DataFrame:
    """
    Generate a realistic sales dataset with 500 rows.
    
    Creates a dataset with columns:
    - date: Transaction dates
    - product: Product names
    - category: Product categories
    - quantity: Quantity sold
    - price: Unit price
    - region: Sales region
    - salesperson: Salesperson name
    
    The dataset includes realistic patterns such as:
    - Seasonal variations in sales
    - Regional preferences for certain products
    - Salesperson performance variations
    - Price variations based on product category
    - Missing values to simulate real-world data quality issues
    
    Returns:
        pandas DataFrame with 500 rows of sales data
    """
    np.random.seed(42)
    
    # Define product categories and products with detailed mappings
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
    products_by_category = {
        "Electronics": ["Laptop", "Smartphone", "Tablet", "Headphones", "Monitor"],
        "Clothing": ["T-Shirt", "Jeans", "Jacket", "Shoes", "Hat"],
        "Home & Garden": ["Chair", "Table", "Lamp", "Plant", "Tool"],
        "Sports": ["Basketball", "Tennis Racket", "Yoga Mat", "Dumbbells", "Bike"],
        "Books": ["Novel", "Textbook", "Cookbook", "Biography", "Comic"]
    }
    
    # Define regions with different characteristics
    regions = ["North", "South", "East", "West", "Central"]
    
    # Sales team with varying performance levels
    salespeople = [
        "Alice Johnson", "Bob Smith", "Carol Williams", "David Brown",
        "Emma Davis", "Frank Miller", "Grace Wilson", "Henry Moore"
    ]
    
    # Generate dates (last 6 months) with realistic distribution
    start_date = datetime.now() - timedelta(days=180)
    dates = []
    
    # Create more realistic date distribution (more recent dates have more sales)
    for i in range(500):
        # Weight towards more recent dates
        days_offset = int(np.random.exponential(scale=30))
        days_offset = min(days_offset, 180)
        date = start_date + timedelta(days=days_offset)
        dates.append(date)
    
    # Sort dates to maintain chronological order
    dates.sort()
    
    # Generate data with realistic patterns
    data = []
    for i in range(500):
        category = np.random.choice(categories)
        product = np.random.choice(products_by_category[category])
        region = np.random.choice(regions)
        salesperson = np.random.choice(salespeople)
        
        # Generate realistic quantities (1-50, with some outliers)
        # Higher quantities are less common
        quantity = np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50],
            p=[0.15, 0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.005]
        )
        
        # Generate realistic prices based on category with some variance
        base_prices = {
            "Electronics": (200, 2000),
            "Clothing": (20, 200),
            "Home & Garden": (30, 500),
            "Sports": (25, 300),
            "Books": (10, 50)
        }
        min_price, max_price = base_prices[category]
        
        # Add some price variation based on region (simulate regional pricing)
        region_multipliers = {
            "North": 1.1,
            "South": 0.95,
            "East": 1.05,
            "West": 1.0,
            "Central": 0.98
        }
        multiplier = region_multipliers.get(region, 1.0)
        
        price = np.random.uniform(min_price, max_price) * multiplier
        price = round(price, 2)
        
        # Add seasonal effects (higher sales in certain months)
        month = dates[i].month
        seasonal_multiplier = 1.0
        if month in [11, 12, 1]:  # Holiday season
            seasonal_multiplier = 1.2
        elif month in [6, 7, 8]:  # Summer
            seasonal_multiplier = 1.1
        
        # Apply seasonal effect to quantity
        quantity = max(1, int(quantity * seasonal_multiplier))
        
        data.append({
            "date": dates[i],
            "product": product,
            "category": category,
            "quantity": quantity,
            "price": price,
            "region": region,
            "salesperson": salesperson
        })
    
    df = pd.DataFrame(data)
    
    # Add some missing values to simulate real-world data quality issues
    # 5% missing in quantity, 2% in price
    missing_indices_qty = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
    df.loc[missing_indices_qty, "quantity"] = np.nan
    
    missing_indices_price = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df.loc[missing_indices_price, "price"] = np.nan
    
    # Add a few duplicate entries (simulating data entry errors)
    duplicate_indices = np.random.choice(df.index, size=int(len(df) * 0.01), replace=False)
    for idx in duplicate_indices:
        duplicate_row = df.loc[idx].copy()
        df = pd.concat([df, duplicate_row.to_frame().T], ignore_index=True)
    
    # Reset index to ensure clean sequential indexing
    df = df.reset_index(drop=True)
    
    return df


# ============================================================================
# Analysis Questions
# ============================================================================

ANALYSIS_QUESTIONS = [
    "What is the total revenue by product category?",
    "Which salesperson had the highest sales in Q1?",
    "Show monthly revenue trend",
    "What is the average order value by region?",
    "Find the top 5 best-selling products",
    "What percentage of sales come from each region?",
    "Which month had the highest total sales?",
    "Calculate the correlation between quantity and price"
]


# ============================================================================
# Sample Runner
# ============================================================================

def Run_Samples():
    """
    Generate sample data and run all analysis questions.
    
    Demonstrates the data analysis agent by:
    1. Generating a sample sales dataset with realistic patterns
    2. Loading and validating the dataset
    3. Asking all predefined questions sequentially
    4. Printing detailed results for each question
    5. Providing a summary of successful vs failed analyses
    
    This function serves as a comprehensive test suite for the data analysis
    agent, showcasing its capabilities across different types of questions.
    """
    print("=" * 70)
    print("Data Analysis Agent - Sample Questions Demo")
    print("=" * 70)
    print()
    
    # Generate sample dataset
    print("Generating sample sales dataset...")
    print("This may take a moment as we create realistic data patterns...")
    df = Generate_Sample_Dataset()
    print(f"Generated dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns)}")
    print()
    
    # Display basic statistics
    print("Dataset Statistics:")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique products: {df['product'].nunique()}")
    print(f"  Unique categories: {df['category'].nunique()}")
    print(f"  Unique regions: {df['region'].nunique()}")
    print(f"  Unique salespeople: {df['salesperson'].nunique()}")
    print(f"  Missing values - Quantity: {df['quantity'].isna().sum()}, Price: {df['price'].isna().sum()}")
    print()
    
    # Save to CSV temporarily
    csv_path = "sample_sales_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path}")
    print()
    
    # Setup analysis system
    print("Setting up analysis system...")
    print("Initializing LLM, analysis config, and execution safety checks...")
    from Main import Setup_Analysis_System
    analysis_graph = Setup_Analysis_System()
    print("System ready!")
    print()
    
    # Load data
    print("Loading data into analysis system...")
    load_result = Load_CSV_Data.invoke({"file_path_or_content": csv_path})
    
    if not load_result.get("success"):
        print(f"Error loading CSV: {load_result.get('error')}")
        return
    
    data_json = load_result.get("dataframe_json")
    schema = load_result.get("schema")
    
    print(f"Data loaded successfully!")
    print(f"  Rows: {load_result.get('row_count', 0)}")
    print(f"  Columns: {load_result.get('column_count', 0)}")
    print(f"  Column types: {', '.join([f'{k}: {v}' for k, v in schema.get('dtypes', {}).items()])}")
    print()
    
    # Run all questions
    print("=" * 70)
    print("Running Analysis Questions")
    print("=" * 70)
    print(f"Processing {len(ANALYSIS_QUESTIONS)} questions...")
    print()
    
    results = []
    start_time = datetime.now()
    
    for i, question in enumerate(ANALYSIS_QUESTIONS, 1):
        print(f"Question {i}/{len(ANALYSIS_QUESTIONS)}: {question}")
        print("-" * 70)
        
        question_start = datetime.now()
        
        try:
            result = analysis_graph.Ask(question=question, data=data_json)
            
            question_duration = (datetime.now() - question_start).total_seconds()
            
            print("Answer:")
            print(result.get("interpretation", "No interpretation available"))
            print()
            
            if result.get("code"):
                print("Generated Code:")
                print("-" * 40)
                print(result["code"])
                print("-" * 40)
                print()
            
            if result.get("execution_result") and result["execution_result"].get("output"):
                output_preview = result["execution_result"]["output"][:200]
                if len(result["execution_result"]["output"]) > 200:
                    output_preview += "..."
                print(f"Execution Output Preview: {output_preview}")
                print()
            
            if result.get("error"):
                print(f"Warning: {result['error']}")
                print()
            
            if result.get("retry_count", 0) > 0:
                print(f"Note: Code was retried {result['retry_count']} time(s)")
                print()
            
            print(f"Processing time: {question_duration:.2f} seconds")
            print()
            
            results.append({
                "question": question,
                "success": not bool(result.get("error")),
                "interpretation": result.get("interpretation"),
                "code": result.get("code"),
                "error": result.get("error"),
                "retry_count": result.get("retry_count", 0),
                "duration": question_duration
            })
        
        except Exception as e:
            question_duration = (datetime.now() - question_start).total_seconds()
            print(f"Error processing question: {str(e)}")
            print(f"Processing time: {question_duration:.2f} seconds")
            print()
            results.append({
                "question": question,
                "success": False,
                "interpretation": None,
                "code": None,
                "error": str(e),
                "retry_count": 0,
                "duration": question_duration
            })
        
        print("=" * 70)
        print()
    
    total_duration = (datetime.now() - start_time).total_seconds()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    successful = sum(1 for r in results if r["success"])
    print(f"Successfully answered: {successful}/{len(results)} questions")
    print(f"Total processing time: {total_duration:.2f} seconds")
    print(f"Average time per question: {total_duration / len(results):.2f} seconds")
    print()
    
    if successful < len(results):
        print("Failed questions:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['question']}")
                if r.get("error"):
                    print(f"    Error: {r['error']}")
                if r.get("retry_count", 0) > 0:
                    print(f"    Retries: {r['retry_count']}")
        print()
    
    # Statistics on retries
    total_retries = sum(r.get("retry_count", 0) for r in results)
    if total_retries > 0:
        print(f"Total code retries across all questions: {total_retries}")
        questions_with_retries = sum(1 for r in results if r.get("retry_count", 0) > 0)
        print(f"Questions requiring retries: {questions_with_retries}")
        print()
    
    # Cleanup
    import os
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
            print(f"Cleaned up temporary file: {csv_path}")
        except:
            pass
    
    print("Demo completed!")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    Run_Samples()
