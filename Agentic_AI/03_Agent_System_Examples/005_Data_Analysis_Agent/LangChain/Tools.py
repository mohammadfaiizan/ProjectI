"""
Tools module for Data Analysis Agent.

This module contains LangChain tools and utility classes for data loading,
code execution, chart generation, data profiling, and safe code execution.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# LangChain Tools
# ============================================================================

@tool
def Load_CSV_Data(file_path_or_content: str) -> Dict[str, Any]:
    """
    Load CSV data from a file path or CSV content string into a pandas DataFrame.
    Returns schema information including column names, types, and sample values.
    
    Args:
        file_path_or_content: Either a file path to a CSV file or CSV content as a string
    
    Returns:
        Dictionary containing:
        - dataframe_json: JSON representation of the DataFrame
        - schema: Dictionary with column names, types, and sample values
        - row_count: Number of rows in the dataset
        - column_count: Number of columns in the dataset
    """
    try:
        # Check if it's a file path or content
        if os.path.exists(file_path_or_content):
            df = pd.read_csv(file_path_or_content)
        else:
            # Treat as CSV content string
            from io import StringIO
            df = pd.read_csv(StringIO(file_path_or_content))
        
        # Generate schema information
        schema = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_values": {}
        }
        
        # Get sample values for each column (first non-null value)
        for col in df.columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                sample_val = non_null_values.iloc[0]
                # Convert to string, truncate if too long
                sample_str = str(sample_val)
                if len(sample_str) > 50:
                    sample_str = sample_str[:50] + "..."
                schema["sample_values"][col] = sample_str
            else:
                schema["sample_values"][col] = None
        
        # Convert DataFrame to JSON for serialization
        dataframe_json = df.to_json(orient="records", date_format="iso")
        
        return {
            "dataframe_json": dataframe_json,
            "schema": schema,
            "row_count": len(df),
            "column_count": len(df.columns),
            "success": True
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "dataframe_json": None,
            "schema": None,
            "row_count": 0,
            "column_count": 0
        }


@tool
def Execute_Pandas_Code(code: str, dataframe_json: str) -> Dict[str, Any]:
    """
    Safely execute pandas code on a DataFrame loaded from JSON.
    Returns execution results including output, errors, and result data.
    
    Args:
        code: Python pandas code to execute (should use 'df' as DataFrame variable)
        dataframe_json: JSON string representation of the DataFrame
    
    Returns:
        Dictionary containing:
        - success: Boolean indicating if execution succeeded
        - output: String output from code execution
        - result_data: JSON representation of result (if applicable)
        - error: Error message if execution failed
    """
    try:
        # Load DataFrame from JSON
        df = pd.read_json(dataframe_json, orient="records")
        
        # Create execution namespace with only safe imports
        safe_namespace = {
            "pd": pd,
            "np": np,
            "df": df,
            "DataFrame": pd.DataFrame,
            "Series": pd.Series,
            "__builtins__": {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "sum": sum,
                "max": max,
                "min": min,
                "abs": abs,
                "round": round,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "sorted": sorted,
                "reversed": reversed
            }
        }
        
        # Execute code
        exec_globals = {}
        exec(code, safe_namespace, exec_globals)
        
        # Try to capture result
        result = None
        if "result" in exec_globals:
            result = exec_globals["result"]
        elif "output" in exec_globals:
            result = exec_globals["output"]
        elif "df_result" in exec_globals:
            result = exec_globals["df_result"]
        
        # Convert result to JSON if it's a DataFrame or Series
        result_data = None
        output_str = ""
        
        if result is not None:
            if isinstance(result, (pd.DataFrame, pd.Series)):
                result_data = result.to_json(orient="records", date_format="iso")
                output_str = str(result)
            elif isinstance(result, (dict, list)):
                result_data = json.dumps(result)
                output_str = str(result)
            else:
                output_str = str(result)
                result_data = json.dumps({"value": str(result)})
        
        return {
            "success": True,
            "output": output_str,
            "result_data": result_data,
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "output": None,
            "result_data": None,
            "error": str(e)
        }


@tool
def Generate_Chart_Code(chart_type: str, data_description: str) -> Dict[str, Any]:
    """
    Generate matplotlib code for creating charts based on chart type and data description.
    
    Args:
        chart_type: Type of chart to generate (e.g., 'bar', 'line', 'scatter', 'histogram', 'pie')
        data_description: Description of the data structure and what to plot
    
    Returns:
        Dictionary containing:
        - code: Generated matplotlib code
        - chart_type: The chart type
        - success: Boolean indicating if code generation succeeded
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        prompt = ChatPromptTemplate.from_template("""
Generate matplotlib code to create a {chart_type} chart based on the following data description:

{data_description}

Requirements:
- Use matplotlib.pyplot as plt
- Assume data is in a pandas DataFrame called 'df'
- Include proper labels, title, and formatting
- Save the chart using plt.savefig('chart.png')
- Return ONLY the Python code, no explanations

Code:
""")
        
        chain = prompt | llm
        response = chain.invoke({
            "chart_type": chart_type,
            "data_description": data_description
        })
        
        code = response.content.strip()
        
        # Remove markdown code blocks if present
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()
        
        return {
            "success": True,
            "code": code,
            "chart_type": chart_type,
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "code": None,
            "chart_type": chart_type,
            "error": str(e)
        }


# ============================================================================
# Data Profiler Class
# ============================================================================

class Data_Profiler:
    """
    Utility class for profiling datasets and generating statistics.
    
    Provides methods for generating statistics, detecting data types,
    finding correlations, and identifying outliers.
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize Data Profiler with a DataFrame.
        
        Args:
            dataframe: pandas DataFrame to profile
        """
        self.df = dataframe.copy()
    
    def Generate_Statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistics for the dataset.
        
        Returns:
            Dictionary containing descriptive statistics, missing values,
            and data quality metrics
        """
        stats = {
            "shape": {
                "rows": len(self.df),
                "columns": len(self.df.columns)
            },
            "missing_values": self.df.isnull().sum().to_dict(),
            "missing_percentage": (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "numeric_stats": {},
            "categorical_stats": {}
        }
        
        # Numeric statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["numeric_stats"] = self.df[numeric_cols].describe().to_dict()
        
        # Categorical statistics
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            value_counts = self.df[col].value_counts()
            stats["categorical_stats"][col] = {
                "unique_count": self.df[col].nunique(),
                "top_values": value_counts.head(10).to_dict(),
                "mode": self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else None
            }
        
        return stats
    
    def Detect_Types(self) -> Dict[str, str]:
        """
        Detect and classify column types (numeric, categorical, datetime, etc.).
        
        Returns:
            Dictionary mapping column names to detected types
        """
        type_mapping = {}
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                type_mapping[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                type_mapping[col] = "datetime"
            elif pd.api.types.is_bool_dtype(dtype):
                type_mapping[col] = "boolean"
            else:
                # Check if it might be a date string
                if self.df[col].dtype == "object":
                    sample = self.df[col].dropna().head(10)
                    try:
                        pd.to_datetime(sample)
                        type_mapping[col] = "datetime_string"
                    except:
                        type_mapping[col] = "categorical"
                else:
                    type_mapping[col] = "categorical"
        
        return type_mapping
    
    def Find_Correlations(self, method: str = "pearson") -> Dict[str, Any]:
        """
        Find correlations between numeric columns.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
        
        Returns:
            Dictionary containing correlation matrix and top correlations
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {
                "correlation_matrix": {},
                "top_correlations": [],
                "message": "Need at least 2 numeric columns for correlation analysis"
            }
        
        corr_matrix = self.df[numeric_cols].corr(method=method)
        
        # Find top correlations (excluding diagonal)
        top_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                top_correlations.append({
                    "column1": col1,
                    "column2": col2,
                    "correlation": float(corr_value)
                })
        
        # Sort by absolute correlation value
        top_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "top_correlations": top_correlations[:10],
            "method": method
        }
    
    def Identify_Outliers(self, method: str = "iqr") -> Dict[str, Any]:
        """
        Identify outliers in numeric columns using IQR or Z-score method.
        
        Args:
            method: Method to use ('iqr' or 'zscore')
        
        Returns:
            Dictionary containing outlier information for each numeric column
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            if method == "iqr":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                outlier_indices = self.df[outlier_mask].index.tolist()
                
            elif method == "zscore":
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outlier_mask = z_scores > 3
                outlier_count = outlier_mask.sum()
                outlier_indices = self.df[outlier_mask].index.tolist()
                lower_bound = None
                upper_bound = None
            
            outliers[col] = {
                "count": int(outlier_count),
                "percentage": float(outlier_count / len(self.df) * 100),
                "indices": outlier_indices[:100],  # Limit to first 100 indices
                "lower_bound": float(lower_bound) if lower_bound is not None else None,
                "upper_bound": float(upper_bound) if upper_bound is not None else None
            }
        
        return {
            "method": method,
            "outliers": outliers,
            "total_outlier_rows": len(set([idx for col_outliers in outliers.values() for idx in col_outliers["indices"]]))
        }


# ============================================================================
# Safe Executor Class
# ============================================================================

class Safe_Executor:
    """
    Utility class for safely executing generated code with security checks.
    
    Prevents execution of dangerous operations like file system access,
    subprocess calls, and other potentially harmful code.
    """
    
    def __init__(self, allowed_imports: List[str], blocked_imports: List[str]):
        """
        Initialize Safe Executor with import restrictions.
        
        Args:
            allowed_imports: List of allowed import module names
            blocked_imports: List of blocked import module names
        """
        self.allowed_imports = allowed_imports
        self.blocked_imports = blocked_imports
    
    def Check_Code_Safety(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if code contains any unsafe operations.
        
        Args:
            code: Python code string to check
        
        Returns:
            Tuple of (is_safe, error_message)
        """
        code_lower = code.lower()
        
        # Check for blocked imports
        for blocked in self.blocked_imports:
            if f"import {blocked}" in code_lower or f"from {blocked}" in code_lower:
                return False, f"Blocked import detected: {blocked}"
        
        # Check for dangerous built-in functions
        dangerous_functions = ["eval", "exec", "__import__", "open", "compile"]
        for func in dangerous_functions:
            if f"{func}(" in code_lower:
                return False, f"Dangerous function detected: {func}"
        
        # Check for file operations (basic check)
        dangerous_patterns = [
            "os.system",
            "os.popen",
            "subprocess",
            "shutil",
            "pickle.load",
            "__builtins__"
        ]
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False, f"Dangerous pattern detected: {pattern}"
        
        return True, None
    
    def Execute_With_Timeout(self, code: str, namespace: Dict[str, Any], timeout: int) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute code with a timeout limit.
        
        Args:
            code: Python code to execute
            namespace: Namespace dictionary for code execution
            timeout: Timeout in seconds
        
        Returns:
            Tuple of (success, result, error_message)
        """
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Code execution exceeded {timeout} seconds")
        
        # Set up timeout (Unix only)
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        
        try:
            # Check code safety first
            is_safe, error_msg = self.Check_Code_Safety(code)
            if not is_safe:
                return False, None, error_msg
            
            # Execute code
            exec_globals = {}
            exec(code, namespace, exec_globals)
            
            # Try to get result
            result = exec_globals.get("result") or exec_globals.get("output")
            
            return True, result, None
        
        except TimeoutError as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, str(e)
        finally:
            # Cancel timeout
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
