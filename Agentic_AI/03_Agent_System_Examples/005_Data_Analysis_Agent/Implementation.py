"""
Data Analysis Agent - Complete Implementation

An intelligent agent that analyzes CSV/tabular data, generates insights,
and creates visualizations using natural language queries.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import warnings
from io import StringIO
import sys
from contextlib import redirect_stdout, redirect_stderr

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not available. Install with: pip install openai")

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


class Data_Loader:
    """Loads and inspects CSV/tabular data files."""
    
    def __init__(self):
        self.dataframe = None
        self.metadata = {}
    
    def load_csv(self, file_path: str, encoding: str = None, delimiter: str = None) -> pd.DataFrame:
        """
        Load CSV file with automatic encoding detection.
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding (auto-detected if None)
            delimiter: CSV delimiter (auto-detected if None)
        
        Returns:
            Loaded DataFrame
        """
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252'] if encoding is None else [encoding]
        delimiters = [',', '\t', ';', '|'] if delimiter is None else [delimiter]
        
        for enc in encodings:
            for delim in delimiters:
                try:
                    df = pd.read_csv(file_path, encoding=enc, delimiter=delim, low_memory=False)
                    self.dataframe = df
                    self._extract_metadata(file_path)
                    return df
                except Exception as e:
                    continue
        
        raise ValueError(f"Could not load file {file_path} with any encoding/delimiter combination")
    
    def _extract_metadata(self, file_path: str):
        """Extract basic metadata about the loaded file."""
        if self.dataframe is None:
            return
        
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        self.metadata = {
            'file_path': file_path,
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'num_rows': len(self.dataframe),
            'num_columns': len(self.dataframe.columns),
            'column_names': list(self.dataframe.columns),
            'load_timestamp': datetime.now().isoformat()
        }
    
    def inspect_schema(self) -> Dict[str, Any]:
        """Get basic schema information."""
        if self.dataframe is None:
            return {}
        
        return {
            'columns': list(self.dataframe.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.dataframe.dtypes.items()},
            'shape': self.dataframe.shape,
            'memory_usage_mb': round(self.dataframe.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }
    
    def get_sample_rows(self, n: int = 5) -> pd.DataFrame:
        """Get sample rows from the dataset."""
        if self.dataframe is None:
            return pd.DataFrame()
        return self.dataframe.head(n)


class Schema_Analyzer:
    """Analyzes dataset schema, statistics, and relationships."""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.schema_info = {}
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive schema analysis."""
        self.schema_info = {
            'column_types': self._analyze_column_types(),
            'statistics': self._calculate_statistics(),
            'missing_values': self._analyze_missing_values(),
            'correlations': self._calculate_correlations(),
            'categorical_info': self._analyze_categorical(),
            'data_quality': self._assess_data_quality()
        }
        return self.schema_info
    
    def _analyze_column_types(self) -> Dict[str, str]:
        """Categorize columns by data type."""
        type_mapping = {}
        for col in self.dataframe.columns:
            dtype = self.dataframe[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                type_mapping[col] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                type_mapping[col] = 'datetime'
            elif pd.api.types.is_bool_dtype(dtype):
                type_mapping[col] = 'boolean'
            else:
                type_mapping[col] = 'categorical'
        return type_mapping
    
    def _calculate_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Calculate descriptive statistics for numeric columns."""
        stats = {}
        numeric_cols = self.dataframe.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats[col] = {
                'mean': float(self.dataframe[col].mean()),
                'median': float(self.dataframe[col].median()),
                'std': float(self.dataframe[col].std()),
                'min': float(self.dataframe[col].min()),
                'max': float(self.dataframe[col].max()),
                'q25': float(self.dataframe[col].quantile(0.25)),
                'q75': float(self.dataframe[col].quantile(0.75)),
                'skewness': float(self.dataframe[col].skew()),
                'kurtosis': float(self.dataframe[col].kurtosis())
            }
        
        return stats
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values in the dataset."""
        missing = self.dataframe.isnull().sum()
        missing_pct = (missing / len(self.dataframe)) * 100
        
        return {
            'total_missing': int(missing.sum()),
            'missing_per_column': {col: int(count) for col, count in missing.items()},
            'missing_percentage_per_column': {col: float(pct) for col, pct in missing_pct.items()},
            'columns_with_missing': [col for col in missing.index if missing[col] > 0]
        }
    
    def _calculate_correlations(self) -> Dict[str, float]:
        """Calculate correlations between numeric columns."""
        numeric_df = self.dataframe.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return {}
        
        corr_matrix = numeric_df.corr()
        correlations = {}
        
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                key = f"{col1}_vs_{col2}"
                correlations[key] = float(corr_matrix.loc[col1, col2])
        
        return correlations
    
    def _analyze_categorical(self) -> Dict[str, Dict[str, Any]]:
        """Analyze categorical columns."""
        cat_info = {}
        categorical_cols = self.dataframe.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            value_counts = self.dataframe[col].value_counts()
            cat_info[col] = {
                'unique_count': int(self.dataframe[col].nunique()),
                'top_values': value_counts.head(10).to_dict(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None
            }
        
        return cat_info
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess overall data quality."""
        total_cells = self.dataframe.shape[0] * self.dataframe.shape[1]
        missing_cells = self.dataframe.isnull().sum().sum()
        
        return {
            'completeness_score': float(1 - (missing_cells / total_cells)),
            'total_cells': int(total_cells),
            'missing_cells': int(missing_cells),
            'duplicate_rows': int(self.dataframe.duplicated().sum())
        }


class Query_Generator:
    """Converts natural language queries to pandas code using LLM."""
    
    def __init__(self, api_key: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4"
    
    def generate_code(self, query: str, schema_info: Dict[str, Any], 
                     sample_data: pd.DataFrame) -> str:
        """
        Generate pandas code from natural language query.
        
        Args:
            query: Natural language question
            schema_info: Schema analysis results
            sample_data: Sample rows for context
        
        Returns:
            Generated pandas code string
        """
        prompt = self._build_prompt(query, schema_info, sample_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            code = response.choices[0].message.content.strip()
            
            if code.startswith("```python"):
                code = code.replace("```python", "").replace("```", "").strip()
            elif code.startswith("```"):
                code = code.replace("```", "").strip()
            
            return code
        except Exception as e:
            raise Exception(f"Failed to generate code: {str(e)}")
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for code generation."""
        return """You are an expert pandas programmer. Generate clean, efficient pandas code 
to answer data analysis questions. Only return the code, no explanations. Use the variable 
name 'df' for the DataFrame. Ensure the code is safe and only uses pandas operations."""
    
    def _build_prompt(self, query: str, schema_info: Dict[str, Any], 
                     sample_data: pd.DataFrame) -> str:
        """Build the prompt for code generation."""
        columns_info = "\n".join([f"- {col}: {dtype}" 
                                  for col, dtype in schema_info.get('column_types', {}).items()])
        
        sample_str = sample_data.head(3).to_string()
        
        prompt = f"""Given the following dataset schema and sample data, write pandas code to answer this question:

Question: {query}

Schema:
{columns_info}

Sample Data (first 3 rows):
{sample_str}

Generate pandas code that answers the question. Use variable name 'df' for the DataFrame.
Return only the code, no markdown formatting."""
        
        return prompt


class Code_Executor:
    """Safely executes generated pandas code."""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.execution_context = {}
    
    def execute(self, code: str) -> Union[pd.DataFrame, pd.Series, Any]:
        """
        Execute pandas code safely.
        
        Args:
            code: Python code string to execute
        
        Returns:
            Execution result (DataFrame, Series, or scalar)
        """
        safe_globals = {
            'pd': pd,
            'np': np,
            'df': self.dataframe.copy(),
            '__builtins__': {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sorted': sorted,
                'reversed': reversed
            }
        }
        
        safe_locals = {}
        
        try:
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals, safe_locals)
            
            result = safe_locals.get('result')
            if result is None:
                result = safe_globals.get('df')
            
            if result is None:
                raise ValueError("Code did not produce a result. Ensure you assign the result to 'result' variable.")
            
            return result
        except Exception as e:
            raise Exception(f"Code execution failed: {str(e)}")


class Visualization_Generator:
    """Generates visualizations from data."""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_chart(self, data: Union[pd.DataFrame, pd.Series], 
                      chart_type: str = "auto", title: str = None,
                      x_label: str = None, y_label: str = None) -> str:
        """
        Generate a visualization.
        
        Args:
            data: Data to visualize
            chart_type: Type of chart (auto, histogram, scatter, bar, line, heatmap)
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
        
        Returns:
            Path to saved visualization file
        """
        if chart_type == "auto":
            chart_type = self._detect_chart_type(data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        plt.figure(figsize=(10, 6))
        
        if chart_type == "histogram" and isinstance(data, pd.Series):
            data.hist(bins=30, edgecolor='black')
            plt.xlabel(x_label or data.name or "Value")
            plt.ylabel(y_label or "Frequency")
        
        elif chart_type == "bar" and isinstance(data, pd.Series):
            data.plot(kind='bar')
            plt.xlabel(x_label or data.index.name or "Category")
            plt.ylabel(y_label or "Value")
            plt.xticks(rotation=45, ha='right')
        
        elif chart_type == "scatter" and isinstance(data, pd.DataFrame):
            if len(data.columns) >= 2:
                plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
                plt.xlabel(x_label or data.columns[0])
                plt.ylabel(y_label or data.columns[1])
        
        elif chart_type == "line" and isinstance(data, pd.Series):
            data.plot(kind='line')
            plt.xlabel(x_label or data.index.name or "Index")
            plt.ylabel(y_label or data.name or "Value")
        
        elif chart_type == "heatmap" and isinstance(data, pd.DataFrame):
            sns.heatmap(data.select_dtypes(include=[np.number]), annot=True, fmt='.2f', cmap='coolwarm')
        
        else:
            data.plot()
        
        plt.title(title or "Data Visualization")
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _detect_chart_type(self, data: Union[pd.DataFrame, pd.Series]) -> str:
        """Automatically detect appropriate chart type."""
        if isinstance(data, pd.Series):
            if pd.api.types.is_numeric_dtype(data):
                if len(data.unique()) > 20:
                    return "histogram"
                else:
                    return "bar"
            else:
                return "bar"
        elif isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                return "scatter"
            elif len(numeric_cols) == 1:
                return "histogram"
            else:
                return "bar"
        return "line"


class Data_Analysis_Agent:
    """Main agent orchestrating all components."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.data_loader = Data_Loader()
        self.schema_analyzer = None
        self.query_generator = Query_Generator(api_key)
        self.code_executor = None
        self.visualization_generator = Visualization_Generator()
        self.dataframe = None
        self.schema_info = {}
    
    def load_data(self, file_path: str) -> Dict[str, Any]:
        """
        Load and profile a dataset.
        
        Args:
            file_path: Path to CSV file
        
        Returns:
            Dictionary with load results and schema information
        """
        print(f"Loading data from {file_path}...")
        self.dataframe = self.data_loader.load_csv(file_path)
        
        print("Analyzing schema...")
        self.schema_analyzer = Schema_Analyzer(self.dataframe)
        self.schema_info = self.schema_analyzer.analyze()
        
        self.code_executor = Code_Executor(self.dataframe)
        
        return {
            'metadata': self.data_loader.metadata,
            'schema': self.schema_info,
            'sample': self.data_loader.get_sample_rows(5).to_dict('records')
        }
    
    def ask_question(self, question: str, generate_visualization: bool = False) -> Dict[str, Any]:
        """
        Answer a natural language question about the data.
        
        Args:
            question: Natural language question
            generate_visualization: Whether to create a visualization
        
        Returns:
            Dictionary with answer, code, and optional visualization
        """
        if self.dataframe is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print(f"Processing question: {question}")
        
        sample_data = self.data_loader.get_sample_rows(3)
        code = self.query_generator.generate_code(question, self.schema_info, sample_data)
        
        print(f"Generated code:\n{code}\n")
        
        result = self.code_executor.execute(code)
        
        answer = {
            'question': question,
            'code': code,
            'result': self._format_result(result)
        }
        
        if generate_visualization and isinstance(result, (pd.DataFrame, pd.Series)):
            try:
                chart_path = self.visualization_generator.generate_chart(result, title=question)
                answer['visualization'] = chart_path
            except Exception as e:
                answer['visualization_error'] = str(e)
        
        return answer
    
    def generate_insights(self) -> List[Dict[str, Any]]:
        """
        Automatically generate insights about the data.
        
        Returns:
            List of insight dictionaries
        """
        if self.dataframe is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        insights = []
        
        missing_info = self.schema_info.get('missing_values', {})
        if missing_info.get('total_missing', 0) > 0:
            insights.append({
                'type': 'data_quality',
                'title': 'Missing Values Detected',
                'description': f"Found {missing_info['total_missing']} missing values across the dataset.",
                'details': missing_info['missing_percentage_per_column']
            })
        
        correlations = self.schema_info.get('correlations', {})
        strong_correlations = {k: v for k, v in correlations.items() if abs(v) > 0.7}
        if strong_correlations:
            insights.append({
                'type': 'correlation',
                'title': 'Strong Correlations Found',
                'description': f"Found {len(strong_correlations)} pairs of strongly correlated variables.",
                'details': strong_correlations
            })
        
        stats = self.schema_info.get('statistics', {})
        for col, col_stats in stats.items():
            if abs(col_stats.get('skewness', 0)) > 2:
                insights.append({
                    'type': 'distribution',
                    'title': f'Skewed Distribution: {col}',
                    'description': f"Column '{col}' has significant skewness ({col_stats['skewness']:.2f}).",
                    'details': col_stats
                })
        
        return insights
    
    def create_report(self, output_file: str = None) -> Dict[str, Any]:
        """
        Create a comprehensive analysis report.
        
        Args:
            output_file: Optional file path to save report as JSON
        
        Returns:
            Complete report dictionary
        """
        if self.dataframe is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'metadata': self.data_loader.metadata,
            'schema_analysis': self.schema_info,
            'insights': self.generate_insights(),
            'summary': {
                'total_rows': len(self.dataframe),
                'total_columns': len(self.dataframe.columns),
                'numeric_columns': len([c for c, t in self.schema_info.get('column_types', {}).items() 
                                       if t == 'numeric']),
                'categorical_columns': len([c for c, t in self.schema_info.get('column_types', {}).items() 
                                           if t == 'categorical']),
                'data_quality_score': self.schema_info.get('data_quality', {}).get('completeness_score', 0)
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to {output_file}")
        
        return report
    
    def _format_result(self, result: Any) -> Any:
        """Format execution result for display."""
        if isinstance(result, pd.DataFrame):
            return {
                'type': 'DataFrame',
                'shape': result.shape,
                'columns': list(result.columns),
                'data': result.head(10).to_dict('records')
            }
        elif isinstance(result, pd.Series):
            return {
                'type': 'Series',
                'length': len(result),
                'data': result.head(20).to_dict()
            }
        elif isinstance(result, (int, float, str, bool)):
            return {
                'type': type(result).__name__,
                'value': result
            }
        else:
            return str(result)


def main():
    """Main function demonstrating the Data Analysis Agent."""
    
    print("=" * 60)
    print("Data Analysis Agent - Demonstration")
    print("=" * 60)
    
    if not OPENAI_AVAILABLE:
        print("Error: OpenAI library not available.")
        print("Please install with: pip install openai")
        print("And set OPENAI_API_KEY environment variable.")
        return
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Using mock mode.")
        print("Set your API key: export OPENAI_API_KEY='your-key'")
        return
    
    agent = Data_Analysis_Agent(api_key=api_key)
    
    print("\n1. Generating sample dataset...")
    np.random.seed(42)
    sample_data = {
        'customer_id': range(1, 1001),
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 15000, 1000).astype(int),
        'purchase_amount': np.random.exponential(100, 1000).astype(int),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 1000),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'satisfaction_score': np.random.randint(1, 6, 1000)
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_file = "sample_customer_data.csv"
    sample_df.to_csv(sample_file, index=False)
    print(f"Created sample dataset: {sample_file}")
    
    print("\n2. Loading and profiling data...")
    load_results = agent.load_data(sample_file)
    print(f"Loaded {load_results['metadata']['num_rows']} rows, "
          f"{load_results['metadata']['num_columns']} columns")
    
    print("\n3. Asking questions about the data...")
    questions = [
        "What is the average age of customers?",
        "What is the total purchase amount by product category?",
        "Show me the top 5 customers by purchase amount",
        "What is the correlation between age and income?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        try:
            answer = agent.ask_question(question, generate_visualization=True)
            print(f"Answer: {answer['result']}")
            if 'visualization' in answer:
                print(f"Visualization saved: {answer['visualization']}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\n4. Generating automatic insights...")
    insights = agent.generate_insights()
    for insight in insights:
        print(f"\n- {insight['title']}: {insight['description']}")
    
    print("\n5. Creating comprehensive report...")
    report = agent.create_report("analysis_report.json")
    print(f"Report created with {len(report['insights'])} insights")
    print(f"Data quality score: {report['summary']['data_quality_score']:.2%}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
