"""
Configuration module for Data Analysis Agent.

This module contains all configuration classes and settings for the LangChain-based
data analysis system, including LLM configuration, analysis settings, and execution
safety parameters.
"""

import os
from typing import Optional, List
from langchain_openai import ChatOpenAI


class LLM_Config:
    """
    Configuration class for Language Model setup.
    
    Handles initialization of ChatOpenAI with configurable parameters
    for model selection, temperature, and API key management.
    Temperature is set to 0 for deterministic code generation.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize LLM configuration.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Sampling temperature (0.0 for deterministic code generation)
            api_key: OpenAI API key (if None, reads from environment)
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    def Get_LLM(self) -> ChatOpenAI:
        """
        Create and return a configured ChatOpenAI instance.
        
        Returns:
            Configured ChatOpenAI instance with temperature=0 for code generation
        """
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            max_tokens=self.max_tokens
        )


class Analysis_Config:
    """
    Configuration class for Data Analysis settings.
    
    Contains settings for data display limits, chart output directories,
    and supported file types for data loading.
    """
    
    def __init__(
        self,
        max_rows_to_display: int = 100,
        chart_output_directory: str = "./charts",
        supported_file_types: Optional[List[str]] = None
    ):
        """
        Initialize analysis configuration.
        
        Args:
            max_rows_to_display: Maximum number of rows to display in results
            chart_output_directory: Directory path for saving generated charts
            supported_file_types: List of supported file extensions (default: ['.csv'])
        """
        self.max_rows_to_display = max_rows_to_display
        self.chart_output_directory = chart_output_directory
        self.supported_file_types = supported_file_types or [".csv"]
        
        # Create chart output directory if it doesn't exist
        os.makedirs(self.chart_output_directory, exist_ok=True)
    
    def Get_Max_Rows_To_Display(self) -> int:
        """Return maximum rows to display in analysis results."""
        return self.max_rows_to_display
    
    def Get_Chart_Output_Directory(self) -> str:
        """Return chart output directory path."""
        return self.chart_output_directory
    
    def Get_Supported_File_Types(self) -> List[str]:
        """Return list of supported file extensions."""
        return self.supported_file_types
    
    def Is_File_Type_Supported(self, file_path: str) -> bool:
        """
        Check if a file type is supported.
        
        Args:
            file_path: Path to the file to check
        
        Returns:
            True if file type is supported, False otherwise
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in self.supported_file_types


class Execution_Config:
    """
    Configuration class for Code Execution settings.
    
    Handles safety parameters for executing generated pandas code,
    including timeout limits, retry counts, and allowed imports.
    """
    
    def __init__(
        self,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        allowed_imports: Optional[List[str]] = None
    ):
        """
        Initialize execution configuration.
        
        Args:
            timeout_seconds: Maximum execution time in seconds
            max_retries: Maximum number of retry attempts for failed executions
            allowed_imports: List of allowed import modules (default: pandas, numpy, matplotlib)
        """
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # Default allowed imports for data analysis
        if allowed_imports is None:
            self.allowed_imports = [
                "pandas",
                "numpy",
                "matplotlib",
                "matplotlib.pyplot",
                "seaborn",
                "datetime",
                "json"
            ]
        else:
            self.allowed_imports = allowed_imports
        
        # Blocked imports for security
        self.blocked_imports = [
            "os",
            "subprocess",
            "sys",
            "shutil",
            "pickle",
            "eval",
            "exec",
            "open",
            "__import__"
        ]
    
    def Get_Timeout_Seconds(self) -> int:
        """Return timeout in seconds for code execution."""
        return self.timeout_seconds
    
    def Get_Max_Retries(self) -> int:
        """Return maximum number of retry attempts."""
        return self.max_retries
    
    def Get_Allowed_Imports(self) -> List[str]:
        """Return list of allowed import modules."""
        return self.allowed_imports
    
    def Get_Blocked_Imports(self) -> List[str]:
        """Return list of blocked import modules."""
        return self.blocked_imports
    
    def Is_Import_Allowed(self, import_name: str) -> bool:
        """
        Check if an import is allowed.
        
        Args:
            import_name: Name of the import module to check
        
        Returns:
            True if import is allowed, False otherwise
        """
        # Check if explicitly blocked
        if import_name in self.blocked_imports:
            return False
        
        # Check if explicitly allowed
        if import_name in self.allowed_imports:
            return True
        
        # Check if it's a submodule of an allowed import
        for allowed in self.allowed_imports:
            if import_name.startswith(allowed + "."):
                return True
        
        return False
