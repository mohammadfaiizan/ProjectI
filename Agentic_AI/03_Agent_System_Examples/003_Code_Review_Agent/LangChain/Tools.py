"""
Tools module for Code Review Agent.
Contains code analysis tools and data models.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
import ast
import re
import keyword


class Issue(BaseModel):
    """Pydantic model for code review issues."""
    severity: str = Field(description="Severity level: CRITICAL, HIGH, MEDIUM, LOW, INFO")
    category: str = Field(description="Issue category: bug, security, style, performance")
    description: str = Field(description="Detailed description of the issue")
    line_number: Optional[int] = Field(default=None, description="Line number where issue occurs")
    suggestion: Optional[str] = Field(default=None, description="Suggested fix or improvement")


class Code_Metrics:
    """Class for calculating code metrics."""
    
    @staticmethod
    def Calculate_Lines_Of_Code(code: str) -> int:
        """
        Calculate total lines of code (excluding empty lines and comments).
        
        Args:
            code: Source code string
            
        Returns:
            Number of non-empty, non-comment lines
        """
        lines = code.split('\n')
        loc = 0
        in_multiline_comment = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_multiline_comment = not in_multiline_comment
                continue
            
            if in_multiline_comment:
                if '"""' in stripped or "'''" in stripped:
                    in_multiline_comment = False
                continue
            
            if stripped.startswith('#'):
                continue
            
            loc += 1
        
        return loc
    
    @staticmethod
    def Count_Functions(code: str) -> int:
        """
        Count number of function definitions.
        
        Args:
            code: Source code string
            
        Returns:
            Number of function definitions
        """
        try:
            tree = ast.parse(code)
            return len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        except SyntaxError:
            return 0
    
    @staticmethod
    def Count_Classes(code: str) -> int:
        """
        Count number of class definitions.
        
        Args:
            code: Source code string
            
        Returns:
            Number of class definitions
        """
        try:
            tree = ast.parse(code)
            return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        except SyntaxError:
            return 0
    
    @staticmethod
    def Analyze_Imports(code: str) -> Dict[str, List[str]]:
        """
        Analyze import statements in code.
        
        Args:
            code: Source code string
            
        Returns:
            Dictionary with 'standard', 'third_party', and 'local' import lists
        """
        try:
            tree = ast.parse(code)
            imports = {
                'standard': [],
                'third_party': [],
                'local': []
            }
            
            standard_libs = {
                'os', 'sys', 'json', 'datetime', 'collections', 'itertools',
                'functools', 'operator', 're', 'math', 'random', 'string',
                'typing', 'dataclasses', 'enum', 'abc', 'contextlib', 'pathlib'
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name in standard_libs:
                            imports['standard'].append(alias.name)
                        else:
                            imports['third_party'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name in standard_libs:
                            imports['standard'].append(node.module)
                        elif '.' in node.module or node.module.startswith('.'):
                            imports['local'].append(node.module)
                        else:
                            imports['third_party'].append(node.module)
            
            return imports
        except SyntaxError:
            return {'standard': [], 'third_party': [], 'local': []}


@tool
def Parse_Python_Code(code: str) -> Dict[str, any]:
    """
    Parse Python code and extract structure information.
    
    Args:
        code: Python source code string
        
    Returns:
        Dictionary containing parsed information:
        - functions: List of function names
        - classes: List of class names
        - imports: Dictionary of imports by type
        - line_count: Total line count
    """
    try:
        tree = ast.parse(code)
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        metrics = Code_Metrics()
        imports = metrics.Analyze_Imports(code)
        line_count = len(code.split('\n'))
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "line_count": line_count,
            "function_count": len(functions),
            "class_count": len(classes)
        }
    except SyntaxError as e:
        return {
            "functions": [],
            "classes": [],
            "imports": {"standard": [], "third_party": [], "local": []},
            "line_count": len(code.split('\n')),
            "function_count": 0,
            "class_count": 0,
            "parse_error": str(e)
        }


@tool
def Calculate_Complexity(code: str) -> Dict[str, int]:
    """
    Calculate cyclomatic complexity metrics for code.
    
    Args:
        code: Python source code string
        
    Returns:
        Dictionary with complexity metrics:
        - max_complexity: Maximum function complexity
        - avg_complexity: Average function complexity
        - total_complexity: Total complexity score
    """
    try:
        tree = ast.parse(code)
        complexities = []
        
        def Calculate_Node_Complexity(node) -> int:
            """Calculate complexity for a single AST node."""
            complexity = 1
            
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    complexity += 1
                elif isinstance(child, ast.ExceptHandler):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
            
            return complexity
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_complexity = Calculate_Node_Complexity(node)
                complexities.append(func_complexity)
        
        if not complexities:
            return {
                "max_complexity": 0,
                "avg_complexity": 0,
                "total_complexity": 0
            }
        
        return {
            "max_complexity": max(complexities),
            "avg_complexity": sum(complexities) / len(complexities),
            "total_complexity": sum(complexities)
        }
    except SyntaxError:
        return {
            "max_complexity": 0,
            "avg_complexity": 0,
            "total_complexity": 0
        }


@tool
def Check_Common_Patterns(code: str) -> List[Dict[str, any]]:
    """
    Check code for common problematic patterns using regex.
    
    Args:
        code: Python source code string
        
    Returns:
        List of dictionaries containing detected issues
    """
    issues = []
    lines = code.split('\n')
    
    patterns = [
        {
            "pattern": r'\beval\s*\(',
            "severity": "CRITICAL",
            "category": "security",
            "description": "Use of eval() function detected - security risk",
            "suggestion": "Avoid eval(). Use ast.literal_eval() or proper parsing instead."
        },
        {
            "pattern": r'\bexec\s*\(',
            "severity": "CRITICAL",
            "category": "security",
            "description": "Use of exec() function detected - security risk",
            "suggestion": "Avoid exec(). Refactor to use safer alternatives."
        },
        {
            "pattern": r'(password|api_key|secret|token)\s*=\s*["\'][^"\']+["\']',
            "severity": "HIGH",
            "category": "security",
            "description": "Hardcoded credentials detected",
            "suggestion": "Use environment variables or secure configuration management."
        },
        {
            "pattern": r'SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*%s|%\(.*\)s',
            "severity": "HIGH",
            "category": "security",
            "description": "Potential SQL injection vulnerability",
            "suggestion": "Use parameterized queries or ORM methods instead of string formatting."
        },
        {
            "pattern": r'except\s*:',
            "severity": "MEDIUM",
            "category": "bug",
            "description": "Bare except clause detected",
            "suggestion": "Specify exception types: except SpecificException:"
        },
        {
            "pattern": r'except\s+Exception\s*:',
            "severity": "LOW",
            "category": "style",
            "description": "Broad exception handling",
            "suggestion": "Catch specific exceptions when possible."
        },
        {
            "pattern": r'print\s*\(',
            "severity": "LOW",
            "category": "style",
            "description": "Print statement detected (may be debug code)",
            "suggestion": "Use proper logging instead of print statements."
        },
        {
            "pattern": r'if\s+.*==\s+True|if\s+.*==\s+False',
            "severity": "LOW",
            "category": "style",
            "description": "Redundant boolean comparison",
            "suggestion": "Use 'if condition:' or 'if not condition:' instead."
        }
    ]
    
    for line_num, line in enumerate(lines, start=1):
        for pattern_info in patterns:
            if re.search(pattern_info["pattern"], line, re.IGNORECASE):
                issues.append({
                    "severity": pattern_info["severity"],
                    "category": pattern_info["category"],
                    "description": pattern_info["description"],
                    "line_number": line_num,
                    "suggestion": pattern_info["suggestion"],
                    "code_snippet": line.strip()
                })
    
    return issues
