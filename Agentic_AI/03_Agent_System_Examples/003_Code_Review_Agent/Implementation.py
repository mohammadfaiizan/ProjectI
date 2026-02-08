"""
Code Review Agent Implementation

A comprehensive code review system that analyzes Python code for bugs,
security issues, style problems, and suggests improvements using OpenAI.
"""

import ast
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from openai import OpenAI


class Severity_Level(Enum):
    """Severity levels for code review issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Issue_Category(Enum):
    """Categories of code review issues."""
    BUG = "bug"
    SECURITY = "security"
    STYLE = "style"
    IMPROVEMENT = "improvement"


@dataclass
class Review_Issue:
    """Represents a single code review issue."""
    category: Issue_Category
    severity: Severity_Level
    line_number: Optional[int]
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class Review_Report:
    """Structured report containing all review findings."""
    file_path: str
    total_issues: int
    issues_by_category: Dict[str, int] = field(default_factory=dict)
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    issues: List[Review_Issue] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "file_path": self.file_path,
            "total_issues": self.total_issues,
            "issues_by_category": self.issues_by_category,
            "issues_by_severity": self.issues_by_severity,
            "issues": [
                {
                    "category": issue.category.value,
                    "severity": issue.severity.value,
                    "line_number": issue.line_number,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                    "code_snippet": issue.code_snippet,
                }
                for issue in self.issues
            ],
            "summary": self.summary,
        }

    def to_markdown(self) -> str:
        """Convert report to markdown format."""
        lines = [
            f"# Code Review Report: {self.file_path}",
            "",
            f"**Total Issues:** {self.total_issues}",
            "",
            "## Summary by Category",
            "",
        ]

        for category, count in self.issues_by_category.items():
            lines.append(f"- **{category.title()}**: {count}")

        lines.extend(["", "## Summary by Severity", ""])

        for severity, count in self.issues_by_severity.items():
            lines.append(f"- **{severity.title()}**: {count}")

        if self.summary:
            lines.extend(["", "## Overall Summary", "", self.summary, ""])

        lines.extend(["", "## Detailed Issues", ""])

        for issue in sorted(
            self.issues,
            key=lambda x: (
                list(Severity_Level).index(x.severity),
                x.line_number or 0,
            ),
        ):
            lines.append(f"### {issue.category.value.title()} - {issue.severity.value.title()}")
            if issue.line_number:
                lines.append(f"**Line {issue.line_number}**")
            lines.append(f"**Message:** {issue.message}")
            if issue.suggestion:
                lines.append(f"**Suggestion:** {issue.suggestion}")
            if issue.code_snippet:
                lines.append(f"**Code:**")
                lines.append("```python")
                lines.append(issue.code_snippet)
                lines.append("```")
            lines.append("")

        return "\n".join(lines)


class Code_Parser:
    """Parses Python code and extracts structural information."""

    def __init__(self, code: str):
        """Initialize parser with code string."""
        self.code = code
        self.ast_tree: Optional[ast.AST] = None
        self.functions: List[Dict[str, Any]] = []
        self.classes: List[Dict[str, Any]] = []
        self.imports: List[str] = []

    def parse(self) -> bool:
        """Parse the code and extract structure."""
        try:
            self.ast_tree = ast.parse(self.code)
            self._extract_functions()
            self._extract_classes()
            self._extract_imports()
            return True
        except SyntaxError as e:
            return False

    def _extract_functions(self):
        """Extract function definitions from AST."""
        self.functions = []
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node),
                    "code": ast.get_source_segment(self.code, node),
                }
                self.functions.append(func_info)

    def _extract_classes(self):
        """Extract class definitions from AST."""
        self.classes = []
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.ClassDef):
                methods = [
                    n.name
                    for n in node.body
                    if isinstance(n, ast.FunctionDef)
                ]
                class_info = {
                    "name": node.name,
                    "line": node.lineno,
                    "methods": methods,
                    "docstring": ast.get_docstring(node),
                    "code": ast.get_source_segment(self.code, node),
                }
                self.classes.append(class_info)

    def _extract_imports(self):
        """Extract import statements from AST."""
        self.imports = []
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    self.imports.append(f"{module}.{alias.name}")


class Bug_Detector:
    """Detects potential bugs in code using LLM analysis."""

    def __init__(self, client: OpenAI, model: str = "gpt-4"):
        """Initialize bug detector with OpenAI client."""
        self.client = client
        self.model = model

    def detect_bugs(
        self, code: str, functions: List[Dict], classes: List[Dict]
    ) -> List[Review_Issue]:
        """Detect bugs in the provided code."""
        issues = []

        prompt = f"""Analyze the following Python code for potential bugs, logic errors, edge cases, and runtime issues.

Code:
```python
{code}
```

Functions:
{json.dumps(functions, indent=2)}

Classes:
{json.dumps(classes, indent=2)}

Identify:
1. Logic errors and incorrect implementations
2. Edge cases and boundary conditions not handled
3. Type inconsistencies and potential runtime errors
4. Resource leaks (files, connections)
5. Incorrect error handling
6. Off-by-one errors and indexing issues

For each issue found, provide:
- Line number (if applicable)
- Severity (critical, high, medium, low)
- Clear description
- Suggestion for fix
- Relevant code snippet

Respond in JSON format:
{{
  "issues": [
    {{
      "line_number": <int or null>,
      "severity": "<critical|high|medium|low>",
      "message": "<description>",
      "suggestion": "<fix suggestion>",
      "code_snippet": "<relevant code>"
    }}
  ]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            for item in result.get("issues", []):
                severity_map = {
                    "critical": Severity_Level.CRITICAL,
                    "high": Severity_Level.HIGH,
                    "medium": Severity_Level.MEDIUM,
                    "low": Severity_Level.LOW,
                }
                issues.append(
                    Review_Issue(
                        category=Issue_Category.BUG,
                        severity=severity_map.get(
                            item.get("severity", "medium").lower(),
                            Severity_Level.MEDIUM,
                        ),
                        line_number=item.get("line_number"),
                        message=item.get("message", ""),
                        suggestion=item.get("suggestion"),
                        code_snippet=item.get("code_snippet"),
                    )
                )
        except Exception as e:
            issues.append(
                Review_Issue(
                    category=Issue_Category.BUG,
                    severity=Severity_Level.LOW,
                    line_number=None,
                    message=f"Error during bug detection: {str(e)}",
                )
            )

        return issues


class Security_Analyzer:
    """Analyzes code for security vulnerabilities."""

    def __init__(self, client: OpenAI, model: str = "gpt-4"):
        """Initialize security analyzer with OpenAI client."""
        self.client = client
        self.model = model

    def analyze_security(
        self, code: str, functions: List[Dict], classes: List[Dict]
    ) -> List[Review_Issue]:
        """Analyze code for security vulnerabilities."""
        issues = []

        # Pattern-based checks
        issues.extend(self._check_hardcoded_secrets(code))
        issues.extend(self._check_sql_injection(code))
        issues.extend(self._check_insecure_random(code))

        # LLM-based analysis
        issues.extend(self._llm_security_analysis(code, functions, classes))

        return issues

    def _check_hardcoded_secrets(self, code: str) -> List[Review_Issue]:
        """Check for hardcoded secrets and credentials."""
        issues = []
        secret_patterns = [
            ("password", "="),
            ("api_key", "="),
            ("secret", "="),
            ("token", "="),
            ("credential", "="),
        ]

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            for pattern, operator in secret_patterns:
                if pattern in line_lower and operator in line:
                    if '"' in line or "'" in line:
                        issues.append(
                            Review_Issue(
                                category=Issue_Category.SECURITY,
                                severity=Severity_Level.CRITICAL,
                                line_number=i,
                                message=f"Potential hardcoded {pattern} detected",
                                suggestion="Use environment variables or secure secret management",
                                code_snippet=line.strip(),
                            )
                        )

        return issues

    def _check_sql_injection(self, code: str) -> List[Review_Issue]:
        """Check for SQL injection vulnerabilities."""
        issues = []
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if "execute" in line.lower() or "query" in line.lower():
                if "%" in line or "+" in line or "format" in line.lower():
                    if "sql" in line.lower() or "db" in line.lower():
                        issues.append(
                            Review_Issue(
                                category=Issue_Category.SECURITY,
                                severity=Severity_Level.HIGH,
                                line_number=i,
                                message="Potential SQL injection vulnerability",
                                suggestion="Use parameterized queries or ORM methods",
                                code_snippet=line.strip(),
                            )
                        )

        return issues

    def _check_insecure_random(self, code: str) -> List[Review_Issue]:
        """Check for insecure random number generation."""
        issues = []
        if "random.random()" in code or "random.randint" in code:
            if "secrets" not in code.lower():
                issues.append(
                    Review_Issue(
                        category=Issue_Category.SECURITY,
                        severity=Severity_Level.MEDIUM,
                        line_number=None,
                        message="Using random module for security-sensitive operations",
                        suggestion="Use secrets module for cryptographically secure random numbers",
                    )
                )

        return issues

    def _llm_security_analysis(
        self, code: str, functions: List[Dict], classes: List[Dict]
    ) -> List[Review_Issue]:
        """Use LLM to analyze security vulnerabilities."""
        issues = []

        prompt = f"""Analyze the following Python code for security vulnerabilities.

Code:
```python
{code}
```

Functions:
{json.dumps(functions, indent=2)}

Classes:
{json.dumps(classes, indent=2)}

Check for:
1. SQL injection vulnerabilities
2. Cross-site scripting (XSS) vulnerabilities
3. Authentication and authorization flaws
4. Insecure deserialization
5. Path traversal vulnerabilities
6. Missing input validation
7. Insecure cryptographic operations
8. Information disclosure

For each vulnerability found, provide:
- Line number (if applicable)
- Severity (critical, high, medium, low)
- Clear description
- Suggestion for fix
- Relevant code snippet

Respond in JSON format:
{{
  "issues": [
    {{
      "line_number": <int or null>,
      "severity": "<critical|high|medium|low>",
      "message": "<description>",
      "suggestion": "<fix suggestion>",
      "code_snippet": "<relevant code>"
    }}
  ]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            for item in result.get("issues", []):
                severity_map = {
                    "critical": Severity_Level.CRITICAL,
                    "high": Severity_Level.HIGH,
                    "medium": Severity_Level.MEDIUM,
                    "low": Severity_Level.LOW,
                }
                issues.append(
                    Review_Issue(
                        category=Issue_Category.SECURITY,
                        severity=severity_map.get(
                            item.get("severity", "medium").lower(),
                            Severity_Level.MEDIUM,
                        ),
                        line_number=item.get("line_number"),
                        message=item.get("message", ""),
                        suggestion=item.get("suggestion"),
                        code_snippet=item.get("code_snippet"),
                    )
                )
        except Exception as e:
            pass

        return issues


class Style_Checker:
    """Checks code style and PEP 8 compliance."""

    def __init__(self, client: OpenAI, model: str = "gpt-4"):
        """Initialize style checker with OpenAI client."""
        self.client = client
        self.model = model

    def check_style(
        self, code: str, functions: List[Dict], classes: List[Dict]
    ) -> List[Review_Issue]:
        """Check code style and PEP 8 compliance."""
        issues = []

        # Basic checks
        issues.extend(self._check_line_length(code))
        issues.extend(self._check_naming_conventions(functions, classes))
        issues.extend(self._check_docstrings(functions, classes))

        # LLM-based style analysis
        issues.extend(self._llm_style_analysis(code, functions, classes))

        return issues

    def _check_line_length(self, code: str) -> List[Review_Issue]:
        """Check for lines exceeding PEP 8 limit."""
        issues = []
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                issues.append(
                    Review_Issue(
                        category=Issue_Category.STYLE,
                        severity=Severity_Level.LOW,
                        line_number=i,
                        message=f"Line exceeds 100 characters ({len(line)} chars)",
                        suggestion="Break long lines or refactor",
                        code_snippet=line.strip(),
                    )
                )

        return issues

    def _check_naming_conventions(
        self, functions: List[Dict], classes: List[Dict]
    ) -> List[Review_Issue]:
        """Check naming conventions."""
        issues = []

        for func in functions:
            name = func["name"]
            if not name.islower() and "_" not in name:
                if not name.startswith("_"):
                    issues.append(
                        Review_Issue(
                            category=Issue_Category.STYLE,
                            severity=Severity_Level.LOW,
                            line_number=func["line"],
                            message=f"Function '{name}' should use snake_case",
                            suggestion=f"Rename to {name.lower()}",
                        )
                    )

        return issues

    def _check_docstrings(
        self, functions: List[Dict], classes: List[Dict]
    ) -> List[Review_Issue]:
        """Check for missing docstrings."""
        issues = []

        for func in functions:
            if not func.get("docstring"):
                issues.append(
                    Review_Issue(
                        category=Issue_Category.STYLE,
                        severity=Severity_Level.LOW,
                        line_number=func["line"],
                        message=f"Function '{func['name']}' missing docstring",
                        suggestion="Add a docstring describing the function's purpose",
                    )
                )

        for cls in classes:
            if not cls.get("docstring"):
                issues.append(
                    Review_Issue(
                        category=Issue_Category.STYLE,
                        severity=Severity_Level.LOW,
                        line_number=cls["line"],
                        message=f"Class '{cls['name']}' missing docstring",
                        suggestion="Add a docstring describing the class's purpose",
                    )
                )

        return issues

    def _llm_style_analysis(
        self, code: str, functions: List[Dict], classes: List[Dict]
    ) -> List[Review_Issue]:
        """Use LLM to analyze code style."""
        issues = []

        prompt = f"""Analyze the following Python code for style issues and PEP 8 compliance.

Code:
```python
{code}
```

Functions:
{json.dumps(functions, indent=2)}

Classes:
{json.dumps(classes, indent=2)}

Check for:
1. PEP 8 violations (spacing, indentation, line length)
2. Naming convention issues
3. Code organization problems
4. Unused imports or variables
5. Code complexity issues
6. Comment quality

For each issue found, provide:
- Line number (if applicable)
- Severity (low or info)
- Clear description
- Suggestion for improvement

Respond in JSON format:
{{
  "issues": [
    {{
      "line_number": <int or null>,
      "severity": "<low|info>",
      "message": "<description>",
      "suggestion": "<improvement suggestion>",
      "code_snippet": "<relevant code>"
    }}
  ]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            for item in result.get("issues", []):
                severity_map = {
                    "low": Severity_Level.LOW,
                    "info": Severity_Level.INFO,
                }
                issues.append(
                    Review_Issue(
                        category=Issue_Category.STYLE,
                        severity=severity_map.get(
                            item.get("severity", "low").lower(),
                            Severity_Level.LOW,
                        ),
                        line_number=item.get("line_number"),
                        message=item.get("message", ""),
                        suggestion=item.get("suggestion"),
                        code_snippet=item.get("code_snippet"),
                    )
                )
        except Exception as e:
            pass

        return issues


class Code_Review_Agent:
    """Main agent that orchestrates code review process."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
    ):
        """Initialize code review agent."""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.bug_detector = Bug_Detector(self.client, model)
        self.security_analyzer = Security_Analyzer(self.client, model)
        self.style_checker = Style_Checker(self.client, model)

    def review_code(self, code: str, file_path: str = "inline") -> Review_Report:
        """Review code string and generate report."""
        parser = Code_Parser(code)
        if not parser.parse():
            return Review_Report(
                file_path=file_path,
                total_issues=1,
                issues=[
                    Review_Issue(
                        category=Issue_Category.BUG,
                        severity=Severity_Level.CRITICAL,
                        line_number=None,
                        message="Failed to parse code: Syntax error",
                    )
                ],
            )

        all_issues = []

        # Run all analyzers
        all_issues.extend(
            self.bug_detector.detect_bugs(
                code, parser.functions, parser.classes
            )
        )
        all_issues.extend(
            self.security_analyzer.analyze_security(
                code, parser.functions, parser.classes
            )
        )
        all_issues.extend(
            self.style_checker.check_style(
                code, parser.functions, parser.classes
            )
        )

        # Generate improvement suggestions
        all_issues.extend(self._suggest_improvements(code, parser))

        # Build report
        report = self._generate_report(file_path, all_issues)
        return report

    def review_file(self, file_path: str) -> Review_Report:
        """Review a Python file and generate report."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            return self.review_code(code, file_path)
        except FileNotFoundError:
            return Review_Report(
                file_path=file_path,
                total_issues=1,
                issues=[
                    Review_Issue(
                        category=Issue_Category.BUG,
                        severity=Severity_Level.CRITICAL,
                        line_number=None,
                        message=f"File not found: {file_path}",
                    )
                ],
            )
        except Exception as e:
            return Review_Report(
                file_path=file_path,
                total_issues=1,
                issues=[
                    Review_Issue(
                        category=Issue_Category.BUG,
                        severity=Severity_Level.CRITICAL,
                        line_number=None,
                        message=f"Error reading file: {str(e)}",
                    )
                ],
            )

    def _suggest_improvements(
        self, code: str, parser: Code_Parser
    ) -> List[Review_Issue]:
        """Suggest code improvements."""
        issues = []

        prompt = f"""Analyze the following Python code and suggest improvements.

Code:
```python
{code}
```

Functions:
{json.dumps(parser.functions, indent=2)}

Classes:
{json.dumps(parser.classes, indent=2)}

Suggest:
1. Refactoring opportunities
2. Design pattern applications
3. Performance optimizations
4. Code organization improvements
5. Best practice recommendations

For each suggestion, provide:
- Line number (if applicable)
- Severity (low or info)
- Clear description
- Specific improvement suggestion

Respond in JSON format:
{{
  "issues": [
    {{
      "line_number": <int or null>,
      "severity": "<low|info>",
      "message": "<description>",
      "suggestion": "<improvement suggestion>",
      "code_snippet": "<relevant code>"
    }}
  ]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            for item in result.get("issues", []):
                severity_map = {
                    "low": Severity_Level.LOW,
                    "info": Severity_Level.INFO,
                }
                issues.append(
                    Review_Issue(
                        category=Issue_Category.IMPROVEMENT,
                        severity=severity_map.get(
                            item.get("severity", "info").lower(),
                            Severity_Level.INFO,
                        ),
                        line_number=item.get("line_number"),
                        message=item.get("message", ""),
                        suggestion=item.get("suggestion"),
                        code_snippet=item.get("code_snippet"),
                    )
                )
        except Exception as e:
            pass

        return issues

    def _generate_report(
        self, file_path: str, issues: List[Review_Issue]
    ) -> Review_Report:
        """Generate structured report from issues."""
        issues_by_category = {}
        issues_by_severity = {}

        for issue in issues:
            cat = issue.category.value
            sev = issue.severity.value

            issues_by_category[cat] = issues_by_category.get(cat, 0) + 1
            issues_by_severity[sev] = issues_by_severity.get(sev, 0) + 1

        summary = self._generate_summary(issues, issues_by_category, issues_by_severity)

        return Review_Report(
            file_path=file_path,
            total_issues=len(issues),
            issues_by_category=issues_by_category,
            issues_by_severity=issues_by_severity,
            issues=issues,
            summary=summary,
        )

    def _generate_summary(
        self,
        issues: List[Review_Issue],
        issues_by_category: Dict[str, int],
        issues_by_severity: Dict[str, int],
    ) -> str:
        """Generate summary text for the report."""
        critical_count = issues_by_severity.get("critical", 0)
        high_count = issues_by_severity.get("high", 0)

        summary_parts = [
            f"Found {len(issues)} total issues across {len(issues_by_category)} categories."
        ]

        if critical_count > 0:
            summary_parts.append(
                f"CRITICAL: {critical_count} critical issues require immediate attention."
            )
        if high_count > 0:
            summary_parts.append(
                f"HIGH: {high_count} high-priority issues should be addressed soon."
            )

        bug_count = issues_by_category.get("bug", 0)
        security_count = issues_by_category.get("security", 0)

        if bug_count > 0:
            summary_parts.append(f"Found {bug_count} potential bugs.")
        if security_count > 0:
            summary_parts.append(f"Found {security_count} security vulnerabilities.")

        return " ".join(summary_parts)


def main():
    """Example usage of the Code Review Agent."""

    # Sample code with intentional issues
    sample_code = """
import random
import os

password = "admin123"
api_key = "sk-1234567890abcdef"

def process_user_data(user_id, query):
    sql = "SELECT * FROM users WHERE id = " + str(user_id)
    result = db.execute(sql)
    return result

def calculate_total(items):
    total = 0
    for i in range(len(items)):
        total += items[i]
    return total / len(items)

class UserManager:
    def __init__(self):
        self.users = []
    
    def addUser(self, name, email):
        self.users.append({"name": name, "email": email})
    
    def getUser(self, id):
        return self.users[id]

def generate_token():
    return random.randint(1000, 9999)

def read_file(filename):
    f = open(filename, 'r')
    data = f.read()
    return data
"""

    print("Initializing Code Review Agent...")
    agent = Code_Review_Agent()

    print("\nReviewing sample code...")
    report = agent.review_code(sample_code, "sample_code.py")

    print("\n" + "=" * 80)
    print(report.to_markdown())
    print("=" * 80)

    print("\nJSON Report:")
    print(json.dumps(report.to_dict(), indent=2))

    # Example: Review a file
    # report = agent.review_file("path/to/your/file.py")
    # print(report.to_markdown())


if __name__ == "__main__":
    main()
