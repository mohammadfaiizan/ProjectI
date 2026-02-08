"""
Agent module for Code Review Agent.
Contains LangGraph-based code review workflow.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from Tools import Issue, Parse_Python_Code, Calculate_Complexity, Check_Common_Patterns
from Config import Review_Config, SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW, SEVERITY_INFO
import json


class Review_State(TypedDict):
    """State schema for code review graph."""
    code: str
    language: str
    parsed_info: Dict[str, Any]
    bug_issues: List[Dict[str, Any]]
    security_issues: List[Dict[str, Any]]
    style_issues: List[Dict[str, Any]]
    performance_issues: List[Dict[str, Any]]
    summary: str
    overall_score: float


class Code_Review_Graph:
    """LangGraph-based code review agent."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        review_config: Optional[Review_Config] = None
    ):
        """
        Initialize code review graph.
        
        Args:
            llm: Language model instance
            review_config: Review configuration settings
        """
        self.llm = llm
        self.review_config = review_config or Review_Config()
        self.graph = self.Build_Graph()
        self.app = self.graph.compile()
    
    def Build_Graph(self) -> StateGraph:
        """
        Build the LangGraph state graph for code review.
        
        Returns:
            Configured StateGraph instance
        """
        workflow = StateGraph(Review_State)
        
        workflow.add_node("Parse_Code", self.Parse_Code)
        workflow.add_node("Check_Bugs", self.Check_Bugs)
        workflow.add_node("Check_Security", self.Check_Security)
        workflow.add_node("Check_Style", self.Check_Style)
        workflow.add_node("Check_Performance", self.Check_Performance)
        workflow.add_node("Aggregate_Report", self.Aggregate_Report)
        
        workflow.set_entry_point("Parse_Code")
        workflow.add_edge("Parse_Code", "Check_Bugs")
        workflow.add_edge("Check_Bugs", "Check_Security")
        workflow.add_edge("Check_Security", "Check_Style")
        workflow.add_edge("Check_Style", "Check_Performance")
        workflow.add_edge("Check_Performance", "Aggregate_Report")
        workflow.add_edge("Aggregate_Report", END)
        
        return workflow
    
    def Parse_Code(self, state: Review_State) -> Review_State:
        """
        Parse code and extract structure information.
        
        Args:
            state: Current review state
            
        Returns:
            Updated state with parsed information
        """
        code = state.get("code", "")
        parsed_info = Parse_Python_Code.invoke({"code": code})
        
        return {
            **state,
            "language": "python",
            "parsed_info": parsed_info,
            "bug_issues": [],
            "security_issues": [],
            "style_issues": [],
            "performance_issues": []
        }
    
    def Check_Bugs(self, state: Review_State) -> Review_State:
        """
        Check code for logical bugs using LLM analysis.
        
        Args:
            state: Current review state
            
        Returns:
            Updated state with bug issues
        """
        if not self.review_config.Is_Check_Enabled("bugs"):
            return {**state, "bug_issues": []}
        
        code = state.get("code", "")
        parsed_info = state.get("parsed_info", {})
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert code reviewer specializing in finding logical bugs.
Analyze the Python code and identify bugs such as:
- Off-by-one errors
- Null/None reference errors
- Incorrect variable usage
- Logic errors in conditionals
- Missing error handling
- Incorrect loop boundaries
- Type mismatches

Return a JSON array of issues. Each issue should have:
- severity: CRITICAL, HIGH, MEDIUM, LOW, or INFO
- category: "bug"
- description: Detailed description of the bug
- line_number: Line number where bug occurs (if applicable)
- suggestion: How to fix the bug

Limit to top 20 most critical bugs."""),
            HumanMessage(content=f"""Code to review:
```python
{code}
```

Code structure:
{json.dumps(parsed_info, indent=2)}

Identify logical bugs in this code.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({})
        
        bug_issues = self._Parse_LLM_Response(response.content, "bug")
        max_issues = self.review_config.Get_Max_Issues("bug_issues")
        bug_issues = bug_issues[:max_issues]
        
        return {**state, "bug_issues": bug_issues}
    
    def Check_Security(self, state: Review_State) -> Review_State:
        """
        Check code for security vulnerabilities.
        
        Args:
            state: Current review state
            
        Returns:
            Updated state with security issues
        """
        if not self.review_config.Is_Check_Enabled("security"):
            return {**state, "security_issues": []}
        
        code = state.get("code", "")
        
        regex_issues = Check_Common_Patterns.invoke({"code": code})
        security_regex = [issue for issue in regex_issues if issue.get("category") == "security"]
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a security expert code reviewer.
Analyze the Python code for security vulnerabilities such as:
- SQL injection
- Command injection
- Path traversal
- Insecure deserialization
- Hardcoded secrets
- Weak cryptography
- XSS vulnerabilities
- CSRF issues
- Authentication/authorization flaws

Return a JSON array of issues. Each issue should have:
- severity: CRITICAL, HIGH, MEDIUM, LOW, or INFO
- category: "security"
- description: Detailed description of the vulnerability
- line_number: Line number where issue occurs
- suggestion: How to fix the security issue

Limit to top 15 most critical security issues."""),
            HumanMessage(content=f"""Code to review:
```python
{code}
```

Identify security vulnerabilities in this code.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({})
        
        llm_issues = self._Parse_LLM_Response(response.content, "security")
        max_issues = self.review_config.Get_Max_Issues("security_issues")
        
        all_security_issues = security_regex + llm_issues
        all_security_issues = all_security_issues[:max_issues]
        
        return {**state, "security_issues": all_security_issues}
    
    def Check_Style(self, state: Review_State) -> Review_State:
        """
        Check code for style and PEP 8 compliance.
        
        Args:
            state: Current review state
            
        Returns:
            Updated state with style issues
        """
        if not self.review_config.Is_Check_Enabled("style"):
            return {**state, "style_issues": []}
        
        code = state.get("code", "")
        parsed_info = state.get("parsed_info", {})
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Python style expert reviewing code for PEP 8 compliance and best practices.
Check for:
- Naming conventions (functions, variables, classes)
- Code formatting and indentation
- Missing docstrings
- Line length violations
- Import organization
- Magic numbers
- Long functions/methods
- Code duplication
- Inconsistent style

Return a JSON array of issues. Each issue should have:
- severity: CRITICAL, HIGH, MEDIUM, LOW, or INFO
- category: "style"
- description: Detailed description of the style issue
- line_number: Line number where issue occurs
- suggestion: How to improve the code style

Limit to top 25 style issues."""),
            HumanMessage(content=f"""Code to review:
```python
{code}
```

Code structure:
{json.dumps(parsed_info, indent=2)}

Identify style and PEP 8 issues in this code.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({})
        
        style_issues = self._Parse_LLM_Response(response.content, "style")
        max_issues = self.review_config.Get_Max_Issues("style_issues")
        style_issues = style_issues[:max_issues]
        
        return {**state, "style_issues": style_issues}
    
    def Check_Performance(self, state: Review_State) -> Review_State:
        """
        Check code for performance anti-patterns.
        
        Args:
            state: Current review state
            
        Returns:
            Updated state with performance issues
        """
        if not self.review_config.Is_Check_Enabled("performance"):
            return {**state, "performance_issues": []}
        
        code = state.get("code", "")
        complexity_metrics = Calculate_Complexity.invoke({"code": code})
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a performance optimization expert.
Analyze the Python code for performance issues such as:
- Inefficient algorithms (O(n²) when O(n) is possible)
- Unnecessary loops or nested loops
- Inefficient string concatenation
- Missing caching opportunities
- Inefficient database queries
- Memory leaks
- Unnecessary object creation
- Inefficient data structures

Return a JSON array of issues. Each issue should have:
- severity: CRITICAL, HIGH, MEDIUM, LOW, or INFO
- category: "performance"
- description: Detailed description of the performance issue
- line_number: Line number where issue occurs
- suggestion: How to optimize the code

Limit to top 15 performance issues."""),
            HumanMessage(content=f"""Code to review:
```python
{code}
```

Complexity metrics:
{json.dumps(complexity_metrics, indent=2)}

Identify performance issues in this code.""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({})
        
        performance_issues = self._Parse_LLM_Response(response.content, "performance")
        max_issues = self.review_config.Get_Max_Issues("performance_issues")
        performance_issues = performance_issues[:max_issues]
        
        return {**state, "performance_issues": performance_issues}
    
    def Aggregate_Report(self, state: Review_State) -> Review_State:
        """
        Aggregate all findings and generate final report.
        
        Args:
            state: Current review state
            
        Returns:
            Updated state with summary and score
        """
        bug_issues = state.get("bug_issues", [])
        security_issues = state.get("security_issues", [])
        style_issues = state.get("style_issues", [])
        performance_issues = state.get("performance_issues", [])
        
        all_issues = bug_issues + security_issues + style_issues + performance_issues
        
        severity_weights = {
            SEVERITY_CRITICAL: 10,
            SEVERITY_HIGH: 5,
            SEVERITY_MEDIUM: 2,
            SEVERITY_LOW: 1,
            SEVERITY_INFO: 0.5
        }
        
        total_weight = sum(severity_weights.get(issue.get("severity", ""), 0) for issue in all_issues)
        max_possible_weight = 100
        score = max(0, 100 - min(total_weight, max_possible_weight))
        
        summary_parts = [
            f"Code Review Summary",
            f"Total Issues Found: {len(all_issues)}",
            f"  - Bugs: {len(bug_issues)}",
            f"  - Security: {len(security_issues)}",
            f"  - Style: {len(style_issues)}",
            f"  - Performance: {len(performance_issues)}",
            f"\nOverall Score: {score:.1f}/100"
        ]
        
        if all_issues:
            critical_count = sum(1 for issue in all_issues if issue.get("severity") == SEVERITY_CRITICAL)
            high_count = sum(1 for issue in all_issues if issue.get("severity") == SEVERITY_HIGH)
            
            if critical_count > 0:
                summary_parts.append(f"\nCRITICAL: {critical_count} critical issues found - immediate attention required!")
            elif high_count > 0:
                summary_parts.append(f"\nHIGH: {high_count} high-severity issues found.")
            else:
                summary_parts.append("\nNo critical or high-severity issues found.")
        
        summary = "\n".join(summary_parts)
        
        return {
            **state,
            "summary": summary,
            "overall_score": score
        }
    
    def _Parse_LLM_Response(self, response_text: str, category: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response and extract issues.
        
        Args:
            response_text: LLM response text
            category: Issue category
            
        Returns:
            List of issue dictionaries
        """
        issues = []
        
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()
            
            parsed = json.loads(json_text)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        item["category"] = category
                        issues.append(item)
        except (json.JSONDecodeError, ValueError):
            pass
        
        return issues
    
    def Review(self, code: str) -> Review_State:
        """
        Run full code review on provided code.
        
        Args:
            code: Source code string to review
            
        Returns:
            Final review state with all findings
        """
        initial_state: Review_State = {
            "code": code,
            "language": "python",
            "parsed_info": {},
            "bug_issues": [],
            "security_issues": [],
            "style_issues": [],
            "performance_issues": [],
            "summary": "",
            "overall_score": 0.0
        }
        
        final_state = self.app.invoke(initial_state)
        return final_state
