"""
Agent module for Data Analysis Agent.

This module contains the LangGraph-based data analysis agent with stateful
workflow for understanding data schema, generating pandas code, executing
code safely, handling errors, and interpreting results.
"""

from typing import TypedDict, Annotated, Literal, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from Config import LLM_Config, Analysis_Config, Execution_Config
from Tools import Load_CSV_Data, Execute_Pandas_Code, Data_Profiler, Safe_Executor


# ============================================================================
# State Definition
# ============================================================================

class Analysis_State(TypedDict):
    """
    State schema for the data analysis agent.
    
    Tracks data, schema, question, generated code, execution results,
    interpretation, errors, and retry count throughout the analysis workflow.
    """
    data_json: Optional[str]
    schema: Optional[Dict[str, Any]]
    question: str
    generated_code: Optional[str]
    execution_result: Optional[Dict[str, Any]]
    interpretation: Optional[str]
    error: Optional[str]
    retry_count: int


# ============================================================================
# Data Analysis Graph
# ============================================================================

class Data_Analysis_Graph:
    """
    LangGraph-based data analysis agent.
    
    Implements a stateful workflow that:
    1. Understands data schema
    2. Generates pandas code to answer questions
    3. Executes code safely
    4. Handles errors with retry logic
    5. Interprets results in natural language
    """
    
    def __init__(
        self,
        llm_config: LLM_Config,
        analysis_config: Analysis_Config,
        execution_config: Execution_Config
    ):
        """
        Initialize the data analysis graph.
        
        Args:
            llm_config: LLM configuration instance
            analysis_config: Analysis configuration instance
            execution_config: Execution configuration instance
        """
        self.llm = llm_config.Get_LLM()
        self.analysis_config = analysis_config
        self.execution_config = execution_config
        self.safe_executor = Safe_Executor(
            allowed_imports=execution_config.Get_Allowed_Imports(),
            blocked_imports=execution_config.Get_Blocked_Imports()
        )
        self.graph = None
        self.app = None
        self._Build_Graph()
    
    def _Build_Graph(self):
        """Build the LangGraph workflow with nodes and edges."""
        workflow = StateGraph(Analysis_State)
        
        # Add nodes
        workflow.add_node("understand_schema", self._Understand_Schema)
        workflow.add_node("generate_code", self._Generate_Code)
        workflow.add_node("execute_code", self._Execute_Code)
        workflow.add_node("handle_error", self._Handle_Error)
        workflow.add_node("interpret_results", self._Interpret_Results)
        
        # Set entry point
        workflow.set_entry_point("understand_schema")
        
        # Add edges
        workflow.add_edge("understand_schema", "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        
        # Conditional edge from execute_code
        workflow.add_conditional_edges(
            "execute_code",
            self._Check_Execution_Result,
            {
                "success": "interpret_results",
                "error": "handle_error"
            }
        )
        
        # Conditional edge from handle_error
        workflow.add_conditional_edges(
            "handle_error",
            self._Check_Retry_Count,
            {
                "retry": "generate_code",
                "max_retries": "interpret_results"
            }
        )
        
        # Interpret results routes to END
        workflow.add_edge("interpret_results", END)
        
        # Compile the graph
        self.graph = workflow
        self.app = workflow.compile()
    
    def _Understand_Schema(self, state: Analysis_State) -> Analysis_State:
        """
        Analyze data structure, types, and sample values.
        
        Uses Data_Profiler to generate comprehensive schema information.
        """
        data_json = state.get("data_json")
        
        if not data_json:
            return {
                **state,
                "error": "No data provided for analysis",
                "schema": None
            }
        
        try:
            import pandas as pd
            import json
            
            # Load DataFrame from JSON
            df = pd.read_json(data_json, orient="records")
            
            # Create profiler and generate statistics
            profiler = Data_Profiler(df)
            stats = profiler.Generate_Statistics()
            type_mapping = profiler.Detect_Types()
            
            # Build comprehensive schema
            schema = {
                "shape": stats["shape"],
                "columns": list(df.columns),
                "dtypes": stats["dtypes"],
                "type_mapping": type_mapping,
                "missing_values": stats["missing_values"],
                "missing_percentage": stats["missing_percentage"],
                "numeric_stats": stats.get("numeric_stats", {}),
                "categorical_stats": stats.get("categorical_stats", {}),
                "sample_data": df.head(5).to_dict(orient="records")
            }
            
            return {
                **state,
                "schema": schema,
                "error": None
            }
        
        except Exception as e:
            return {
                **state,
                "error": f"Error understanding schema: {str(e)}",
                "schema": None
            }
    
    def _Generate_Code(self, state: Analysis_State) -> Analysis_State:
        """
        Generate pandas code to answer the user's question.
        
        Uses LLM to generate code based on schema and question.
        """
        schema = state.get("schema")
        question = state.get("question", "")
        data_json = state.get("data_json")
        
        if not schema:
            return {
                **state,
                "error": "Schema not available. Cannot generate code.",
                "generated_code": None
            }
        
        # Build schema description for prompt
        schema_description = f"""
Data Shape: {schema.get('shape', {}).get('rows', 0)} rows, {schema.get('shape', {}).get('columns', 0)} columns
Columns: {', '.join(schema.get('columns', []))}

Column Types:
"""
        for col, col_type in schema.get("type_mapping", {}).items():
            dtype = schema.get("dtypes", {}).get(col, "unknown")
            schema_description += f"  - {col}: {col_type} ({dtype})\n"
        
        schema_description += "\nSample Data (first 3 rows):\n"
        for i, row in enumerate(schema.get("sample_data", [])[:3]):
            schema_description += f"  Row {i+1}: {row}\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data analyst. Generate pandas code to answer the user's question about the dataset.

The data is already loaded in a variable called 'df' (pandas DataFrame).
Your code should:
1. Use pandas operations to analyze the data
2. Store the result in a variable called 'result'
3. If the result is a DataFrame or Series, it will be automatically converted to JSON
4. Use only pandas, numpy, and standard Python built-ins
5. Do NOT import any modules (they are already available as pd, np)
6. Do NOT use file operations, subprocess, or system calls
7. Return ONLY the Python code, no explanations or markdown

Example format:
result = df.groupby('category')['sales'].sum()
"""),
            ("human", """Schema Information:
{schema}

Question: {question}

Generate pandas code to answer this question:""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "schema": schema_description,
            "question": question
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
            **state,
            "generated_code": code,
            "error": None
        }
    
    def _Execute_Code(self, state: Analysis_State) -> Analysis_State:
        """
        Safely execute the generated pandas code.
        
        Uses Safe_Executor to execute code with safety checks and timeout.
        """
        code = state.get("generated_code")
        data_json = state.get("data_json")
        
        if not code:
            return {
                **state,
                "execution_result": {
                    "success": False,
                    "error": "No code to execute"
                },
                "error": "No code generated"
            }
        
        if not data_json:
            return {
                **state,
                "execution_result": {
                    "success": False,
                    "error": "No data available"
                },
                "error": "No data available"
            }
        
        try:
            # Use Execute_Pandas_Code tool
            result = Execute_Pandas_Code.invoke({
                "code": code,
                "dataframe_json": data_json
            })
            
            return {
                **state,
                "execution_result": result,
                "error": None if result.get("success") else result.get("error")
            }
        
        except Exception as e:
            return {
                **state,
                "execution_result": {
                    "success": False,
                    "error": str(e)
                },
                "error": str(e)
            }
    
    def _Handle_Error(self, state: Analysis_State) -> Analysis_State:
        """
        Handle execution errors by asking LLM to fix the code.
        
        Analyzes the error and generates a corrected version of the code.
        """
        error = state.get("error")
        code = state.get("generated_code")
        question = state.get("question", "")
        schema = state.get("schema")
        
        if not error:
            return state
        
        # Build schema description
        schema_description = ""
        if schema:
            schema_description = f"""
Data Shape: {schema.get('shape', {}).get('rows', 0)} rows, {schema.get('shape', {}).get('columns', 0)} columns
Columns: {', '.join(schema.get('columns', []))}
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are debugging pandas code. The previous code failed with an error.
Analyze the error and generate a corrected version of the code.

The data is already loaded in a variable called 'df' (pandas DataFrame).
Your corrected code should:
1. Fix the error from the previous attempt
2. Store the result in a variable called 'result'
3. Use only pandas, numpy, and standard Python built-ins
4. Return ONLY the corrected Python code, no explanations or markdown
"""),
            ("human", """Original Question: {question}

Schema Information:
{schema}

Previous Code:
```python
{code}
```

Error Message:
{error}

Generate corrected code:""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "question": question,
            "schema": schema_description,
            "code": code,
            "error": error
        })
        
        corrected_code = response.content.strip()
        
        # Remove markdown code blocks if present
        if corrected_code.startswith("```python"):
            corrected_code = corrected_code[9:]
        elif corrected_code.startswith("```"):
            corrected_code = corrected_code[3:]
        if corrected_code.endswith("```"):
            corrected_code = corrected_code[:-3]
        corrected_code = corrected_code.strip()
        
        return {
            **state,
            "generated_code": corrected_code,
            "error": None,
            "retry_count": state.get("retry_count", 0) + 1
        }
    
    def _Interpret_Results(self, state: Analysis_State) -> Analysis_State:
        """
        Interpret execution results in natural language.
        
        Uses LLM to explain the analysis results to the user.
        """
        execution_result = state.get("execution_result")
        question = state.get("question", "")
        code = state.get("generated_code")
        error = state.get("error")
        
        if error and not execution_result:
            interpretation = f"I encountered an error while analyzing the data: {error}"
            return {
                **state,
                "interpretation": interpretation
            }
        
        if not execution_result or not execution_result.get("success"):
            error_msg = execution_result.get("error", "Unknown error") if execution_result else "No execution result"
            interpretation = f"I was unable to complete the analysis. Error: {error_msg}"
            return {
                **state,
                "interpretation": interpretation
            }
        
        output = execution_result.get("output", "")
        result_data = execution_result.get("result_data")
        
        # Parse result data if available
        result_summary = ""
        if result_data:
            try:
                import json
                parsed_data = json.loads(result_data)
                if isinstance(parsed_data, list) and len(parsed_data) > 0:
                    result_summary = f"Found {len(parsed_data)} result rows. "
                    if len(parsed_data) <= 5:
                        result_summary += f"Results: {parsed_data}"
                    else:
                        result_summary += f"First 3 results: {parsed_data[:3]}"
                elif isinstance(parsed_data, dict):
                    result_summary = f"Result: {parsed_data}"
            except:
                result_summary = output[:500]  # Truncate if too long
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analyst explaining results to a user.
Interpret the analysis results in clear, natural language.
Focus on answering the user's question directly and explaining what the data shows.
Be concise but informative."""),
            ("human", """Question: {question}

Code Executed:
```python
{code}
```

Execution Output:
{output}

Result Data:
{result_summary}

Provide a clear interpretation of these results:""")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "question": question,
            "code": code,
            "output": output,
            "result_summary": result_summary
        })
        
        interpretation = response.content.strip()
        
        return {
            **state,
            "interpretation": interpretation
        }
    
    def _Check_Execution_Result(self, state: Analysis_State) -> Literal["success", "error"]:
        """
        Check if code execution was successful.
        
        Returns 'success' if execution succeeded, 'error' otherwise.
        """
        execution_result = state.get("execution_result")
        error = state.get("error")
        
        if execution_result and execution_result.get("success") and not error:
            return "success"
        return "error"
    
    def _Check_Retry_Count(self, state: Analysis_State) -> Literal["retry", "max_retries"]:
        """
        Check if we should retry or give up.
        
        Returns 'retry' if retry count is below max, 'max_retries' otherwise.
        """
        retry_count = state.get("retry_count", 0)
        max_retries = self.execution_config.Get_Max_Retries()
        
        if retry_count < max_retries:
            return "retry"
        return "max_retries"
    
    def Ask(self, question: str, data: str) -> Dict[str, Any]:
        """
        Ask a question about the data and get analysis results.
        
        Args:
            question: Natural language question about the data
            data: JSON string representation of the DataFrame
        
        Returns:
            Dictionary containing:
            - interpretation: Natural language interpretation of results
            - code: Generated pandas code
            - execution_result: Execution results
            - schema: Data schema information
            - error: Error message if any
        """
        if not self.app:
            raise RuntimeError("Graph not compiled. Call _Build_Graph() first.")
        
        initial_state = {
            "data_json": data,
            "schema": None,
            "question": question,
            "generated_code": None,
            "execution_result": None,
            "interpretation": None,
            "error": None,
            "retry_count": 0
        }
        
        result = self.app.invoke(initial_state)
        
        return {
            "interpretation": result.get("interpretation", "No interpretation available"),
            "code": result.get("generated_code"),
            "execution_result": result.get("execution_result"),
            "schema": result.get("schema"),
            "error": result.get("error"),
            "retry_count": result.get("retry_count", 0)
        }
