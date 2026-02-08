"""
Tools module for Multi-Agent Task Solver.
Provides specialized tools for each agent type and utility classes.
"""

import json
import re
from typing import Dict, List, Any, Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


@tool
def Search_Information(query: str) -> str:
    """
    Mock web search tool for research agent.
    Simulates searching for information on the web.
    
    Args:
        query: Search query string
        
    Returns:
        Mock search results as formatted string
    """
    mock_results = {
        "python web frameworks": """
        Top Python Web Frameworks:
        1. Django - Full-featured framework with ORM, admin panel, and security features
        2. Flask - Lightweight and flexible microframework
        3. FastAPI - Modern framework for building APIs with automatic documentation
        4. Pyramid - Flexible framework suitable for both small and large applications
        5. Tornado - Asynchronous framework for handling long-lived connections
        """,
        "microservices vs monolith": """
        Architecture Comparison:
        Monolith:
        - Single deployable unit
        - Easier development and testing
        - Simpler deployment
        - Can become complex as it grows
        
        Microservices:
        - Multiple independent services
        - Better scalability
        - Technology diversity
        - More complex deployment and monitoring
        """,
        "ai agents trends": """
        Current AI Agent Trends (2024-2025):
        - ReAct (Reasoning + Acting) pattern gaining popularity
        - Multi-agent systems for complex problem solving
        - Agentic AI frameworks: LangChain, LangGraph, AutoGen
        - Integration with LLMs for autonomous task execution
        - Focus on tool use and function calling capabilities
        """
    }
    
    query_lower = query.lower()
    for key, value in mock_results.items():
        if key in query_lower:
            return value
    
    return f"Search results for '{query}': Information found through comprehensive research. " \
           f"Key points include relevant details about {query} based on current knowledge base."


@tool
def Write_Python_Code(requirements: str) -> str:
    """
    Code generation tool for coding specialist.
    Generates Python code based on requirements.
    
    Args:
        requirements: Description of what code needs to be written
        
    Returns:
        Generated Python code as string
    """
    requirements_lower = requirements.lower()
    
    if "flask" in requirements_lower or "hello world" in requirements_lower:
        return '''from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '<h1>Hello, World!</h1>'

if __name__ == '__main__':
    app.run(debug=True)'''
    
    elif "microservice" in requirements_lower:
        return '''# Sample Microservice Structure
# service.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"data": "sample data"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# requirements.txt
# flask==2.3.0'''
    
    elif "react agent" in requirements_lower or "agent" in requirements_lower:
        return '''# Basic ReAct Agent Example
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools to answer questions."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Define tools here
tools = []

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What is the weather today?"})
print(result["output"])'''
    
    else:
        return f'''# Generated Python Code
# Requirements: {requirements}

def main():
    """
    Implementation based on requirements.
    """
    # TODO: Implement functionality
    pass

if __name__ == '__main__':
    main()'''


@tool
def Write_Content(brief: str) -> str:
    """
    Content writing tool for writing specialist.
    Generates written content based on a brief.
    
    Args:
        brief: Content brief describing what to write
        
    Returns:
        Generated content as string
    """
    brief_lower = brief.lower()
    
    if "comparison" in brief_lower or "framework" in brief_lower:
        return """Python Web Frameworks Comparison

Django:
- Full-featured framework with built-in ORM, admin panel, and authentication
- Best for: Large applications, rapid development, content management
- Learning curve: Moderate to steep
- Performance: Good for most use cases

Flask:
- Lightweight microframework with minimal dependencies
- Best for: Small to medium applications, APIs, custom solutions
- Learning curve: Gentle
- Performance: Excellent, minimal overhead

FastAPI:
- Modern framework with automatic API documentation
- Best for: APIs, high-performance applications, async operations
- Learning curve: Moderate
- Performance: Excellent, comparable to Node.js

Recommendation: Choose Django for full-featured apps, Flask for flexibility, FastAPI for APIs."""
    
    elif "recommendation" in brief_lower or "microservice" in brief_lower:
        return """Microservices vs Monolith: Recommendation Report

Executive Summary:
The choice between microservices and monolith depends on team size, complexity, and scalability needs.

When to Choose Monolith:
- Small to medium teams
- Simple to moderate complexity
- Single deployment unit is sufficient
- Faster initial development needed

When to Choose Microservices:
- Large, distributed teams
- High scalability requirements
- Need for technology diversity
- Independent deployment cycles

Recommendation:
Start with a monolith for MVP and early stages. Migrate to microservices when:
1. Team grows beyond 10-15 developers
2. Different services have different scaling needs
3. Independent deployment becomes critical
4. Clear service boundaries emerge"""
    
    elif "summary" in brief_lower or "trends" in brief_lower or "ai agents" in brief_lower:
        return """AI Agents: Current Trends and Overview

Key Trends:
1. ReAct Pattern: Combining reasoning and acting for better problem-solving
2. Multi-Agent Systems: Multiple specialized agents collaborating
3. Framework Adoption: LangChain, LangGraph, AutoGen gaining traction
4. Tool Integration: Agents using external APIs and functions
5. Autonomous Execution: Agents handling end-to-end workflows

Statistics:
- 70% of AI applications now incorporate agentic patterns
- Multi-agent systems show 40% better performance on complex tasks
- ReAct agents achieve 85% accuracy on tool-using tasks

Beginner-Friendly Explanation:
AI agents are programs that can think, plan, and act autonomously. Unlike simple chatbots, agents can:
- Break down complex problems into steps
- Use tools and APIs to gather information
- Make decisions based on context
- Execute actions to complete tasks

Think of them as AI assistants that can actually do things, not just answer questions."""
    
    else:
        return f"""Content Generated Based on Brief

Brief: {brief}

[Content would be generated here based on the specific requirements in the brief. This includes relevant information, structured presentation, and clear explanations tailored to the audience.]"""


@tool
def Analyze_Data(data: str, question: str) -> str:
    """
    Data analysis tool for analysis specialist.
    Analyzes provided data to answer questions.
    
    Args:
        data: Data to analyze (can be text, structured data, etc.)
        question: Question to answer based on the data
        
    Returns:
        Analysis results as string
    """
    question_lower = question.lower()
    
    if "pros" in question_lower or "cons" in question_lower or "compare" in question_lower:
        return """Analysis: Pros and Cons Comparison

Pros of Monolith:
- Simpler development and debugging
- Easier testing (single codebase)
- Faster initial development
- Lower operational complexity

Cons of Monolith:
- Can become difficult to maintain at scale
- Limited scalability options
- Technology lock-in
- Slower deployment cycles

Pros of Microservices:
- Independent scaling
- Technology diversity
- Team autonomy
- Fault isolation

Cons of Microservices:
- Increased complexity
- Network latency
- Distributed system challenges
- More operational overhead"""
    
    elif "statistics" in question_lower or "trends" in question_lower:
        return """Data Analysis: AI Agent Trends Statistics

From the research data:
- Framework Adoption: LangChain leads with 45% market share
- Use Cases: 60% in automation, 30% in customer service, 10% in research
- Performance Metrics: ReAct agents show 85% task completion rate
- Growth Rate: 200% year-over-year increase in agent implementations
- Average Agents per System: 3-5 agents for typical multi-agent setups"""
    
    else:
        return f"""Data Analysis Results

Data Provided: {data[:200]}...

Question: {question}

Analysis: Based on the provided data, the key insights are:
1. Relevant patterns and trends identified
2. Important correlations found
3. Recommendations derived from the analysis
4. Statistical summary of key metrics"""
    
    return "Analysis completed successfully."


@tool
def Review_Output(content: str, criteria: str) -> str:
    """
    Quality review tool for reviewing agent outputs.
    Evaluates content against specified criteria.
    
    Args:
        content: Content to review
        criteria: Review criteria
        
    Returns:
        Review feedback as string
    """
    review_points = []
    
    if "completeness" in criteria.lower():
        review_points.append("Completeness: Content covers all required aspects")
    
    if "accuracy" in criteria.lower():
        review_points.append("Accuracy: Information is correct and up-to-date")
    
    if "clarity" in criteria.lower():
        review_points.append("Clarity: Content is well-structured and easy to understand")
    
    if "code quality" in criteria.lower() or "code" in criteria.lower():
        review_points.append("Code Quality: Code follows best practices and is well-commented")
    
    if not review_points:
        review_points.append("General Review: Content meets quality standards")
    
    return f"""Quality Review Results

Criteria: {criteria}

Review Points:
{chr(10).join(f'- {point}' for point in review_points)}

Overall Assessment: Content quality is satisfactory and meets the specified criteria.
Recommendations: Minor improvements could enhance clarity and completeness."""


class Task_Parser:
    """Parses complex task descriptions into structured subtasks."""
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initialize task parser.
        
        Args:
            llm: Language model for parsing tasks
        """
        self.llm = llm
        self.parsing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a task decomposition expert. Break down complex tasks into 
            clear, actionable subtasks. Each subtask should be:
            1. Specific and actionable
            2. Assigned to an appropriate specialist type (research, coding, writing, analysis)
            3. Independent enough to be executed separately
            
            Return a JSON array of subtasks, each with:
            - "description": task description
            - "specialist": specialist type (research/coding/writing/analysis)
            - "order": execution order (integer)
            """),
            ("human", "Task: {task}")
        ])
    
    def parse_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Parse a complex task into structured subtasks.
        
        Args:
            task_description: Complex task description
            
        Returns:
            List of subtask dictionaries
        """
        chain = self.parsing_prompt | self.llm
        response = chain.invoke({"task": task_description})
        
        content = response.content
        
        try:
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                subtasks = json.loads(json_match.group())
            else:
                subtasks = self._fallback_parse(task_description)
        except json.JSONDecodeError:
            subtasks = self._fallback_parse(task_description)
        
        return subtasks
    
    def _fallback_parse(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Fallback parsing when LLM parsing fails.
        Uses keyword-based heuristics.
        
        Args:
            task_description: Task description
            
        Returns:
            List of subtask dictionaries
        """
        task_lower = task_description.lower()
        subtasks = []
        order = 1
        
        if any(word in task_lower for word in ["research", "find", "search"]):
            subtasks.append({
                "description": "Research and gather information",
                "specialist": "research",
                "order": order
            })
            order += 1
        
        if any(word in task_lower for word in ["write code", "code", "implement", "example"]):
            subtasks.append({
                "description": "Generate code implementation",
                "specialist": "coding",
                "order": order
            })
            order += 1
        
        if any(word in task_lower for word in ["write", "create", "report", "summary"]):
            subtasks.append({
                "description": "Create written content",
                "specialist": "writing",
                "order": order
            })
            order += 1
        
        if any(word in task_lower for word in ["analyze", "compare", "evaluate"]):
            subtasks.append({
                "description": "Analyze and evaluate",
                "specialist": "analysis",
                "order": order
            })
            order += 1
        
        if not subtasks:
            subtasks.append({
                "description": task_description,
                "specialist": "research",
                "order": 1
            })
        
        return subtasks


class Result_Aggregator:
    """Combines outputs from multiple agents into a final result."""
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initialize result aggregator.
        
        Args:
            llm: Language model for aggregation
        """
        self.llm = llm
        self.aggregation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a result aggregation expert. Combine outputs from multiple 
            specialist agents into a coherent, well-structured final result. The aggregated 
            result should:
            1. Integrate all specialist outputs seamlessly
            2. Maintain logical flow and structure
            3. Remove redundancies
            4. Ensure completeness
            
            Provide a comprehensive final result that addresses the original task."""),
            ("human", """Original Task: {task}
            
            Agent Outputs:
            {outputs}
            
            Please aggregate these into a final comprehensive result.""")
        ])
    
    def aggregate(self, task: str, agent_outputs: Dict[str, str]) -> str:
        """
        Aggregate multiple agent outputs into final result.
        
        Args:
            task: Original task description
            agent_outputs: Dictionary mapping agent names to their outputs
            
        Returns:
            Aggregated final result
        """
        outputs_text = "\n\n".join(
            f"{agent_name}:\n{output}"
            for agent_name, output in agent_outputs.items()
        )
        
        chain = self.aggregation_prompt | self.llm
        response = chain.invoke({
            "task": task,
            "outputs": outputs_text
        })
        
        return response.content
