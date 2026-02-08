"""
CrewAI Advanced Workflows: Complex Multi-Agent Systems

This module demonstrates advanced CrewAI patterns including:
1. Research and Content Pipeline with 4 agents
2. Code Review Crew with specialized reviewers
3. Customer Support Crew with classification and escalation
4. Crew with Memory for learning across runs
5. Async Tasks for parallel execution
6. Callbacks and Event Handling for monitoring
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from typing import List, Dict, Optional, Callable
import asyncio
import json
from datetime import datetime


# ============================================================================
# ADVANCED CUSTOM TOOLS
# ============================================================================

@tool
def analyze_code_complexity(code: str) -> Dict[str, float]:
    """
    Analyze code complexity metrics.
    
    Args:
        code: Source code to analyze
        
    Returns:
        Dictionary with complexity metrics
    """
    lines = code.split('\n')
    functions = code.count('def ')
    classes = code.count('class ')
    
    # Simple complexity calculation
    complexity = len(lines) * 0.1 + functions * 2 + classes * 3
    
    return {
        "lines_of_code": len(lines),
        "function_count": functions,
        "class_count": classes,
        "complexity_score": complexity
    }


@tool
def check_security_vulnerabilities(code: str) -> List[Dict[str, str]]:
    """
    Check for common security vulnerabilities in code.
    
    Args:
        code: Source code to check
        
    Returns:
        List of potential vulnerabilities
    """
    vulnerabilities = []
    
    # Simple pattern matching for common issues
    if 'eval(' in code:
        vulnerabilities.append({
            "type": "Dangerous Function",
            "severity": "High",
            "description": "Use of eval() function detected"
        })
    
    if 'password' in code.lower() and '=' in code:
        vulnerabilities.append({
            "type": "Hardcoded Credentials",
            "severity": "Critical",
            "description": "Potential hardcoded password detected"
        })
    
    if 'sql' in code.lower() and '+' in code:
        vulnerabilities.append({
            "type": "SQL Injection Risk",
            "severity": "High",
            "description": "String concatenation in SQL queries detected"
        })
    
    return vulnerabilities


@tool
def classify_customer_query(query: str) -> Dict[str, str]:
    """
    Classify a customer support query into categories.
    
    Args:
        query: Customer query text
        
    Returns:
        Classification result with category and priority
    """
    query_lower = query.lower()
    
    # Simple classification logic
    if any(word in query_lower for word in ['bug', 'error', 'broken', 'not working']):
        category = "Technical Issue"
        priority = "High"
    elif any(word in query_lower for word in ['refund', 'cancel', 'money', 'payment']):
        category = "Billing"
        priority = "High"
    elif any(word in query_lower for word in ['how', 'tutorial', 'guide', 'help']):
        category = "How-To"
        priority = "Medium"
    elif any(word in query_lower for word in ['feature', 'request', 'suggestion']):
        category = "Feature Request"
        priority = "Low"
    else:
        category = "General Inquiry"
        priority = "Medium"
    
    return {
        "category": category,
        "priority": priority,
        "requires_escalation": priority == "High"
    }


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for relevant information.
    
    Args:
        query: Search query
        
    Returns:
        Relevant knowledge base articles
    """
    # In a real implementation, this would query a knowledge base
    return f"Knowledge base results for '{query}': Found relevant articles and documentation."


@tool
def create_support_ticket(ticket_data: Dict[str, str]) -> str:
    """
    Create a support ticket in the system.
    
    Args:
        ticket_data: Dictionary with ticket information
        
    Returns:
        Ticket ID
    """
    ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return f"Created ticket {ticket_id} with data: {json.dumps(ticket_data)}"


# ============================================================================
# EXAMPLE 1: RESEARCH AND CONTENT PIPELINE
# ============================================================================

def research_and_content_pipeline():
    """
    Advanced content creation pipeline with 4 specialized agents:
    1. Researcher - gathers comprehensive information
    2. Analyst - analyzes and synthesizes research
    3. Writer - creates content based on analysis
    4. Editor - reviews and improves content
    
    Returns:
        Crew execution result
    """
    # Researcher Agent
    researcher = Agent(
        role="Senior Research Specialist",
        goal="Conduct comprehensive research on assigned topics and gather reliable information",
        backstory="""You are a senior research specialist with 15 years of experience
                   in academic and industry research. You have expertise in information
                   gathering, source verification, and data collection. You excel at
                   finding authoritative sources and ensuring information accuracy.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_knowledge_base]
    )
    
    # Analyst Agent
    analyst = Agent(
        role="Data Analyst and Strategist",
        goal="Analyze research data, identify patterns, and develop strategic insights",
        backstory="""You are a data analyst with expertise in processing complex
                   information and extracting actionable insights. You have worked
                   for top consulting firms and excel at identifying trends, patterns,
                   and strategic implications from research data.""",
        verbose=True,
        allow_delegation=False
    )
    
    # Writer Agent
    writer = Agent(
        role="Senior Content Writer",
        goal="Create high-quality, engaging content based on research and analysis",
        backstory="""You are an award-winning content writer with expertise in
                   creating compelling narratives from complex information. You have
                   written for major publications and excel at making technical
                   content accessible and engaging for diverse audiences.""",
        verbose=True,
        allow_delegation=False
    )
    
    # Editor Agent
    editor = Agent(
        role="Senior Editor and Quality Assurance Specialist",
        goal="Review, edit, and ensure content meets highest quality standards",
        backstory="""You are a senior editor with 20 years of experience in
                   publishing and content quality assurance. You have edited
                   content for major publications and have a reputation for
                   meticulous attention to detail and high standards.""",
        verbose=True,
        allow_delegation=False
    )
    
    # Define tasks
    research_task = Task(
        description="""Research the topic of 'The Future of Remote Work: Trends and Implications'.
                      Gather comprehensive information including:
                      - Current remote work statistics and trends
                      - Technology enabling remote work
                      - Challenges and solutions
                      - Future predictions and implications
                      - Expert opinions and case studies""",
        expected_output="Comprehensive research report with sources, statistics, and key findings",
        agent=researcher
    )
    
    analysis_task = Task(
        description="""Analyze the research data on remote work trends. Identify:
                      - Key patterns and trends
                      - Significant correlations
                      - Strategic implications for businesses
                      - Opportunities and challenges
                      - Actionable insights for decision-makers""",
        expected_output="Strategic analysis report with insights, implications, and recommendations",
        agent=analyst,
        context=[research_task]
    )
    
    writing_task = Task(
        description="""Write a comprehensive 2000-word article on 'The Future of Remote Work'
                      based on the research and analysis. The article should:
                      - Have a compelling introduction
                      - Present key findings clearly
                      - Include data visualizations descriptions
                      - Provide actionable insights
                      - End with a strong conclusion
                      - Be engaging and accessible""",
        expected_output="Complete article ready for publication, approximately 2000 words",
        agent=writer,
        context=[research_task, analysis_task]
    )
    
    editing_task = Task(
        description="""Review and edit the article for:
                      - Accuracy (verify facts against research)
                      - Clarity and readability
                      - Structure and flow
                      - Grammar and style
                      - Engagement and impact
                      Ensure the article meets publication standards.""",
        expected_output="Edited and polished article ready for publication",
        agent=editor,
        context=[writing_task]
    )
    
    # Create crew
    crew = Crew(
        agents=[researcher, analyst, writer, editor],
        tasks=[research_task, analysis_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute crew
    result = crew.kickoff()
    return result


# ============================================================================
# EXAMPLE 2: CODE REVIEW CREW
# ============================================================================

def code_review_crew():
    """
    Specialized code review crew with three types of reviewers:
    1. Bug Reviewer - identifies bugs and issues
    2. Security Reviewer - checks for security vulnerabilities
    3. Performance Reviewer - analyzes performance and optimization
    
    Returns:
        Crew execution result
    """
    # Bug Reviewer Agent
    bug_reviewer = Agent(
        role="Senior Code Reviewer - Bug Detection Specialist",
        goal="Identify bugs, logic errors, and potential issues in code",
        backstory="""You are a senior software engineer with 12 years of experience
                   in code review and quality assurance. You have an exceptional eye
                   for detail and have caught thousands of bugs before they reached
                   production. You specialize in logic errors, edge cases, and
                   potential runtime issues.""",
        verbose=True,
        allow_delegation=False,
        tools=[analyze_code_complexity]
    )
    
    # Security Reviewer Agent
    security_reviewer = Agent(
        role="Security Specialist - Code Security Auditor",
        goal="Identify security vulnerabilities and security best practice violations",
        backstory="""You are a cybersecurity expert specializing in secure coding
                   practices. You have worked for security firms and have extensive
                   knowledge of common vulnerabilities, OWASP Top 10, and secure
                   coding patterns. You excel at identifying security risks in code.""",
        verbose=True,
        allow_delegation=False,
        tools=[check_security_vulnerabilities]
    )
    
    # Performance Reviewer Agent
    performance_reviewer = Agent(
        role="Performance Engineer - Code Optimization Specialist",
        goal="Identify performance issues and optimization opportunities",
        backstory="""You are a performance engineering expert with expertise in
                   code optimization, algorithm efficiency, and system performance.
                   You have optimized code for high-traffic systems and excel at
                   identifying bottlenecks and optimization opportunities.""",
        verbose=True,
        allow_delegation=False,
        tools=[analyze_code_complexity]
    )
    
    # Sample code for review
    sample_code = """
def process_user_data(user_id, data):
    password = data.get('password')
    query = "SELECT * FROM users WHERE id = " + user_id
    
    result = eval(query)
    
    for item in data:
        process_item(item)
        process_item(item)  # Duplicate processing
    
    return result
"""
    
    # Define review tasks
    bug_review_task = Task(
        description=f"""Review the following code for bugs, logic errors, and potential issues:
                     
                     {sample_code}
                     
                     Identify:
                     - Logic errors
                     - Edge cases not handled
                     - Potential runtime errors
                     - Code smells
                     - Best practice violations""",
        expected_output="Detailed bug report with identified issues and recommendations",
        agent=bug_reviewer
    )
    
    security_review_task = Task(
        description=f"""Review the following code for security vulnerabilities:
                     
                     {sample_code}
                     
                     Check for:
                     - Injection vulnerabilities
                     - Authentication issues
                     - Authorization problems
                     - Data exposure risks
                     - Security best practice violations""",
        expected_output="Security audit report with vulnerabilities and remediation steps",
        agent=security_reviewer
    )
    
    performance_review_task = Task(
        description=f"""Review the following code for performance issues:
                     
                     {sample_code}
                     
                     Analyze:
                     - Algorithm efficiency
                     - Unnecessary computations
                     - Memory usage
                     - Optimization opportunities
                     - Scalability concerns""",
        expected_output="Performance analysis report with issues and optimization recommendations",
        agent=performance_reviewer
    )
    
    # Create crew
    crew = Crew(
        agents=[bug_reviewer, security_reviewer, performance_reviewer],
        tasks=[bug_review_task, security_review_task, performance_review_task],
        process=Process.sequential,  # Can also use hierarchical with manager
        verbose=True
    )
    
    # Execute crew
    result = crew.kickoff()
    return result


# ============================================================================
# EXAMPLE 3: CUSTOMER SUPPORT CREW
# ============================================================================

def customer_support_crew():
    """
    Customer support crew with classification, specialist routing, and escalation:
    1. Classifier - categorizes customer queries
    2. Technical Specialist - handles technical issues
    3. Billing Specialist - handles billing inquiries
    4. Escalation Agent - handles complex or high-priority issues
    
    Returns:
        Crew execution result
    """
    # Classifier Agent
    classifier = Agent(
        role="Customer Query Classifier",
        goal="Accurately classify customer queries and route them to appropriate specialists",
        backstory="""You are an expert in customer service classification with
                   experience analyzing thousands of customer queries. You excel
                   at quickly understanding query intent and routing to the
                   right specialist.""",
        verbose=True,
        allow_delegation=True,
        tools=[classify_customer_query]
    )
    
    # Technical Specialist Agent
    technical_specialist = Agent(
        role="Technical Support Specialist",
        goal="Resolve technical issues and provide technical assistance to customers",
        backstory="""You are a technical support specialist with deep product
                   knowledge and troubleshooting expertise. You excel at diagnosing
                   technical issues and providing clear, actionable solutions to
                   customers.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_knowledge_base]
    )
    
    # Billing Specialist Agent
    billing_specialist = Agent(
        role="Billing and Account Specialist",
        goal="Handle billing inquiries, refunds, and account-related issues",
        backstory="""You are a billing specialist with expertise in payment
                   processing, refunds, and account management. You excel at
                   resolving billing issues quickly and maintaining customer
                   satisfaction.""",
        verbose=True,
        allow_delegation=False
    )
    
    # Escalation Agent
    escalation_agent = Agent(
        role="Senior Support Escalation Specialist",
        goal="Handle complex, high-priority, or escalated customer issues",
        backstory="""You are a senior support specialist who handles the most
                   complex customer issues. You have authority to make decisions,
                   create tickets, and coordinate with other departments to
                   resolve critical issues.""",
        verbose=True,
        allow_delegation=False,
        tools=[create_support_ticket, search_knowledge_base]
    )
    
    # Sample customer query
    customer_query = "I'm getting an error when trying to process my payment. The system says my card is invalid but it works everywhere else. I need this resolved urgently!"
    
    # Define tasks
    classification_task = Task(
        description=f"""Classify the following customer query and determine routing:
                     
                     Customer Query: {customer_query}
                     
                     Classify the query and determine:
                     - Category (Technical, Billing, How-To, Feature Request, General)
                     - Priority level
                     - Whether escalation is needed
                     - Recommended specialist""",
        expected_output="Classification result with category, priority, and routing recommendation",
        agent=classifier
    )
    
    technical_task = Task(
        description=f"""Handle the technical aspect of this customer query:
                     
                     Query: {customer_query}
                     
                     Provide:
                     - Troubleshooting steps
                     - Potential solutions
                     - Knowledge base references
                     - Next steps if issue persists""",
        expected_output="Technical support response with solutions and next steps",
        agent=technical_specialist,
        context=[classification_task]
    )
    
    billing_task = Task(
        description=f"""Handle the billing aspect of this customer query:
                     
                     Query: {customer_query}
                     
                     Address:
                     - Payment processing issues
                     - Card validation problems
                     - Account verification needs
                     - Refund or credit options if applicable""",
        expected_output="Billing support response addressing payment issues",
        agent=billing_specialist,
        context=[classification_task]
    )
    
    escalation_task = Task(
        description=f"""Handle escalation for this high-priority customer query:
                     
                     Query: {customer_query}
                     
                     If escalation is needed:
                     - Create support ticket
                     - Coordinate with relevant departments
                     - Provide customer with escalation details
                     - Ensure timely resolution""",
        expected_output="Escalation response with ticket information and resolution plan",
        agent=escalation_agent,
        context=[classification_task]
    )
    
    # Create crew
    crew = Crew(
        agents=[classifier, technical_specialist, billing_specialist, escalation_agent],
        tasks=[classification_task, technical_task, billing_task, escalation_task],
        process=Process.hierarchical,  # Classifier acts as manager
        verbose=True
    )
    
    # Execute crew
    result = crew.kickoff()
    return result


# ============================================================================
# EXAMPLE 4: CREW WITH MEMORY
# ============================================================================

def crew_with_memory():
    """
    Demonstrates a crew with memory enabled for learning across runs.
    The crew remembers previous interactions and can build upon them.
    
    Returns:
        Crew execution result
    """
    researcher = Agent(
        role="Research Assistant",
        goal="Conduct research and remember findings for future reference",
        backstory="""You are a research assistant who maintains detailed records
                   of research findings. You excel at building upon previous
                   research and avoiding redundant work.""",
        verbose=True,
        memory=True  # Enable memory for this agent
    )
    
    writer = Agent(
        role="Content Writer",
        goal="Create content based on research, building upon previous work",
        backstory="""You are a writer who references previous work and maintains
                   consistency across multiple content pieces. You remember what
                   has been written before.""",
        verbose=True,
        memory=True  # Enable memory for this agent
    )
    
    # First research task
    initial_research_task = Task(
        description="""Research the topic of 'Artificial Intelligence in Education'.
                      This is the first research session on this topic.""",
        expected_output="Initial research findings on AI in Education",
        agent=researcher
    )
    
    # Follow-up research task (will use memory)
    followup_research_task = Task(
        description="""Continue research on 'Artificial Intelligence in Education',
                      building upon previous findings. Focus on recent developments
                      since the last research session.""",
        expected_output="Updated research findings building on previous work",
        agent=researcher,
        context=[initial_research_task]
    )
    
    # Writing task that references previous work
    writing_task = Task(
        description="""Write an article on 'AI in Education' that references
                      and builds upon previous research and writing on this topic.
                      Ensure consistency with previous content.""",
        expected_output="Article that builds upon previous research and maintains consistency",
        agent=writer,
        context=[followup_research_task]
    )
    
    # Create crew with memory enabled
    crew = Crew(
        agents=[researcher, writer],
        tasks=[initial_research_task, followup_research_task, writing_task],
        process=Process.sequential,
        memory=True,  # Enable crew-level memory
        verbose=True
    )
    
    # Execute crew
    result = crew.kickoff()
    return result


# ============================================================================
# EXAMPLE 5: ASYNC TASKS FOR PARALLEL EXECUTION
# ============================================================================

def crew_with_async_tasks():
    """
    Demonstrates async tasks that can execute in parallel when possible.
    Multiple independent research tasks run concurrently.
    
    Returns:
        Crew execution result
    """
    researcher = Agent(
        role="Research Specialist",
        goal="Conduct independent research on assigned topics",
        backstory="""You are a research specialist who can work independently
                   on multiple topics simultaneously. You excel at parallel
                   research tasks.""",
        verbose=True,
        tools=[search_knowledge_base]
    )
    
    synthesizer = Agent(
        role="Information Synthesizer",
        goal="Synthesize information from multiple research sources",
        backstory="""You are an expert at combining information from multiple
                   sources into coherent insights. You excel at identifying
                   connections and patterns across different research areas.""",
        verbose=True
    )
    
    # Independent research tasks (can run in parallel)
    research_task_1 = Task(
        description="Research 'Machine Learning Algorithms'",
        expected_output="Research findings on ML algorithms",
        agent=researcher,
        async_execution=True  # Enable async execution
    )
    
    research_task_2 = Task(
        description="Research 'Deep Learning Applications'",
        expected_output="Research findings on deep learning applications",
        agent=researcher,
        async_execution=True  # Enable async execution
    )
    
    research_task_3 = Task(
        description="Research 'Neural Network Architectures'",
        expected_output="Research findings on neural network architectures",
        agent=researcher,
        async_execution=True  # Enable async execution
    )
    
    # Synthesis task (depends on all research tasks)
    synthesis_task = Task(
        description="""Synthesize findings from all three research tasks.
                      Identify common themes, connections, and create a
                      comprehensive overview.""",
        expected_output="Synthesized report combining all research findings",
        agent=synthesizer,
        context=[research_task_1, research_task_2, research_task_3]
    )
    
    # Create crew
    crew = Crew(
        agents=[researcher, synthesizer],
        tasks=[research_task_1, research_task_2, research_task_3, synthesis_task],
        process=Process.sequential,  # Process handles async tasks automatically
        verbose=True
    )
    
    # Execute crew
    result = crew.kickoff()
    return result


# ============================================================================
# EXAMPLE 6: CALLBACKS AND EVENT HANDLING
# ============================================================================

def task_start_callback(task: Task):
    """Callback function called when a task starts"""
    print(f"[CALLBACK] Task started: {task.description[:50]}...")
    print(f"[CALLBACK] Assigned agent: {task.agent.role}")
    print(f"[CALLBACK] Timestamp: {datetime.now().isoformat()}")


def task_complete_callback(task: Task, result: str):
    """Callback function called when a task completes"""
    print(f"[CALLBACK] Task completed: {task.description[:50]}...")
    print(f"[CALLBACK] Result length: {len(result)} characters")
    print(f"[CALLBACK] Timestamp: {datetime.now().isoformat()}")


def crew_start_callback(crew: Crew):
    """Callback function called when crew execution starts"""
    print(f"[CALLBACK] Crew execution started")
    print(f"[CALLBACK] Number of agents: {len(crew.agents)}")
    print(f"[CALLBACK] Number of tasks: {len(crew.tasks)}")
    print(f"[CALLBACK] Process type: {crew.process}")
    print(f"[CALLBACK] Timestamp: {datetime.now().isoformat()}")


def crew_complete_callback(crew: Crew, result: str):
    """Callback function called when crew execution completes"""
    print(f"[CALLBACK] Crew execution completed")
    print(f"[CALLBACK] Final result length: {len(str(result))} characters")
    print(f"[CALLBACK] Timestamp: {datetime.now().isoformat()}")


def crew_with_callbacks():
    """
    Demonstrates crew execution with callback functions for monitoring.
    Callbacks are triggered at key execution points.
    
    Returns:
        Crew execution result
    """
    researcher = Agent(
        role="Research Analyst",
        goal="Conduct research with monitoring",
        backstory="""You are a research analyst who works efficiently
                   and provides detailed progress updates.""",
        verbose=True,
        tools=[search_knowledge_base]
    )
    
    writer = Agent(
        role="Content Writer",
        goal="Write content with progress tracking",
        backstory="""You are a writer who provides clear progress
                   updates during content creation.""",
        verbose=True
    )
    
    research_task = Task(
        description="Research 'Sustainable Energy Solutions'",
        expected_output="Research findings on sustainable energy",
        agent=researcher
    )
    
    writing_task = Task(
        description="Write article based on research",
        expected_output="Complete article",
        agent=writer,
        context=[research_task]
    )
    
    # Note: CrewAI callback system may vary by version
    # This demonstrates the concept - actual implementation may differ
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute with monitoring
    print("Starting crew execution with callbacks...")
    print("=" * 60)
    
    # Call callbacks manually (in actual implementation, these would be automatic)
    crew_start_callback(crew)
    
    # Execute crew
    result = crew.kickoff()
    
    # Call completion callback
    crew_complete_callback(crew, str(result))
    
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to demonstrate advanced CrewAI workflows.
    Uncomment the example you want to run.
    """
    print("CrewAI Advanced Workflows")
    print("=" * 60)
    
    # Example 1: Research and Content Pipeline
    # print("\n1. Research and Content Pipeline (4 Agents)")
    # print("-" * 60)
    # result1 = research_and_content_pipeline()
    # print(f"\nResult: {result1}")
    
    # Example 2: Code Review Crew
    # print("\n2. Code Review Crew (3 Specialized Reviewers)")
    # print("-" * 60)
    # result2 = code_review_crew()
    # print(f"\nResult: {result2}")
    
    # Example 3: Customer Support Crew
    # print("\n3. Customer Support Crew (Classification & Escalation)")
    # print("-" * 60)
    # result3 = customer_support_crew()
    # print(f"\nResult: {result3}")
    
    # Example 4: Crew with Memory
    # print("\n4. Crew with Memory (Learning Across Runs)")
    # print("-" * 60)
    # result4 = crew_with_memory()
    # print(f"\nResult: {result4}")
    
    # Example 5: Async Tasks
    # print("\n5. Crew with Async Tasks (Parallel Execution)")
    # print("-" * 60)
    # result5 = crew_with_async_tasks()
    # print(f"\nResult: {result5}")
    
    # Example 6: Callbacks and Event Handling
    # print("\n6. Crew with Callbacks (Event Monitoring)")
    # print("-" * 60)
    # result6 = crew_with_callbacks()
    # print(f"\nResult: {result6}")
    
    print("\nNote: Uncomment examples in main() to run them.")
    print("Make sure to configure your LLM API keys before running.")
    print("\nAdvanced Features:")
    print("- Multi-agent pipelines with 4+ agents")
    print("- Specialized review crews")
    print("- Customer support with classification")
    print("- Memory-enabled crews")
    print("- Async/parallel task execution")
    print("- Callback and event monitoring")


if __name__ == "__main__":
    main()
