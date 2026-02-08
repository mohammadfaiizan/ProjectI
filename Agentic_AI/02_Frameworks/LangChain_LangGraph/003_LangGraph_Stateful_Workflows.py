"""
Comprehensive LangGraph Stateful Workflows Implementation

This module demonstrates various patterns for building stateful workflows
using LangGraph, including basic graphs, ReAct agents, multi-step pipelines,
human-in-the-loop workflows, multi-agent systems, and parallel execution.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Literal
from typing_extensions import NotRequired
import operator
from functools import reduce

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================================
# 1. BASIC STATEGRAPH
# ============================================================================

class BasicState(TypedDict):
    """State schema for basic StateGraph example."""
    counter: int
    messages: Annotated[List[BaseMessage], add_messages]
    result: NotRequired[str]


def Basic_StateGraph_Example():
    """
    Demonstrates a basic StateGraph with state schema, nodes, and edges.
    
    This example shows:
    - Defining state using TypedDict
    - Adding nodes to the graph
    - Connecting nodes with edges
    - Compiling and running the graph
    """
    def increment_node(state: BasicState) -> BasicState:
        """Node that increments the counter."""
        return {
            "counter": state["counter"] + 1,
            "messages": state["messages"],
        }
    
    def process_node(state: BasicState) -> BasicState:
        """Node that processes the counter value."""
        counter_value = state["counter"]
        result_message = f"Counter value is: {counter_value}"
        return {
            "counter": counter_value,
            "messages": state["messages"] + [HumanMessage(content=result_message)],
            "result": result_message,
        }
    
    def finalize_node(state: BasicState) -> BasicState:
        """Final node that adds a completion message."""
        return {
            "counter": state["counter"],
            "messages": state["messages"] + [AIMessage(content="Processing complete!")],
            "result": state.get("result", ""),
        }
    
    # Build the graph
    workflow = StateGraph(BasicState)
    
    # Add nodes
    workflow.add_node("increment", increment_node)
    workflow.add_node("process", process_node)
    workflow.add_node("finalize", finalize_node)
    
    # Add edges
    workflow.set_entry_point("increment")
    workflow.add_edge("increment", "process")
    workflow.add_edge("process", "finalize")
    workflow.add_edge("finalize", END)
    
    # Compile the graph
    app = workflow.compile()
    
    # Run the graph
    initial_state = {
        "counter": 0,
        "messages": [HumanMessage(content="Starting workflow")],
    }
    
    result = app.invoke(initial_state)
    print("Basic StateGraph Result:")
    print(f"Final counter: {result['counter']}")
    print(f"Result: {result.get('result', 'N/A')}")
    print(f"Messages: {len(result['messages'])} messages")
    
    return result


# ============================================================================
# 2. REACT AGENT WITH LANGGRAPH
# ============================================================================

class ReActState(TypedDict):
    """State schema for ReAct agent."""
    messages: Annotated[List[BaseMessage], add_messages]


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


def ReAct_Agent_Example():
    """
    Implements a ReAct (Reasoning + Acting) agent using LangGraph.
    
    This example demonstrates:
    - State with messages
    - LLM node that decides tool calls
    - Tool execution node
    - Conditional routing (continue vs end)
    - Full working ReAct loop
    """
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Bind tools to LLM
    tools = [calculate, get_weather]
    llm_with_tools = llm.bind_tools(tools)
    
    def llm_node(state: ReActState) -> ReActState:
        """Node that calls LLM and gets response with potential tool calls."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def tool_node(state: ReActState) -> ReActState:
        """Node that executes tool calls."""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_node_instance = ToolNode(tools)
            return tool_node_instance.invoke(state)
        
        return state
    
    def should_continue(state: ReActState) -> Literal["tools", "end"]:
        """Conditional routing function."""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"
    
    # Build the graph
    workflow = StateGraph(ReActState)
    
    # Add nodes
    workflow.add_node("agent", llm_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edge
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    # Run the graph
    initial_state = {
        "messages": [
            HumanMessage(content="What is 15 * 23? Also, what's the weather in New York?")
        ],
    }
    
    result = app.invoke(initial_state)
    print("\nReAct Agent Result:")
    for message in result["messages"]:
        print(f"{message.__class__.__name__}: {message.content[:200]}")
    
    return result


# ============================================================================
# 3. MULTI-STEP WORKFLOW
# ============================================================================

class WorkflowState(TypedDict):
    """State schema for multi-step workflow."""
    topic: str
    research: NotRequired[str]
    draft: NotRequired[str]
    review_feedback: NotRequired[str]
    final_version: NotRequired[str]
    revision_count: int
    passed_review: NotRequired[bool]


def Multi_Step_Workflow_Example():
    """
    Demonstrates a multi-step workflow: Research -> Write -> Review -> Revise.
    
    This example shows:
    - Sequential pipeline with multiple stages
    - Conditional edge for review pass/fail
    - State tracking across nodes
    - Iterative revision process
    """
    def research_node(state: WorkflowState) -> WorkflowState:
        """Research phase: gather information about the topic."""
        topic = state["topic"]
        research_content = f"""
        Research Summary for: {topic}
        
        Key Points:
        1. {topic} is an important subject in modern technology
        2. It involves multiple components and considerations
        3. Best practices include careful planning and execution
        4. Common challenges include scalability and maintainability
        
        Sources consulted: Academic papers, industry reports, expert opinions
        """
        return {
            **state,
            "research": research_content.strip(),
        }
    
    def write_node(state: WorkflowState) -> WorkflowState:
        """Write phase: create initial draft based on research."""
        research = state.get("research", "")
        topic = state["topic"]
        
        draft_content = f"""
        Draft Article: {topic}
        
        Introduction:
        This article explores {topic} and its implications.
        
        Main Content:
        {research}
        
        Conclusion:
        In summary, {topic} represents a significant area of study.
        """
        return {
            **state,
            "draft": draft_content.strip(),
        }
    
    def review_node(state: WorkflowState) -> WorkflowState:
        """Review phase: evaluate the draft quality."""
        draft = state.get("draft", "")
        revision_count = state.get("revision_count", 0)
        
        # Simulate review: pass if draft is substantial and revision count is low
        word_count = len(draft.split())
        passed = word_count > 100 and revision_count < 3
        
        feedback = (
            "Excellent work! The draft meets all quality standards."
            if passed
            else f"Needs improvement. Current word count: {word_count}. Please revise."
        )
        
        return {
            **state,
            "review_feedback": feedback,
            "passed_review": passed,
        }
    
    def revise_node(state: WorkflowState) -> WorkflowState:
        """Revise phase: improve the draft based on feedback."""
        draft = state.get("draft", "")
        feedback = state.get("review_feedback", "")
        revision_count = state.get("revision_count", 0)
        
        revised_draft = f"""
        {draft}
        
        [REVISED - Iteration {revision_count + 1}]
        
        Additional improvements based on feedback:
        - Enhanced clarity and detail
        - Added more comprehensive examples
        - Improved structure and flow
        - Expanded on key concepts
        
        Feedback addressed: {feedback}
        """
        
        return {
            **state,
            "draft": revised_draft.strip(),
            "revision_count": revision_count + 1,
        }
    
    def finalize_node(state: WorkflowState) -> WorkflowState:
        """Finalize phase: create final version."""
        draft = state.get("draft", "")
        final_version = f"""
        FINAL VERSION
        
        {draft}
        
        ---
        Approved and finalized after {state.get('revision_count', 0)} revision(s)
        """
        
        return {
            **state,
            "final_version": final_version.strip(),
        }
    
    def should_revise(state: WorkflowState) -> Literal["revise", "finalize"]:
        """Conditional routing based on review result."""
        if state.get("passed_review", False):
            return "finalize"
        return "revise"
    
    # Build the graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("write", write_node)
    workflow.add_node("review", review_node)
    workflow.add_node("revise", revise_node)
    workflow.add_node("finalize", finalize_node)
    
    # Set entry point
    workflow.set_entry_point("research")
    
    # Add edges
    workflow.add_edge("research", "write")
    workflow.add_edge("write", "review")
    
    # Conditional edge from review
    workflow.add_conditional_edges(
        "review",
        should_revise,
        {
            "revise": "revise",
            "finalize": "finalize",
        },
    )
    
    # Edge from revise back to review
    workflow.add_edge("revise", "review")
    
    # Final edge
    workflow.add_edge("finalize", END)
    
    # Compile the graph
    app = workflow.compile()
    
    # Run the graph
    initial_state = {
        "topic": "Machine Learning Applications",
        "revision_count": 0,
    }
    
    result = app.invoke(initial_state)
    print("\nMulti-Step Workflow Result:")
    print(f"Topic: {result['topic']}")
    print(f"Revision Count: {result['revision_count']}")
    print(f"Passed Review: {result.get('passed_review', False)}")
    print(f"Final Version Length: {len(result.get('final_version', ''))} characters")
    
    return result


# ============================================================================
# 4. HUMAN-IN-THE-LOOP
# ============================================================================

class HumanApprovalState(TypedDict):
    """State schema for human-in-the-loop workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    approval_status: NotRequired[str]
    task_description: str


def Human_In_The_Loop_Example():
    """
    Demonstrates human-in-the-loop workflow with checkpointing.
    
    This example shows:
    - Interrupt node for human approval
    - Checkpointing with MemorySaver
    - Resume from checkpoint
    - State persistence across interruptions
    """
    # Create memory saver for checkpointing
    memory = MemorySaver()
    
    def generate_proposal_node(state: HumanApprovalState) -> HumanApprovalState:
        """Generate a proposal that needs approval."""
        task = state["task_description"]
        proposal = f"""
        Proposal for: {task}
        
        Plan:
        1. Initial analysis and requirements gathering
        2. Design phase with stakeholder input
        3. Implementation with iterative testing
        4. Deployment and monitoring
        
        Estimated timeline: 4-6 weeks
        Budget: $50,000 - $75,000
        """
        
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=f"Generated proposal:\n{proposal}")
            ],
        }
    
    def human_approval_node(state: HumanApprovalState) -> HumanApprovalState:
        """Node that waits for human approval (interrupt point)."""
        # In a real implementation, this would wait for human input
        # For demonstration, we simulate approval
        approval_status = "approved"  # Would come from human input
        
        return {
            **state,
            "approval_status": approval_status,
            "messages": state["messages"] + [
                HumanMessage(content=f"Human approval: {approval_status}")
            ],
        }
    
    def execute_node(state: HumanApprovalState) -> HumanApprovalState:
        """Execute the approved proposal."""
        status = state.get("approval_status", "pending")
        
        if status == "approved":
            execution_result = "Task execution started successfully."
        else:
            execution_result = "Execution pending approval."
        
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=execution_result)
            ],
        }
    
    def check_approval(state: HumanApprovalState) -> Literal["approval", "execute"]:
        """Conditional routing based on approval status."""
        if state.get("approval_status"):
            return "execute"
        return "approval"
    
    # Build the graph
    workflow = StateGraph(HumanApprovalState)
    
    # Add nodes
    workflow.add_node("generate", generate_proposal_node)
    workflow.add_node("approval", human_approval_node)
    workflow.add_node("execute", execute_node)
    
    # Set entry point
    workflow.set_entry_point("generate")
    
    # Add edges
    workflow.add_edge("generate", "approval")
    
    # Conditional edge from approval
    workflow.add_conditional_edges(
        "approval",
        check_approval,
        {
            "approval": "approval",  # Loop back if not approved
            "execute": "execute",
        },
    )
    
    workflow.add_edge("execute", END)
    
    # Compile with checkpointing
    app = workflow.compile(checkpointer=memory)
    
    # Configuration for interrupts
    config = {"configurable": {"thread_id": "human-approval-1"}}
    
    # Run the graph (first part)
    initial_state = {
        "messages": [],
        "task_description": "Build a new customer portal",
    }
    
    # Invoke up to approval point
    result = app.invoke(initial_state, config)
    print("\nHuman-in-the-Loop Result:")
    print(f"Approval Status: {result.get('approval_status', 'Pending')}")
    print(f"Messages: {len(result['messages'])} messages")
    
    # Simulate resuming from checkpoint
    # In real scenario, human would provide input here
    resume_result = app.invoke(
        {"approval_status": "approved"},
        config,
    )
    
    print(f"After Resume - Messages: {len(resume_result['messages'])} messages")
    
    return resume_result


# ============================================================================
# 5. MULTI-AGENT GRAPH
# ============================================================================

class MultiAgentState(TypedDict):
    """State schema for multi-agent system."""
    messages: Annotated[List[BaseMessage], add_messages]
    current_agent: NotRequired[str]
    task_type: str


def Multi_Agent_Graph_Example():
    """
    Demonstrates a multi-agent system with supervisor routing.
    
    This example shows:
    - Supervisor agent that routes to specialists
    - Multiple specialist nodes (coder, writer, analyst)
    - Shared state between agents
    - Dynamic routing based on task type
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def supervisor_node(state: MultiAgentState) -> MultiAgentState:
        """Supervisor node that routes tasks to appropriate specialists."""
        messages = state["messages"]
        task_type = state.get("task_type", "general")
        
        # Determine which agent should handle this
        routing_map = {
            "code": "coder",
            "write": "writer",
            "analyze": "analyst",
        }
        
        agent = routing_map.get(task_type, "writer")
        
        supervisor_message = f"Routing task to {agent} agent."
        
        return {
            **state,
            "current_agent": agent,
            "messages": messages + [AIMessage(content=supervisor_message)],
        }
    
    def coder_agent_node(state: MultiAgentState) -> MultiAgentState:
        """Specialist agent for coding tasks."""
        messages = state["messages"]
        last_human_message = next(
            (msg for msg in reversed(messages) if isinstance(msg, HumanMessage)),
            None
        )
        
        if last_human_message:
            task = last_human_message.content
            code_response = f"""
            Code Solution:
            
            def solve_task():
                # Implementation for: {task}
                result = "Task completed"
                return result
            
            # Usage
            solution = solve_task()
            print(solution)
            """
            
            return {
                **state,
                "messages": messages + [AIMessage(content=code_response)],
            }
        
        return state
    
    def writer_agent_node(state: MultiAgentState) -> MultiAgentState:
        """Specialist agent for writing tasks."""
        messages = state["messages"]
        last_human_message = next(
            (msg for msg in reversed(messages) if isinstance(msg, HumanMessage)),
            None
        )
        
        if last_human_message:
            task = last_human_message.content
            written_content = f"""
            Written Content:
            
            Title: {task}
            
            Introduction:
            This article addresses the topic of {task} comprehensively.
            
            Main Body:
            {task} is an important subject that requires careful consideration.
            We explore various aspects and provide detailed insights.
            
            Conclusion:
            In summary, {task} represents a significant area worthy of attention.
            """
            
            return {
                **state,
                "messages": messages + [AIMessage(content=written_content)],
            }
        
        return state
    
    def analyst_agent_node(state: MultiAgentState) -> MultiAgentState:
        """Specialist agent for analysis tasks."""
        messages = state["messages"]
        last_human_message = next(
            (msg for msg in reversed(messages) if isinstance(msg, HumanMessage)),
            None
        )
        
        if last_human_message:
            task = last_human_message.content
            analysis = f"""
            Analysis Report:
            
            Task: {task}
            
            Key Findings:
            1. Primary consideration: Data quality and accuracy
            2. Secondary factor: Performance optimization
            3. Risk assessment: Low to moderate
            4. Recommendations: Proceed with caution, implement monitoring
            
            Metrics:
            - Confidence level: 85%
            - Estimated impact: High
            - Resource requirements: Medium
            """
            
            return {
                **state,
                "messages": messages + [AIMessage(content=analysis)],
            }
        
        return state
    
    def route_to_agent(state: MultiAgentState) -> Literal["coder", "writer", "analyst"]:
        """Route to appropriate specialist agent."""
        agent = state.get("current_agent", "writer")
        return agent
    
    # Build the graph
    workflow = StateGraph(MultiAgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("coder", coder_agent_node)
    workflow.add_node("writer", writer_agent_node)
    workflow.add_node("analyst", analyst_agent_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edge from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "coder": "coder",
            "writer": "writer",
            "analyst": "analyst",
        },
    )
    
    # All agents end the workflow
    workflow.add_edge("coder", END)
    workflow.add_edge("writer", END)
    workflow.add_edge("analyst", END)
    
    # Compile the graph
    app = workflow.compile()
    
    # Run the graph with different task types
    test_cases = [
        {"task_type": "code", "messages": [HumanMessage(content="Create a function to sort a list")]},
        {"task_type": "write", "messages": [HumanMessage(content="Write an article about AI")]},
        {"task_type": "analyze", "messages": [HumanMessage(content="Analyze market trends")]},
    ]
    
    print("\nMulti-Agent Graph Results:")
    for i, test_case in enumerate(test_cases, 1):
        result = app.invoke(test_case)
        print(f"\nTest {i} - Task Type: {test_case['task_type']}")
        print(f"Current Agent: {result.get('current_agent', 'N/A')}")
        print(f"Messages: {len(result['messages'])} messages")
    
    return result


# ============================================================================
# 6. PARALLEL EXECUTION
# ============================================================================

class ParallelState(TypedDict):
    """State schema for parallel execution."""
    input_data: str
    branch_a_result: NotRequired[str]
    branch_b_result: NotRequired[str]
    branch_c_result: NotRequired[str]
    combined_result: NotRequired[str]


def Parallel_Execution_Example():
    """
    Demonstrates parallel execution with fan-out and fan-in pattern.
    
    This example shows:
    - Branching paths in graph
    - Fan-out: single input to multiple parallel branches
    - Fan-in: multiple branches converge to single output
    - Independent parallel processing
    """
    def input_node(state: ParallelState) -> ParallelState:
        """Initial node that prepares data for parallel processing."""
        input_data = state["input_data"]
        return {
            **state,
            "input_data": input_data,
        }
    
    def branch_a_node(state: ParallelState) -> ParallelState:
        """First parallel branch: data validation."""
        input_data = state["input_data"]
        result = f"""
        Branch A - Validation Results:
        Input: {input_data}
        Status: Valid
        Checks: Format OK, Length OK, Content OK
        """
        return {
            **state,
            "branch_a_result": result.strip(),
        }
    
    def branch_b_node(state: ParallelState) -> ParallelState:
        """Second parallel branch: data transformation."""
        input_data = state["input_data"]
        transformed = input_data.upper().replace(" ", "_")
        result = f"""
        Branch B - Transformation Results:
        Original: {input_data}
        Transformed: {transformed}
        Method: Uppercase with underscore replacement
        """
        return {
            **state,
            "branch_b_result": result.strip(),
        }
    
    def branch_c_node(state: ParallelState) -> ParallelState:
        """Third parallel branch: data analysis."""
        input_data = state["input_data"]
        word_count = len(input_data.split())
        char_count = len(input_data)
        result = f"""
        Branch C - Analysis Results:
        Input: {input_data}
        Word Count: {word_count}
        Character Count: {char_count}
        Average Word Length: {char_count / word_count if word_count > 0 else 0:.2f}
        """
        return {
            **state,
            "branch_c_result": result.strip(),
        }
    
    def combine_node(state: ParallelState) -> ParallelState:
        """Fan-in node: combine results from all parallel branches."""
        branch_a = state.get("branch_a_result", "")
        branch_b = state.get("branch_b_result", "")
        branch_c = state.get("branch_c_result", "")
        
        combined = f"""
        Combined Results from Parallel Execution:
        
        {branch_a}
        
        {branch_b}
        
        {branch_c}
        
        Summary: All branches completed successfully.
        """
        
        return {
            **state,
            "combined_result": combined.strip(),
        }
    
    # Build the graph
    workflow = StateGraph(ParallelState)
    
    # Add nodes
    workflow.add_node("input", input_node)
    workflow.add_node("branch_a", branch_a_node)
    workflow.add_node("branch_b", branch_b_node)
    workflow.add_node("branch_c", branch_c_node)
    workflow.add_node("combine", combine_node)
    
    # Set entry point
    workflow.set_entry_point("input")
    
    # Fan-out: input goes to all three branches
    workflow.add_edge("input", "branch_a")
    workflow.add_edge("input", "branch_b")
    workflow.add_edge("input", "branch_c")
    
    # Fan-in: all branches converge to combine node
    workflow.add_edge("branch_a", "combine")
    workflow.add_edge("branch_b", "combine")
    workflow.add_edge("branch_c", "combine")
    
    # Final edge
    workflow.add_edge("combine", END)
    
    # Compile the graph
    app = workflow.compile()
    
    # Run the graph
    initial_state = {
        "input_data": "Sample data for parallel processing",
    }
    
    result = app.invoke(initial_state)
    print("\nParallel Execution Result:")
    print(f"Branch A Result Length: {len(result.get('branch_a_result', ''))} chars")
    print(f"Branch B Result Length: {len(result.get('branch_b_result', ''))} chars")
    print(f"Branch C Result Length: {len(result.get('branch_c_result', ''))} chars")
    print(f"Combined Result Length: {len(result.get('combined_result', ''))} chars")
    
    return result


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run all LangGraph workflow examples.
    
    Each example demonstrates different patterns and capabilities:
    1. Basic StateGraph - Simple state management
    2. ReAct Agent - Tool-using agent with reasoning
    3. Multi-Step Workflow - Sequential pipeline with conditionals
    4. Human-in-the-Loop - Checkpointing and interruptions
    5. Multi-Agent Graph - Supervisor routing to specialists
    6. Parallel Execution - Fan-out/fan-in pattern
    """
    print("=" * 80)
    print("LangGraph Stateful Workflows - Comprehensive Examples")
    print("=" * 80)
    
    try:
        # Example 1: Basic StateGraph
        print("\n" + "=" * 80)
        print("Example 1: Basic StateGraph")
        print("=" * 80)
        Basic_StateGraph_Example()
        
        # Example 2: ReAct Agent
        print("\n" + "=" * 80)
        print("Example 2: ReAct Agent")
        print("=" * 80)
        ReAct_Agent_Example()
        
        # Example 3: Multi-Step Workflow
        print("\n" + "=" * 80)
        print("Example 3: Multi-Step Workflow")
        print("=" * 80)
        Multi_Step_Workflow_Example()
        
        # Example 4: Human-in-the-Loop
        print("\n" + "=" * 80)
        print("Example 4: Human-in-the-Loop")
        print("=" * 80)
        Human_In_The_Loop_Example()
        
        # Example 5: Multi-Agent Graph
        print("\n" + "=" * 80)
        print("Example 5: Multi-Agent Graph")
        print("=" * 80)
        Multi_Agent_Graph_Example()
        
        # Example 6: Parallel Execution
        print("\n" + "=" * 80)
        print("Example 6: Parallel Execution")
        print("=" * 80)
        Parallel_Execution_Example()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        print("Note: Some examples may require API keys (OpenAI, Anthropic)")
        print("Please configure your environment variables accordingly.")
        raise


if __name__ == "__main__":
    main()
