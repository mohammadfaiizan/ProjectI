"""
Agent module for Multi-Agent Task Solver.
Implements the multi-agent graph using LangGraph with supervisor pattern.
"""

from typing import TypedDict, List, Dict, Any, Literal, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .Config import LLM_Config, Agent_Config, Routing_Config
from .Tools import (
    Search_Information,
    Write_Python_Code,
    Write_Content,
    Analyze_Data,
    Review_Output,
    Task_Parser,
    Result_Aggregator
)


class Solver_State(TypedDict):
    """State dictionary for the multi-agent solver."""
    task: str
    subtasks: List[Dict[str, Any]]
    current_agent: str
    agent_outputs: Dict[str, str]
    final_result: str
    messages: Annotated[List[Any], add_messages]
    iteration: int
    completed_subtasks: List[int]


class Multi_Agent_Graph:
    """Multi-agent graph implementing supervisor pattern for task solving."""
    
    def __init__(
        self,
        llm_config: LLM_Config,
        agent_config: Agent_Config,
        routing_config: Routing_Config
    ):
        """
        Initialize multi-agent graph.
        
        Args:
            llm_config: LLM configuration
            agent_config: Agent configuration
            routing_config: Routing configuration
        """
        self.llm_config = llm_config
        self.agent_config = agent_config
        self.routing_config = routing_config
        
        self.task_parser = Task_Parser(llm_config.get_supervisor_llm())
        self.result_aggregator = Result_Aggregator(llm_config.get_aggregator_llm())
        
        self.graph = None
        self.memory = MemorySaver()
    
    def Supervisor_Node(self, state: Solver_State) -> Dict[str, Any]:
        """
        Supervisor node that decomposes tasks and routes to specialists.
        
        Args:
            state: Current solver state
            
        Returns:
            Updated state with routing decision
        """
        if state["iteration"] >= self.agent_config.max_iterations:
            return {
                "current_agent": "aggregator",
                "messages": state["messages"]
            }
        
        if not state.get("subtasks"):
            subtasks = self.task_parser.parse_task(state["task"])
            return {
                "subtasks": subtasks,
                "current_agent": self._select_next_specialist(subtasks, state),
                "messages": state["messages"]
            }
        
        completed = state.get("completed_subtasks", [])
        remaining_subtasks = [
            st for st in state["subtasks"]
            if st["order"] not in completed
        ]
        
        if not remaining_subtasks:
            return {
                "current_agent": "aggregator",
                "messages": state["messages"]
            }
        
        next_specialist = self._select_next_specialist(remaining_subtasks, state)
        
        return {
            "current_agent": next_specialist,
            "messages": state["messages"]
        }
    
    def _select_next_specialist(
        self,
        subtasks: List[Dict[str, Any]],
        state: Solver_State
    ) -> str:
        """
        Select the next specialist to handle based on subtasks.
        
        Args:
            subtasks: List of remaining subtasks
            state: Current state
            
        Returns:
            Name of the specialist to route to
        """
        if not subtasks:
            return "aggregator"
        
        next_subtask = min(subtasks, key=lambda x: x["order"])
        return next_subtask["specialist"]
    
    def Research_Node(self, state: Solver_State) -> Dict[str, Any]:
        """
        Research specialist node for information gathering tasks.
        
        Args:
            state: Current solver state
            
        Returns:
            Updated state with research results
        """
        llm = self.llm_config.get_research_llm()
        tools = [Search_Information]
        tool_node = ToolNode(tools)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research specialist. Your task is to gather 
            comprehensive information on the given topic. Use the Search_Information 
            tool to find relevant information."""),
            ("human", "Research Task: {task}\n\nCurrent Subtask: {subtask}")
        ])
        
        completed = state.get("completed_subtasks", [])
        current_subtask = next(
            (st for st in state["subtasks"] if st["order"] not in completed),
            None
        )
        
        if not current_subtask:
            return {"messages": state["messages"]}
        
        chain = prompt | llm.bind_tools(tools)
        response = chain.invoke({
            "task": state["task"],
            "subtask": current_subtask["description"]
        })
        
        messages = state["messages"] + [response]
        
        if response.tool_calls:
            tool_results = tool_node.invoke({"messages": messages})
            messages.extend(tool_results["messages"])
            
            final_response = llm.invoke(messages)
            messages.append(final_response)
            
            agent_output = final_response.content
            agent_outputs = state.get("agent_outputs", {})
            agent_outputs["research"] = agent_outputs.get("research", "") + "\n\n" + agent_output if "research" in agent_outputs else agent_output
            
            completed.append(current_subtask["order"])
            
            return {
                "agent_outputs": agent_outputs,
                "completed_subtasks": completed,
                "current_agent": "supervisor",
                "messages": messages,
                "iteration": state["iteration"] + 1
            }
        
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["research"] = response.content
        completed.append(current_subtask["order"])
        
        return {
            "agent_outputs": agent_outputs,
            "completed_subtasks": completed,
            "current_agent": "supervisor",
            "messages": messages,
            "iteration": state["iteration"] + 1
        }
    
    def Coding_Node(self, state: Solver_State) -> Dict[str, Any]:
        """
        Coding specialist node for code generation tasks.
        
        Args:
            state: Current solver state
            
        Returns:
            Updated state with code generation results
        """
        llm = self.llm_config.get_coding_llm()
        tools = [Write_Python_Code]
        tool_node = ToolNode(tools)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a coding specialist. Your task is to generate 
            high-quality Python code based on requirements. Use the Write_Python_Code 
            tool to create code implementations."""),
            ("human", "Coding Task: {task}\n\nRequirements: {subtask}")
        ])
        
        completed = state.get("completed_subtasks", [])
        current_subtask = next(
            (st for st in state["subtasks"] if st["order"] not in completed),
            None
        )
        
        if not current_subtask:
            return {"messages": state["messages"]}
        
        chain = prompt | llm.bind_tools(tools)
        response = chain.invoke({
            "task": state["task"],
            "subtask": current_subtask["description"]
        })
        
        messages = state["messages"] + [response]
        
        if response.tool_calls:
            tool_results = tool_node.invoke({"messages": messages})
            messages.extend(tool_results["messages"])
            
            final_response = llm.invoke(messages)
            messages.append(final_response)
            
            agent_output = final_response.content
            agent_outputs = state.get("agent_outputs", {})
            agent_outputs["coding"] = agent_outputs.get("coding", "") + "\n\n" + agent_output if "coding" in agent_outputs else agent_output
            
            completed.append(current_subtask["order"])
            
            return {
                "agent_outputs": agent_outputs,
                "completed_subtasks": completed,
                "current_agent": "supervisor",
                "messages": messages,
                "iteration": state["iteration"] + 1
            }
        
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["coding"] = response.content
        completed.append(current_subtask["order"])
        
        return {
            "agent_outputs": agent_outputs,
            "completed_subtasks": completed,
            "current_agent": "supervisor",
            "messages": messages,
            "iteration": state["iteration"] + 1
        }
    
    def Writing_Node(self, state: Solver_State) -> Dict[str, Any]:
        """
        Writing specialist node for content creation tasks.
        
        Args:
            state: Current solver state
            
        Returns:
            Updated state with writing results
        """
        llm = self.llm_config.get_writing_llm()
        tools = [Write_Content]
        tool_node = ToolNode(tools)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a writing specialist. Your task is to create 
            high-quality written content based on the brief. Use the Write_Content 
            tool to generate content."""),
            ("human", "Writing Task: {task}\n\nBrief: {subtask}")
        ])
        
        completed = state.get("completed_subtasks", [])
        current_subtask = next(
            (st for st in state["subtasks"] if st["order"] not in completed),
            None
        )
        
        if not current_subtask:
            return {"messages": state["messages"]}
        
        chain = prompt | llm.bind_tools(tools)
        response = chain.invoke({
            "task": state["task"],
            "subtask": current_subtask["description"]
        })
        
        messages = state["messages"] + [response]
        
        if response.tool_calls:
            tool_results = tool_node.invoke({"messages": messages})
            messages.extend(tool_results["messages"])
            
            final_response = llm.invoke(messages)
            messages.append(final_response)
            
            agent_output = final_response.content
            agent_outputs = state.get("agent_outputs", {})
            agent_outputs["writing"] = agent_outputs.get("writing", "") + "\n\n" + agent_output if "writing" in agent_outputs else agent_output
            
            completed.append(current_subtask["order"])
            
            return {
                "agent_outputs": agent_outputs,
                "completed_subtasks": completed,
                "current_agent": "supervisor",
                "messages": messages,
                "iteration": state["iteration"] + 1
            }
        
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["writing"] = response.content
        completed.append(current_subtask["order"])
        
        return {
            "agent_outputs": agent_outputs,
            "completed_subtasks": completed,
            "current_agent": "supervisor",
            "messages": messages,
            "iteration": state["iteration"] + 1
        }
    
    def Analysis_Node(self, state: Solver_State) -> Dict[str, Any]:
        """
        Analysis specialist node for data analysis tasks.
        
        Args:
            state: Current solver state
            
        Returns:
            Updated state with analysis results
        """
        llm = self.llm_config.get_analysis_llm()
        tools = [Analyze_Data, Review_Output]
        tool_node = ToolNode(tools)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an analysis specialist. Your task is to analyze 
            data and provide insights. Use the Analyze_Data tool for analysis and 
            Review_Output for quality checks."""),
            ("human", "Analysis Task: {task}\n\nQuestion: {subtask}")
        ])
        
        completed = state.get("completed_subtasks", [])
        current_subtask = next(
            (st for st in state["subtasks"] if st["order"] not in completed),
            None
        )
        
        if not current_subtask:
            return {"messages": state["messages"]}
        
        chain = prompt | llm.bind_tools(tools)
        response = chain.invoke({
            "task": state["task"],
            "subtask": current_subtask["description"]
        })
        
        messages = state["messages"] + [response]
        
        if response.tool_calls:
            tool_results = tool_node.invoke({"messages": messages})
            messages.extend(tool_results["messages"])
            
            final_response = llm.invoke(messages)
            messages.append(final_response)
            
            agent_output = final_response.content
            agent_outputs = state.get("agent_outputs", {})
            agent_outputs["analysis"] = agent_outputs.get("analysis", "") + "\n\n" + agent_output if "analysis" in agent_outputs else agent_output
            
            completed.append(current_subtask["order"])
            
            return {
                "agent_outputs": agent_outputs,
                "completed_subtasks": completed,
                "current_agent": "supervisor",
                "messages": messages,
                "iteration": state["iteration"] + 1
            }
        
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs["analysis"] = response.content
        completed.append(current_subtask["order"])
        
        return {
            "agent_outputs": agent_outputs,
            "completed_subtasks": completed,
            "current_agent": "supervisor",
            "messages": messages,
            "iteration": state["iteration"] + 1
        }
    
    def Aggregator_Node(self, state: Solver_State) -> Dict[str, Any]:
        """
        Aggregator node that combines all specialist outputs.
        
        Args:
            state: Current solver state
            
        Returns:
            Updated state with final aggregated result
        """
        agent_outputs = state.get("agent_outputs", {})
        
        if not agent_outputs:
            return {
                "final_result": "No agent outputs to aggregate.",
                "current_agent": END
            }
        
        final_result = self.result_aggregator.aggregate(
            state["task"],
            agent_outputs
        )
        
        return {
            "final_result": final_result,
            "current_agent": END
        }
    
    def Route_To_Specialist(self, state: Solver_State) -> Literal["research", "coding", "writing", "analysis", "aggregator", "supervisor"]:
        """
        Route function to determine next node based on current agent.
        
        Args:
            state: Current solver state
            
        Returns:
            Name of the next node to execute
        """
        current_agent = state.get("current_agent", "supervisor")
        
        if current_agent == "research":
            return "research"
        elif current_agent == "coding":
            return "coding"
        elif current_agent == "writing":
            return "writing"
        elif current_agent == "analysis":
            return "analysis"
        elif current_agent == "aggregator":
            return "aggregator"
        else:
            return "supervisor"
    
    def Build_Graph(self) -> StateGraph:
        """
        Build the LangGraph state graph with supervisor pattern.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        workflow = StateGraph(Solver_State)
        
        workflow.add_node("supervisor", self.Supervisor_Node)
        workflow.add_node("research", self.Research_Node)
        workflow.add_node("coding", self.Coding_Node)
        workflow.add_node("writing", self.Writing_Node)
        workflow.add_node("analysis", self.Analysis_Node)
        workflow.add_node("aggregator", self.Aggregator_Node)
        
        workflow.set_entry_point("supervisor")
        
        workflow.add_conditional_edges(
            "supervisor",
            self.Route_To_Specialist,
            {
                "research": "research",
                "coding": "coding",
                "writing": "writing",
                "analysis": "analysis",
                "aggregator": "aggregator",
                "supervisor": "supervisor"
            }
        )
        
        workflow.add_edge("research", "supervisor")
        workflow.add_edge("coding", "supervisor")
        workflow.add_edge("writing", "supervisor")
        workflow.add_edge("analysis", "supervisor")
        workflow.add_edge("aggregator", END)
        
        self.graph = workflow.compile(checkpointer=self.memory)
        return self.graph
    
    def Solve(self, task: str) -> str:
        """
        Solve a task using the multi-agent system.
        
        Args:
            task: Task description to solve
            
        Returns:
            Final aggregated result
        """
        if self.graph is None:
            self.Build_Graph()
        
        initial_state = {
            "task": task,
            "subtasks": [],
            "current_agent": "supervisor",
            "agent_outputs": {},
            "final_result": "",
            "messages": [],
            "iteration": 0,
            "completed_subtasks": []
        }
        
        config = {"configurable": {"thread_id": "1"}}
        
        final_state = None
        for state in self.graph.stream(initial_state, config):
            final_state = state
        
        if final_state:
            last_node = list(final_state.keys())[-1]
            if last_node == "aggregator":
                return final_state[last_node].get("final_result", "No result generated.")
        
        return "Task solving completed, but no final result was generated."
