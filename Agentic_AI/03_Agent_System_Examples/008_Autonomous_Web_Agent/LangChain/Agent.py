"""
Agent module for Autonomous Web Agent.
Implements a ReAct-style agent using LangGraph for web navigation and data extraction.
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, List, Dict, Any, Optional, Literal
from typing_extensions import Annotated
import operator
import json

from Config import LLM_Config, Browser_Config, Agent_Config
from Tools import Fetch_Web_Page, Extract_Links, Extract_Tables, Search_Page_Content, Get_Browser, Get_Parser


class Web_State(TypedDict):
    """State dictionary for the web agent."""
    task: str
    current_url: Optional[str]
    visited_urls: List[str]
    extracted_data: List[Dict[str, Any]]
    links: List[Dict[str, str]]
    plan: List[str]
    action_history: List[Dict[str, Any]]
    result: Optional[str]
    iteration: int
    max_iterations: int
    messages: Annotated[List, add_messages]


class Web_Agent_Graph:
    """Web agent graph implementing ReAct loop for autonomous web navigation."""
    
    def __init__(
        self,
        llm_config: LLM_Config,
        browser_config: Browser_Config,
        agent_config: Agent_Config
    ):
        """
        Initialize web agent graph.
        
        Args:
            llm_config: LLM configuration
            browser_config: Browser configuration
            agent_config: Agent configuration
        """
        self.llm_config = llm_config
        self.browser_config = browser_config
        self.agent_config = agent_config
        self.llm = llm_config.Get_LLM()
        self.graph = None
        self._Build_Graph()
    
    def _Build_Graph(self):
        """Build the LangGraph state graph."""
        workflow = StateGraph(Web_State)
        
        # Add nodes
        workflow.add_node("Plan_Action", self._Plan_Action)
        workflow.add_node("Navigate_Page", self._Navigate_Page)
        workflow.add_node("Extract_Data", self._Extract_Data)
        workflow.add_node("Follow_Links", self._Follow_Links)
        workflow.add_node("Evaluate_Progress", self._Evaluate_Progress)
        workflow.add_node("Compile_Result", self._Compile_Result)
        
        # Set entry point
        workflow.set_entry_point("Plan_Action")
        
        # Add edges
        workflow.add_edge("Plan_Action", "Navigate_Page")
        workflow.add_edge("Navigate_Page", "Extract_Data")
        workflow.add_edge("Extract_Data", "Follow_Links")
        workflow.add_edge("Follow_Links", "Evaluate_Progress")
        
        # Conditional routing from Evaluate_Progress
        workflow.add_conditional_edges(
            "Evaluate_Progress",
            self._Should_Continue,
            {
                "continue": "Plan_Action",
                "complete": "Compile_Result",
                "max_iterations": "Compile_Result"
            }
        )
        
        workflow.add_edge("Compile_Result", END)
        
        self.graph = workflow.compile()
    
    def _Plan_Action(self, state: Web_State) -> Web_State:
        """
        Plan the next action based on current state and task.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with action plan
        """
        task = state["task"]
        iteration = state["iteration"]
        visited_urls = state["visited_urls"]
        extracted_data = state["extracted_data"]
        action_history = state["action_history"]
        
        # Build context from history
        history_summary = ""
        if action_history:
            recent_actions = action_history[-3:]  # Last 3 actions
            history_summary = "\n".join([
                f"- {action.get('action', 'unknown')}: {action.get('result', '')[:100]}"
                for action in recent_actions
            ])
        
        system_prompt = """You are an autonomous web agent that navigates websites to complete tasks.
You can perform the following actions:
1. navigate - Navigate to a URL
2. extract - Extract specific data from the current page
3. search - Search for information within a page
4. follow_link - Follow a relevant link to gather more information
5. finish - Complete the task when sufficient information is gathered

Analyze the task and current state, then decide on the next action.
Return your decision as JSON with 'action' and 'reasoning' fields.
For 'navigate' or 'follow_link', include 'url' field.
For 'extract' or 'search', include 'query' field."""
        
        user_prompt = f"""Task: {task}
Current Iteration: {iteration}/{state['max_iterations']}
Visited URLs: {len(visited_urls)} pages
Extracted Data Points: {len(extracted_data)}

Recent Actions:
{history_summary}

What should be the next action? Consider:
- Have we gathered enough information to complete the task?
- Are there relevant links to follow?
- Do we need to extract more specific data?
- Should we search for particular information?"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content
        
        # Parse response (try JSON first, fallback to text parsing)
        try:
            # Try to extract JSON from response
            if "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                action_data = json.loads(response_text[json_start:json_end])
            else:
                # Fallback parsing
                action_data = {"action": "navigate", "reasoning": response_text}
        except:
            action_data = {"action": "navigate", "reasoning": response_text}
        
        action = action_data.get("action", "navigate").lower()
        reasoning = action_data.get("reasoning", "")
        
        # Update plan
        plan = state.get("plan", [])
        plan.append(f"Iteration {iteration}: {action} - {reasoning}")
        
        # Update action history
        action_history.append({
            "iteration": iteration,
            "action": action,
            "reasoning": reasoning,
            "data": action_data
        })
        
        state["plan"] = plan
        state["action_history"] = action_history
        state["messages"] = state.get("messages", []) + messages + [response]
        
        return state
    
    def _Navigate_Page(self, state: Web_State) -> Web_State:
        """
        Navigate to a page and fetch its content.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with page content
        """
        action_history = state["action_history"]
        if not action_history:
            return state
        
        last_action = action_history[-1]
        action_data = last_action.get("data", {})
        
        # Determine URL to navigate to
        url = None
        if action_data.get("url"):
            url = action_data["url"]
        elif state.get("current_url"):
            url = state["current_url"]
        elif state.get("links"):
            # Use first available link
            url = state["links"][0]["url"]
        
        if not url:
            # Use start URL if available
            url = state.get("current_url")
        
        if url:
            visited_urls = state.get("visited_urls", [])
            if url not in visited_urls:
                visited_urls.append(url)
                state["visited_urls"] = visited_urls
            
            state["current_url"] = url
            
            # Fetch page content
            try:
                content = Fetch_Web_Page.invoke({"url": url})
                state["_page_content"] = content  # Store temporarily
            except Exception as e:
                state["_page_content"] = f"Error fetching page: {str(e)}"
        
        return state
    
    def _Extract_Data(self, state: Web_State) -> Web_State:
        """
        Extract specific data from the current page.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with extracted data
        """
        task = state["task"]
        current_url = state.get("current_url")
        page_content = state.get("_page_content", "")
        action_history = state["action_history"]
        
        if not page_content and current_url:
            page_content = Fetch_Web_Page.invoke({"url": current_url})
        
        if not page_content:
            return state
        
        # Use LLM to extract relevant data
        extraction_prompt = f"""Task: {task}
Current Page URL: {current_url}
Page Content (first 5000 chars):
{page_content[:5000]}

Extract relevant information from this page that helps complete the task.
Return a structured summary of key findings."""
        
        messages = [
            SystemMessage(content="You are a data extraction assistant. Extract relevant information from web pages."),
            HumanMessage(content=extraction_prompt)
        ]
        
        response = self.llm.invoke(messages)
        extracted_text = response.content
        
        # Store extracted data
        extracted_data = state.get("extracted_data", [])
        extracted_data.append({
            "url": current_url,
            "data": extracted_text,
            "iteration": state["iteration"]
        })
        
        state["extracted_data"] = extracted_data
        state["messages"] = state.get("messages", []) + messages + [response]
        
        # Update action history
        action_history.append({
            "iteration": state["iteration"],
            "action": "extract",
            "result": extracted_text[:200]
        })
        state["action_history"] = action_history
        
        return state
    
    def _Follow_Links(self, state: Web_State) -> Web_State:
        """
        Identify and follow relevant links.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with extracted links
        """
        current_url = state.get("current_url")
        task = state["task"]
        visited_urls = state.get("visited_urls", [])
        max_depth = self.agent_config.Get_Max_Depth()
        
        # Check depth
        current_depth = len(visited_urls)
        if current_depth >= max_depth:
            state["links"] = []
            return state
        
        if not current_url:
            state["links"] = []
            return state
        
        # Extract links from current page
        try:
            links = Extract_Links.invoke({"url": current_url})
            
            # Filter out already visited links
            unvisited_links = [
                link for link in links
                if link["url"] not in visited_urls
            ]
            
            # Use LLM to select most relevant links
            if unvisited_links and self.agent_config.Get_Enable_Link_Following():
                link_descriptions = "\n".join([
                    f"- {link['text']}: {link['url']}"
                    for link in unvisited_links[:10]  # Limit to 10 for LLM
                ])
                
                selection_prompt = f"""Task: {task}
Available Links:
{link_descriptions}

Select up to {self.agent_config.Get_Max_Links_Per_Page()} most relevant links for completing the task.
Return JSON with 'selected_urls' array."""
                
                messages = [
                    SystemMessage(content="You are a link selection assistant. Choose relevant links based on the task."),
                    HumanMessage(content=selection_prompt)
                ]
                
                response = self.llm.invoke(messages)
                response_text = response.content
                
                # Parse selected URLs
                selected_links = []
                try:
                    if "{" in response_text:
                        json_start = response_text.find("{")
                        json_end = response_text.rfind("}") + 1
                        selection_data = json.loads(response_text[json_start:json_end])
                        selected_urls = selection_data.get("selected_urls", [])
                        
                        for link in unvisited_links:
                            if link["url"] in selected_urls:
                                selected_links.append(link)
                                if len(selected_links) >= self.agent_config.Get_Max_Links_Per_Page():
                                    break
                except:
                    # Fallback: use first few links
                    selected_links = unvisited_links[:self.agent_config.Get_Max_Links_Per_Page()]
                
                state["links"] = selected_links
            else:
                state["links"] = []
        except Exception as e:
            state["links"] = []
        
        return state
    
    def _Evaluate_Progress(self, state: Web_State) -> Web_State:
        """
        Evaluate if task is complete or needs more iterations.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with evaluation result
        """
        task = state["task"]
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        extracted_data = state.get("extracted_data", [])
        links = state.get("links", [])
        
        # Check iteration limit
        if iteration >= max_iterations:
            state["_should_continue"] = "max_iterations"
            return state
        
        # Use LLM to evaluate progress
        data_summary = "\n\n".join([
            f"From {data['url']}:\n{data['data'][:500]}"
            for data in extracted_data[-3:]  # Last 3 extractions
        ])
        
        evaluation_prompt = f"""Task: {task}
Current Iteration: {iteration}/{max_iterations}
Extracted Data Points: {len(extracted_data)}

Recent Extractions:
{data_summary}

Available Links: {len(links)} unvisited links

Evaluate if we have gathered sufficient information to complete the task.
Return JSON with 'complete' (boolean) and 'reasoning' (string) fields."""
        
        messages = [
            SystemMessage(content="You are a task evaluation assistant. Determine if enough information has been gathered."),
            HumanMessage(content=evaluation_prompt)
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content
        
        # Parse evaluation
        is_complete = False
        try:
            if "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                eval_data = json.loads(response_text[json_start:json_end])
                is_complete = eval_data.get("complete", False)
        except:
            # Default: continue if we have links, complete if no links
            is_complete = len(links) == 0 and len(extracted_data) > 0
        
        state["messages"] = state.get("messages", []) + messages + [response]
        
        if is_complete:
            state["_should_continue"] = "complete"
        else:
            state["_should_continue"] = "continue"
            state["iteration"] = iteration + 1
        
        return state
    
    def _Compile_Result(self, state: Web_State) -> Web_State:
        """
        Compile all extracted data into final answer.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with final result
        """
        task = state["task"]
        extracted_data = state.get("extracted_data", [])
        visited_urls = state.get("visited_urls", [])
        
        # Compile all extracted data
        all_data = "\n\n".join([
            f"Source: {data['url']}\nData: {data['data']}"
            for data in extracted_data
        ])
        
        compilation_prompt = f"""Task: {task}

Information Gathered from {len(visited_urls)} pages:
{all_data}

Compile a comprehensive answer to the task based on all gathered information.
Provide a well-structured summary that directly addresses the task."""
        
        messages = [
            SystemMessage(content="You are a result compilation assistant. Synthesize information into a comprehensive answer."),
            HumanMessage(content=compilation_prompt)
        ]
        
        response = self.llm.invoke(messages)
        final_result = response.content
        
        state["result"] = final_result
        state["messages"] = state.get("messages", []) + messages + [response]
        
        return state
    
    def _Should_Continue(self, state: Web_State) -> Literal["continue", "complete", "max_iterations"]:
        """
        Determine next step based on evaluation.
        
        Args:
            state: Current agent state
            
        Returns:
            Next step: "continue", "complete", or "max_iterations"
        """
        return state.get("_should_continue", "continue")
    
    def Execute_Task(self, task: str, start_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a web task.
        
        Args:
            task: Task description
            start_url: Optional starting URL
            
        Returns:
            Dictionary with result and metadata
        """
        initial_state: Web_State = {
            "task": task,
            "current_url": start_url,
            "visited_urls": [],
            "extracted_data": [],
            "links": [],
            "plan": [],
            "action_history": [],
            "result": None,
            "iteration": 0,
            "max_iterations": self.agent_config.Get_Max_Iterations(),
            "messages": []
        }
        
        if start_url:
            initial_state["visited_urls"].append(start_url)
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "result": final_state.get("result", "Task incomplete"),
            "visited_urls": final_state.get("visited_urls", []),
            "extracted_data": final_state.get("extracted_data", []),
            "plan": final_state.get("plan", []),
            "iterations": final_state.get("iteration", 0)
        }
