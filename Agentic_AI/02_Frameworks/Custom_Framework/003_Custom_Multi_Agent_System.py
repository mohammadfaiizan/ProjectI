"""
Custom Multi-Agent System - Built from Scratch
Demonstrates building a multi-agent system without any agent frameworks

This implementation includes:
- Base agent class for all agents
- Orchestrator for task routing and coordination
- Message bus for inter-agent communication
- Specialist agents (Research, Writer, Reviewer, Coder)
- Task decomposition and pipeline execution
"""

import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from openai import OpenAI


# ============================================================================
# Data Models
# ============================================================================

class Task_Status(Enum):
    """Status of a task"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a task to be executed"""
    id: str
    description: str
    agent_type: Optional[str] = None
    status: Task_Status = Task_Status.PENDING
    result: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent_Message:
    """Message between agents"""
    id: str
    sender: str
    receiver: Optional[str]  # None for broadcast
    message_type: str
    content: Any
    timestamp: float = field(default_factory=time.time)
    task_id: Optional[str] = None


@dataclass
class Agent_Capability:
    """Describes what an agent can do"""
    agent_type: str
    capabilities: List[str]
    description: str


# ============================================================================
# Message Bus
# ============================================================================

class Message_Bus:
    """Handles inter-agent communication"""
    
    def __init__(self):
        self.messages: deque = deque()
        self.agent_queues: Dict[str, deque] = {}
        self.message_history: List[Agent_Message] = []
        self.message_counter = 0
    
    def send(self, sender: str, receiver: str, message_type: str, content: Any, task_id: Optional[str] = None) -> str:
        """
        Send a message from one agent to another
        
        Args:
            sender: Sender agent ID
            receiver: Receiver agent ID
            message_type: Type of message
            content: Message content
            task_id: Optional task ID
            
        Returns:
            Message ID
        """
        msg_id = f"msg_{self.message_counter}"
        self.message_counter += 1
        
        message = Agent_Message(
            id=msg_id,
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            content=content,
            task_id=task_id
        )
        
        # Add to receiver's queue
        if receiver not in self.agent_queues:
            self.agent_queues[receiver] = deque()
        self.agent_queues[receiver].append(message)
        
        # Add to history
        self.message_history.append(message)
        
        return msg_id
    
    def broadcast(self, sender: str, message_type: str, content: Any, task_id: Optional[str] = None) -> List[str]:
        """
        Broadcast a message to all agents
        
        Args:
            sender: Sender agent ID
            message_type: Type of message
            content: Message content
            task_id: Optional task ID
            
        Returns:
            List of message IDs
        """
        msg_ids = []
        for agent_id in self.agent_queues.keys():
            if agent_id != sender:
                msg_id = self.send(sender, agent_id, message_type, content, task_id)
                msg_ids.append(msg_id)
        return msg_ids
    
    def receive(self, agent_id: str) -> Optional[Agent_Message]:
        """
        Receive a message for an agent
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Message or None if queue is empty
        """
        if agent_id not in self.agent_queues:
            return None
        
        queue = self.agent_queues[agent_id]
        if queue:
            return queue.popleft()
        return None
    
    def peek(self, agent_id: str) -> Optional[Agent_Message]:
        """Peek at next message without removing it"""
        if agent_id not in self.agent_queues:
            return None
        
        queue = self.agent_queues[agent_id]
        if queue:
            return queue[0]
        return None
    
    def register_agent(self, agent_id: str):
        """Register an agent with the message bus"""
        if agent_id not in self.agent_queues:
            self.agent_queues[agent_id] = deque()
    
    def get_message_history(self, agent_id: Optional[str] = None) -> List[Agent_Message]:
        """Get message history, optionally filtered by agent"""
        if agent_id:
            return [msg for msg in self.message_history if msg.sender == agent_id or msg.receiver == agent_id]
        return self.message_history.copy()


# ============================================================================
# Base Agent
# ============================================================================

class Base_Agent:
    """Base class for all agents"""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        llm_client: OpenAI,
        message_bus: Message_Bus,
        model: str = "gpt-4",
        temperature: float = 0.7
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.llm_client = llm_client
        self.message_bus = message_bus
        self.model = model
        self.temperature = temperature
        
        self.memory: List[Dict] = []
        self.tools: Dict[str, Callable] = {}
        
        # Register with message bus
        self.message_bus.register_agent(agent_id)
    
    def add_to_memory(self, role: str, content: str):
        """Add a message to agent's memory"""
        self.memory.append({"role": role, "content": content})
    
    def register_tool(self, name: str, function: Callable):
        """Register a tool for this agent"""
        self.tools[name] = function
    
    def call_llm(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """Call the LLM with messages"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        try:
            response = self.llm_client.chat.completions.create(**kwargs)
            return {
                "content": response.choices[0].message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in (response.choices[0].message.tool_calls or [])
                ]
            }
        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")
    
    def process_task(self, task: Task) -> str:
        """
        Process a task (to be implemented by subclasses)
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        raise NotImplementedError("Subclasses must implement process_task")
    
    def send_message(self, receiver: str, message_type: str, content: Any, task_id: Optional[str] = None):
        """Send a message to another agent"""
        return self.message_bus.send(self.agent_id, receiver, message_type, content, task_id)
    
    def broadcast_message(self, message_type: str, content: Any, task_id: Optional[str] = None):
        """Broadcast a message to all agents"""
        return self.message_bus.broadcast(self.agent_id, message_type, content, task_id)
    
    def receive_message(self) -> Optional[Agent_Message]:
        """Receive a message"""
        return self.message_bus.receive(self.agent_id)
    
    def get_capabilities(self) -> Agent_Capability:
        """Get agent capabilities"""
        return Agent_Capability(
            agent_type=self.agent_type,
            capabilities=list(self.tools.keys()),
            description=f"{self.agent_type} agent"
        )


# ============================================================================
# Specialist Agents
# ============================================================================

class Research_Agent(Base_Agent):
    """Agent specialized in research and information gathering"""
    
    def __init__(self, agent_id: str, llm_client: OpenAI, message_bus: Message_Bus, **kwargs):
        super().__init__(agent_id, "research", llm_client, message_bus, **kwargs)
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup research-specific tools"""
        def web_search(query: str) -> str:
            # Mock web search
            return json.dumps({
                "query": query,
                "results": [
                    {
                        "title": f"Research result for: {query}",
                        "summary": f"This is a mock research result about {query}. In production, this would call a real search API.",
                        "sources": [f"https://example.com/{query.replace(' ', '-')}"]
                    }
                ]
            })
        
        self.register_tool("web_search", web_search)
    
    def process_task(self, task: Task) -> str:
        """Process a research task"""
        self.add_to_memory("user", f"Research task: {task.description}")
        
        # Build prompt
        system_prompt = """You are a research agent. Your job is to search for information and provide comprehensive summaries.
Use the web_search tool to find information, then synthesize the results into a clear summary."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            *self.memory
        ]
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        response = self.call_llm(messages, tools)
        
        # Handle tool calls
        if response.get("tool_calls"):
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                
                if tool_name in self.tools:
                    result = self.tools[tool_name](**tool_args)
                    self.add_to_memory("tool", f"Tool {tool_name} result: {result}")
            
            # Get final response
            messages.append({"role": "assistant", "content": response["content"]})
            final_response = self.call_llm(messages, tools)
            result = final_response["content"]
        else:
            result = response["content"]
        
        self.add_to_memory("assistant", result)
        return result


class Writer_Agent(Base_Agent):
    """Agent specialized in writing content"""
    
    def __init__(self, agent_id: str, llm_client: OpenAI, message_bus: Message_Bus, **kwargs):
        super().__init__(agent_id, "writer", llm_client, message_bus, **kwargs)
    
    def process_task(self, task: Task) -> str:
        """Process a writing task"""
        self.add_to_memory("user", f"Writing task: {task.description}")
        
        system_prompt = """You are a professional writer. Create well-structured, engaging content based on the requirements.
Write clearly, concisely, and ensure the content is appropriate for the target audience."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            *self.memory
        ]
        
        response = self.call_llm(messages)
        result = response["content"]
        
        self.add_to_memory("assistant", result)
        return result


class Reviewer_Agent(Base_Agent):
    """Agent specialized in reviewing and critiquing content"""
    
    def __init__(self, agent_id: str, llm_client: OpenAI, message_bus: Message_Bus, **kwargs):
        super().__init__(agent_id, "reviewer", llm_client, message_bus, **kwargs)
    
    def process_task(self, task: Task) -> str:
        """Process a review task"""
        content_to_review = task.metadata.get("content", task.description)
        self.add_to_memory("user", f"Review this content: {content_to_review}")
        
        system_prompt = """You are a critical reviewer. Analyze content for:
- Clarity and readability
- Accuracy and correctness
- Structure and organization
- Grammar and style
- Overall quality

Provide constructive feedback and suggestions for improvement."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            *self.memory
        ]
        
        response = self.call_llm(messages)
        result = response["content"]
        
        self.add_to_memory("assistant", result)
        return result


class Coder_Agent(Base_Agent):
    """Agent specialized in writing code"""
    
    def __init__(self, agent_id: str, llm_client: OpenAI, message_bus: Message_Bus, **kwargs):
        super().__init__(agent_id, "coder", llm_client, message_bus, **kwargs)
    
    def process_task(self, task: Task) -> str:
        """Process a coding task"""
        self.add_to_memory("user", f"Coding task: {task.description}")
        
        system_prompt = """You are an expert Python programmer. Write clean, well-documented, and efficient code.
Follow Python best practices and PEP 8 style guidelines.
Include comments explaining complex logic."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            *self.memory
        ]
        
        response = self.call_llm(messages)
        result = response["content"]
        
        self.add_to_memory("assistant", result)
        return result


# ============================================================================
# Task Decomposer
# ============================================================================

class Task_Decomposer:
    """Decomposes complex tasks into subtasks"""
    
    def __init__(self, llm_client: OpenAI):
        self.llm_client = llm_client
    
    def decompose(self, task_description: str, available_agents: List[str]) -> List[Task]:
        """
        Decompose a task into subtasks
        
        Args:
            task_description: Original task description
            available_agents: List of available agent types
            
        Returns:
            List of subtasks
        """
        prompt = f"""Given this task: "{task_description}"

Available agent types: {', '.join(available_agents)}

Break this task down into subtasks that can be handled by the available agents.
Return a JSON array of tasks, each with:
- description: What needs to be done
- agent_type: Which agent should handle it (one of: {', '.join(available_agents)})
- dependencies: List of task IDs this depends on (empty array if none)

Example format:
[
  {{
    "description": "Research topic X",
    "agent_type": "research",
    "dependencies": []
  }},
  {{
    "description": "Write article about X",
    "agent_type": "writer",
    "dependencies": ["task_0"]
  }}
]

Return only valid JSON, no other text."""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a task decomposition expert. Break down complex tasks into manageable subtasks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            task_data = json.loads(content)
            
            # Convert to Task objects
            tasks = []
            for i, task_dict in enumerate(task_data):
                task = Task(
                    id=f"task_{i}",
                    description=task_dict["description"],
                    agent_type=task_dict.get("agent_type"),
                    dependencies=[f"task_{dep}" if isinstance(dep, int) else dep for dep in task_dict.get("dependencies", [])]
                )
                tasks.append(task)
            
            return tasks
        
        except Exception as e:
            # Fallback: create a single task
            print(f"Error decomposing task: {e}. Creating single task.")
            return [Task(id="task_0", description=task_description)]


# ============================================================================
# Pipeline Runner
# ============================================================================

class Pipeline_Runner:
    """Runs agents in sequence or parallel"""
    
    def __init__(self, agents: Dict[str, Base_Agent]):
        self.agents = agents
    
    def run_sequential(self, tasks: List[Task]) -> Dict[str, str]:
        """
        Run tasks sequentially
        
        Args:
            tasks: List of tasks to execute
            
        Returns:
            Dictionary mapping task IDs to results
        """
        results = {}
        
        for task in tasks:
            # Check dependencies
            if task.dependencies:
                for dep_id in task.dependencies:
                    if dep_id not in results:
                        raise ValueError(f"Task {task.id} depends on {dep_id} which hasn't been completed")
            
            # Find agent
            agent = self._find_agent(task.agent_type)
            if not agent:
                results[task.id] = f"Error: No agent found for type {task.agent_type}"
                task.status = Task_Status.FAILED
                continue
            
            # Execute task
            try:
                task.status = Task_Status.IN_PROGRESS
                result = agent.process_task(task)
                results[task.id] = result
                task.result = result
                task.status = Task_Status.COMPLETED
            except Exception as e:
                results[task.id] = f"Error: {str(e)}"
                task.status = Task_Status.FAILED
        
        return results
    
    def run_parallel(self, tasks: List[Task]) -> Dict[str, str]:
        """
        Run independent tasks in parallel (simplified - actually runs sequentially)
        In a real implementation, this would use threading/async
        
        Args:
            tasks: List of tasks to execute
            
        Returns:
            Dictionary mapping task IDs to results
        """
        # Group tasks by dependencies
        independent_tasks = [t for t in tasks if not t.dependencies]
        dependent_tasks = [t for t in tasks if t.dependencies]
        
        results = {}
        
        # Run independent tasks first
        for task in independent_tasks:
            agent = self._find_agent(task.agent_type)
            if agent:
                try:
                    task.status = Task_Status.IN_PROGRESS
                    result = agent.process_task(task)
                    results[task.id] = result
                    task.result = result
                    task.status = Task_Status.COMPLETED
                except Exception as e:
                    results[task.id] = f"Error: {str(e)}"
                    task.status = Task_Status.FAILED
        
        # Run dependent tasks
        for task in dependent_tasks:
            # Check dependencies
            if all(dep_id in results for dep_id in task.dependencies):
                agent = self._find_agent(task.agent_type)
                if agent:
                    # Add dependency results to task metadata
                    task.metadata["dependency_results"] = {
                        dep_id: results[dep_id] for dep_id in task.dependencies
                    }
                    
                    try:
                        task.status = Task_Status.IN_PROGRESS
                        result = agent.process_task(task)
                        results[task.id] = result
                        task.result = result
                        task.status = Task_Status.COMPLETED
                    except Exception as e:
                        results[task.id] = f"Error: {str(e)}"
                        task.status = Task_Status.FAILED
        
        return results
    
    def _find_agent(self, agent_type: Optional[str]) -> Optional[Base_Agent]:
        """Find an agent by type"""
        if not agent_type:
            return None
        
        for agent in self.agents.values():
            if agent.agent_type == agent_type:
                return agent
        return None


# ============================================================================
# Orchestrator
# ============================================================================

class Orchestrator:
    """Orchestrates multi-agent workflows"""
    
    def __init__(
        self,
        agents: Dict[str, Base_Agent],
        message_bus: Message_Bus,
        task_decomposer: Task_Decomposer,
        pipeline_runner: Pipeline_Runner
    ):
        self.agents = agents
        self.message_bus = message_bus
        self.task_decomposer = task_decomposer
        self.pipeline_runner = pipeline_runner
        self.task_counter = 0
    
    def process_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process a user request through the multi-agent system
        
        Args:
            user_request: User's request
            
        Returns:
            Dictionary with results and metadata
        """
        # Decompose task
        available_agents = list(set(agent.agent_type for agent in self.agents.values()))
        subtasks = self.task_decomposer.decompose(user_request, available_agents)
        
        # Execute pipeline
        results = self.pipeline_runner.run_sequential(subtasks)
        
        # Aggregate results
        final_result = self._aggregate_results(subtasks, results)
        
        return {
            "original_request": user_request,
            "subtasks": [
                {
                    "id": task.id,
                    "description": task.description,
                    "agent_type": task.agent_type,
                    "status": task.status.value,
                    "result": task.result
                }
                for task in subtasks
            ],
            "results": results,
            "final_result": final_result
        }
    
    def _aggregate_results(self, tasks: List[Task], results: Dict[str, str]) -> str:
        """Aggregate results from multiple tasks"""
        # Simple aggregation: combine all results
        aggregated = []
        for task in tasks:
            if task.status == Task_Status.COMPLETED:
                aggregated.append(f"Task: {task.description}\nResult: {results[task.id]}\n")
        
        return "\n".join(aggregated)
    
    def select_agent(self, task_description: str) -> Optional[Base_Agent]:
        """Select the best agent for a task"""
        # Simple selection: use task decomposer's recommendation
        # In a more sophisticated system, this would use agent capabilities
        available_agents = list(set(agent.agent_type for agent in self.agents.values()))
        subtasks = self.task_decomposer.decompose(task_description, available_agents)
        
        if subtasks:
            agent_type = subtasks[0].agent_type
            for agent in self.agents.values():
                if agent.agent_type == agent_type:
                    return agent
        
        return None


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function demonstrating the multi-agent system"""
    
    # Configuration
    API_KEY = "your-openai-api-key-here"  # Replace with your API key
    MODEL = "gpt-4"
    
    print("=" * 70)
    print("Custom Multi-Agent System Demo")
    print("=" * 70)
    print()
    
    # Initialize components
    print("Initializing components...")
    
    # LLM Client
    llm_client = OpenAI(api_key=API_KEY)
    
    # Message Bus
    message_bus = Message_Bus()
    
    # Create specialist agents
    agents = {}
    
    research_agent = Research_Agent("research_1", llm_client, message_bus, model=MODEL)
    agents["research_1"] = research_agent
    
    writer_agent = Writer_Agent("writer_1", llm_client, message_bus, model=MODEL)
    agents["writer_1"] = writer_agent
    
    reviewer_agent = Reviewer_Agent("reviewer_1", llm_client, message_bus, model=MODEL)
    agents["reviewer_1"] = reviewer_agent
    
    coder_agent = Coder_Agent("coder_1", llm_client, message_bus, model=MODEL)
    agents["coder_1"] = coder_agent
    
    print(f"Created {len(agents)} agents:")
    for agent_id, agent in agents.items():
        print(f"  - {agent_id}: {agent.agent_type} agent")
    
    # Task Decomposer
    task_decomposer = Task_Decomposer(llm_client)
    
    # Pipeline Runner
    pipeline_runner = Pipeline_Runner(agents)
    
    # Orchestrator
    orchestrator = Orchestrator(agents, message_bus, task_decomposer, pipeline_runner)
    
    print("\nMulti-agent system initialized!")
    print()
    
    # Sample complex request
    complex_request = """Create a comprehensive guide about Python decorators. 
    Research the topic, write a detailed article, and then review it for quality."""
    
    print("Processing complex request...")
    print("-" * 70)
    print(f"Request: {complex_request}")
    print("-" * 70)
    print()
    
    try:
        result = orchestrator.process_request(complex_request)
        
        print("Task Decomposition:")
        print("-" * 70)
        for i, subtask in enumerate(result["subtasks"], 1):
            print(f"{i}. [{subtask['agent_type']}] {subtask['description']}")
            print(f"   Status: {subtask['status']}")
            if subtask['result']:
                result_preview = subtask['result'][:200] + "..." if len(subtask['result']) > 200 else subtask['result']
                print(f"   Result: {result_preview}")
            print()
        
        print("=" * 70)
        print("Final Aggregated Result:")
        print("=" * 70)
        print(result["final_result"])
        
        print("\n" + "=" * 70)
        print("Message Bus Statistics:")
        print("=" * 70)
        print(f"Total messages: {len(message_bus.message_history)}")
        print(f"Registered agents: {len(message_bus.agent_queues)}")
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    import os
    
    # Try to get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("Found OPENAI_API_KEY in environment")
    
    main()
