"""
Multi-Agent Task Solver Implementation
A complete multi-agent system where an orchestrator decomposes complex tasks
and delegates to specialist agents using OpenAI.
"""

import os
import json
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from openai import OpenAI


class Message_Type(Enum):
    """Types of messages between agents."""
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"


@dataclass
class Message:
    """Represents a message between agents."""
    sender: str
    receiver: str
    content: str
    message_type: Message_Type
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class Message_Bus:
    """Manages inter-agent communication."""
    
    def __init__(self):
        self.message_queues: Dict[str, List[Message]] = {}
        self.message_history: List[Message] = []
        self.agents: Dict[str, Any] = {}
    
    def register_agent(self, agent_id: str, agent: Any):
        """Register an agent with the message bus."""
        self.agents[agent_id] = agent
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = []
    
    def send_message(self, message: Message):
        """Send a message to the specified receiver."""
        if message.receiver == "broadcast":
            for agent_id in self.agents.keys():
                if agent_id != message.sender:
                    self.message_queues[agent_id].append(message)
        else:
            if message.receiver not in self.message_queues:
                self.message_queues[message.receiver] = []
            self.message_queues[message.receiver].append(message)
        
        self.message_history.append(message)
    
    def get_messages(self, agent_id: str) -> List[Message]:
        """Get all pending messages for an agent."""
        if agent_id not in self.message_queues:
            return []
        messages = self.message_queues[agent_id]
        self.message_queues[agent_id] = []
        return messages
    
    def get_message_history(self) -> List[Message]:
        """Get complete message history."""
        return self.message_history


class Base_Agent:
    """Base class for all specialist agents."""
    
    def __init__(self, agent_id: str, agent_name: str, client: OpenAI, message_bus: Message_Bus):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.client = client
        self.message_bus = message_bus
        self.capabilities: List[str] = []
        self.message_bus.register_agent(agent_id, self)
    
    def think(self, context: str) -> str:
        """Use LLM to reason about the context."""
        system_prompt = f"You are {self.agent_name}, a specialist agent. Analyze the situation and plan your approach."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in thinking: {str(e)}"
    
    def act(self, action_description: str) -> str:
        """Execute an action based on the description."""
        # Base implementation - specialized agents override this
        return f"Executed: {action_description}"
    
    def respond(self, query: str, context: Optional[str] = None) -> str:
        """Generate a response to a query."""
        system_prompt = f"You are {self.agent_name}. Provide a helpful and accurate response."
        user_content = query
        if context:
            user_content = f"Context: {context}\n\nQuery: {query}"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def send_message(self, receiver: str, content: str, message_type: Message_Type, metadata: Optional[Dict] = None):
        """Send a message to another agent."""
        message = Message(
            sender=self.agent_id,
            receiver=receiver,
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )
        self.message_bus.send_message(message)
    
    def receive_messages(self) -> List[Message]:
        """Receive pending messages."""
        return self.message_bus.get_messages(self.agent_id)
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities."""
        return self.capabilities
    
    def process_task(self, task_description: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a task assignment. Override in subclasses."""
        reasoning = self.think(f"Task: {task_description}\nContext: {context or {}}")
        result = self.act(task_description)
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "task": task_description,
            "reasoning": reasoning,
            "result": result,
            "status": "completed"
        }


class Research_Agent(Base_Agent):
    """Specialist agent for research and information gathering."""
    
    def __init__(self, agent_id: str, client: OpenAI, message_bus: Message_Bus):
        super().__init__(agent_id, "Research Agent", client, message_bus)
        self.capabilities = ["web_research", "information_gathering", "fact_verification", "source_citation"]
    
    def act(self, action_description: str) -> str:
        """Perform research actions."""
        # Simulate web search and research
        research_prompt = f"""You are a research specialist. Perform research on the following topic:
        
{action_description}

Provide:
1. Key findings and information
2. Relevant sources and citations
3. Important facts and statistics
4. Current state of knowledge on the topic

Format your response as a comprehensive research summary."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert researcher. Provide accurate, well-sourced information."},
                    {"role": "user", "content": research_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Research error: {str(e)}"
    
    def process_task(self, task_description: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process research task."""
        reasoning = self.think(f"Research task: {task_description}")
        research_result = self.act(task_description)
        
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "task": task_description,
            "reasoning": reasoning,
            "result": research_result,
            "sources": "Multiple sources synthesized",
            "status": "completed"
        }


class Coding_Agent(Base_Agent):
    """Specialist agent for coding and software development."""
    
    def __init__(self, agent_id: str, client: OpenAI, message_bus: Message_Bus):
        super().__init__(agent_id, "Coding Agent", client, message_bus)
        self.capabilities = ["code_generation", "code_review", "debugging", "documentation", "testing"]
    
    def act(self, action_description: str) -> str:
        """Perform coding actions."""
        coding_prompt = f"""You are a software development specialist. Complete the following coding task:

{action_description}

Provide:
1. Complete, working code
2. Code comments and documentation
3. Explanation of the approach
4. Any important considerations or best practices

Format your response with clear code blocks."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert software developer. Write clean, efficient, well-documented code."},
                    {"role": "user", "content": coding_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Coding error: {str(e)}"
    
    def process_task(self, task_description: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process coding task."""
        reasoning = self.think(f"Coding task: {task_description}")
        code_result = self.act(task_description)
        
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "task": task_description,
            "reasoning": reasoning,
            "result": code_result,
            "code_provided": True,
            "status": "completed"
        }


class Writing_Agent(Base_Agent):
    """Specialist agent for content creation and writing."""
    
    def __init__(self, agent_id: str, client: OpenAI, message_bus: Message_Bus):
        super().__init__(agent_id, "Writing Agent", client, message_bus)
        self.capabilities = ["content_generation", "editing", "formatting", "summarization", "translation"]
    
    def act(self, action_description: str) -> str:
        """Perform writing actions."""
        writing_prompt = f"""You are a professional writer and content creator. Complete the following writing task:

{action_description}

Provide:
1. Well-structured, engaging content
2. Appropriate tone and style
3. Clear organization and flow
4. Professional formatting

Create high-quality written content."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert writer. Create clear, engaging, well-formatted content."},
                    {"role": "user", "content": writing_prompt}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Writing error: {str(e)}"
    
    def process_task(self, task_description: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process writing task."""
        reasoning = self.think(f"Writing task: {task_description}")
        writing_result = self.act(task_description)
        
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "task": task_description,
            "reasoning": reasoning,
            "result": writing_result,
            "word_count": len(writing_result.split()),
            "status": "completed"
        }


class Analysis_Agent(Base_Agent):
    """Specialist agent for data analysis and interpretation."""
    
    def __init__(self, agent_id: str, client: OpenAI, message_bus: Message_Bus):
        super().__init__(agent_id, "Analysis Agent", client, message_bus)
        self.capabilities = ["data_analysis", "statistical_analysis", "comparison", "trend_identification", "recommendations"]
    
    def act(self, action_description: str) -> str:
        """Perform analysis actions."""
        analysis_prompt = f"""You are a data analysis specialist. Complete the following analysis task:

{action_description}

Provide:
1. Detailed analysis of the data or situation
2. Key insights and findings
3. Statistical observations if applicable
4. Comparisons and trends
5. Actionable recommendations

Format your response as a comprehensive analysis report."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert data analyst. Provide thorough, accurate analysis with clear insights."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def process_task(self, task_description: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process analysis task."""
        reasoning = self.think(f"Analysis task: {task_description}")
        analysis_result = self.act(task_description)
        
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "task": task_description,
            "reasoning": reasoning,
            "result": analysis_result,
            "insights_provided": True,
            "status": "completed"
        }


class Task_Decomposer:
    """Decomposes complex tasks into subtasks."""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def decompose_task(self, task_description: str) -> Dict[str, Any]:
        """Break down a complex task into subtasks."""
        decomposition_prompt = f"""Analyze the following complex task and break it down into manageable subtasks.

Task: {task_description}

For each subtask, provide:
1. Subtask description
2. Required agent type (research, coding, writing, analysis)
3. Dependencies (which subtasks must complete first)
4. Expected output

Return your response as a JSON object with this structure:
{{
    "subtasks": [
        {{
            "id": "subtask_1",
            "description": "subtask description",
            "agent_type": "research|coding|writing|analysis",
            "dependencies": [],
            "expected_output": "what this subtask should produce"
        }}
    ],
    "execution_order": ["subtask_1", "subtask_2", ...],
    "estimated_complexity": "low|medium|high"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at task decomposition. Break down complex tasks into clear, actionable subtasks."},
                    {"role": "user", "content": decomposition_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            # Fallback decomposition
            return {
                "subtasks": [
                    {
                        "id": "subtask_1",
                        "description": task_description,
                        "agent_type": "research",
                        "dependencies": [],
                        "expected_output": "Task completion"
                    }
                ],
                "execution_order": ["subtask_1"],
                "estimated_complexity": "medium"
            }


class Orchestrator:
    """Orchestrates multi-agent task execution."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.message_bus = Message_Bus()
        self.task_decomposer = Task_Decomposer(client)
        self.agents: Dict[str, Base_Agent] = {}
        self.task_results: Dict[str, Dict[str, Any]] = {}
        
        # Initialize specialist agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all specialist agents."""
        self.agents["research"] = Research_Agent("research_agent", self.client, self.message_bus)
        self.agents["coding"] = Coding_Agent("coding_agent", self.client, self.message_bus)
        self.agents["writing"] = Writing_Agent("writing_agent", self.client, self.message_bus)
        self.agents["analysis"] = Analysis_Agent("analysis_agent", self.client, self.message_bus)
    
    def decompose_task(self, task_description: str) -> Dict[str, Any]:
        """Decompose a complex task into subtasks."""
        return self.task_decomposer.decompose_task(task_description)
    
    def route_subtask(self, subtask: Dict[str, Any]) -> Optional[Base_Agent]:
        """Route a subtask to the appropriate agent."""
        agent_type = subtask.get("agent_type", "research").lower()
        
        agent_mapping = {
            "research": "research",
            "coding": "coding",
            "code": "coding",
            "writing": "writing",
            "write": "writing",
            "analysis": "analysis",
            "analyze": "analysis"
        }
        
        mapped_type = agent_mapping.get(agent_type, "research")
        return self.agents.get(mapped_type)
    
    def execute_plan(self, plan: Dict[str, Any], task_id: str = "main_task") -> Dict[str, Any]:
        """Execute a plan of subtasks."""
        subtasks = plan.get("subtasks", [])
        execution_order = plan.get("execution_order", [])
        
        results = {}
        completed_subtasks = set()
        
        # Execute subtasks in order
        for subtask_id in execution_order:
            subtask = next((s for s in subtasks if s["id"] == subtask_id), None)
            if not subtask:
                continue
            
            # Check dependencies
            dependencies = subtask.get("dependencies", [])
            if not all(dep in completed_subtasks for dep in dependencies):
                print(f"Skipping {subtask_id}: dependencies not met")
                continue
            
            # Route to appropriate agent
            agent = self.route_subtask(subtask)
            if not agent:
                print(f"No agent found for subtask {subtask_id}")
                continue
            
            # Execute subtask
            print(f"Executing {subtask_id} with {agent.agent_name}...")
            subtask_result = agent.process_task(
                subtask["description"],
                context={"subtask_id": subtask_id, "dependencies": dependencies}
            )
            
            results[subtask_id] = subtask_result
            completed_subtasks.add(subtask_id)
            self.task_results[subtask_id] = subtask_result
            
            # Allow agents to communicate
            self._process_agent_messages()
        
        return results
    
    def _process_agent_messages(self):
        """Process pending messages between agents."""
        for agent_id, agent in self.agents.items():
            messages = agent.receive_messages()
            for message in messages:
                if message.message_type == Message_Type.QUERY:
                    # Agent is asking a question
                    response_agent = self.agents.get(message.receiver)
                    if response_agent:
                        response = response_agent.respond(message.content)
                        response_agent.send_message(
                            message.sender,
                            response,
                            Message_Type.RESPONSE
                        )
    
    def aggregate_results(self, subtask_results: Dict[str, Dict[str, Any]], original_task: str) -> str:
        """Aggregate results from multiple agents into final output."""
        aggregation_prompt = f"""You are synthesizing results from multiple specialist agents to complete a complex task.

Original Task: {original_task}

Agent Results:
{json.dumps(subtask_results, indent=2)}

Synthesize these results into a coherent, comprehensive final output that:
1. Integrates all agent contributions
2. Resolves any conflicts or contradictions
3. Provides a complete answer to the original task
4. Maintains quality and coherence
5. Cites which agents contributed what information

Provide the final synthesized result."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at synthesizing information from multiple sources into coherent outputs."},
                    {"role": "user", "content": aggregation_prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback aggregation
            aggregated = f"Task: {original_task}\n\n"
            for subtask_id, result in subtask_results.items():
                aggregated += f"\n{result.get('agent_name', 'Agent')} Result:\n{result.get('result', 'No result')}\n"
            return aggregated
    
    def solve_task(self, task_description: str) -> Dict[str, Any]:
        """Complete workflow: decompose, execute, aggregate."""
        print(f"\n{'='*60}")
        print(f"Solving Task: {task_description}")
        print(f"{'='*60}\n")
        
        # Step 1: Decompose task
        print("Step 1: Decomposing task...")
        plan = self.decompose_task(task_description)
        print(f"Created {len(plan.get('subtasks', []))} subtasks\n")
        
        # Step 2: Execute plan
        print("Step 2: Executing subtasks...")
        subtask_results = self.execute_plan(plan, task_id="main_task")
        print(f"\nCompleted {len(subtask_results)} subtasks\n")
        
        # Step 3: Aggregate results
        print("Step 3: Aggregating results...")
        final_result = self.aggregate_results(subtask_results, task_description)
        
        return {
            "original_task": task_description,
            "plan": plan,
            "subtask_results": subtask_results,
            "final_result": final_result,
            "status": "completed"
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of task execution statistics."""
        return {
            "total_tasks_executed": len(self.task_results),
            "agents_used": list(set(r.get("agent_name") for r in self.task_results.values())),
            "message_count": len(self.message_bus.get_message_history())
        }


def main():
    """Main function demonstrating the multi-agent task solver."""
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it using: export OPENAI_API_KEY='your-key-here'")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Create orchestrator
    orchestrator = Orchestrator(client)
    
    # Example complex task
    complex_task = """Create a comprehensive report on Python web frameworks. 
Include: research on popular frameworks (Flask, Django, FastAPI), 
code examples comparing Flask and Django for a simple REST API, 
analysis of performance metrics and use cases, and a well-formatted 
summary document with recommendations."""
    
    # Solve the task
    result = orchestrator.solve_task(complex_task)
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(result["final_result"])
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Original Task: {result['original_task']}")
    print(f"Number of Subtasks: {len(result['plan'].get('subtasks', []))}")
    print(f"Completed Subtasks: {len(result['subtask_results'])}")
    print("\nSubtask Details:")
    for subtask_id, subtask_result in result['subtask_results'].items():
        print(f"\n  {subtask_id}:")
        print(f"    Agent: {subtask_result.get('agent_name')}")
        print(f"    Status: {subtask_result.get('status')}")
        print(f"    Result Preview: {subtask_result.get('result', '')[:100]}...")


if __name__ == "__main__":
    main()
