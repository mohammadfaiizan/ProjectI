"""
Custom Agent Framework - Single Agent Implementation
Built from scratch using only the OpenAI package (no agent frameworks)

This implementation demonstrates:
- LLM client wrapper
- Tool registry and execution
- Memory management
- Prompt management
- Complete agent loop with perception-reasoning-action pattern
"""

import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from openai import OpenAI


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Message:
    """Represents a single message in the conversation"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert message to OpenAI API format"""
        msg = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


@dataclass
class Tool_Call:
    """Represents a tool call request"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Tool_Definition:
    """Definition of a tool available to the agent"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema
    function: Callable


# ============================================================================
# LLM Client
# ============================================================================

class LLM_Client:
    """Wrapper around OpenAI API for chat completions"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.7):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def complete(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict:
        """
        Send a completion request to the LLM
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            stream: Whether to stream the response
            
        Returns:
            Response dictionary from OpenAI API
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        if stream:
            kwargs["stream"] = True
        
        for attempt in range(self.max_retries):
            try:
                if stream:
                    return self.client.chat.completions.create(**kwargs)
                else:
                    response = self.client.chat.completions.create(**kwargs)
                    return {
                        "id": response.id,
                        "choices": [
                            {
                                "message": {
                                    "role": choice.message.role,
                                    "content": choice.message.content,
                                    "tool_calls": [
                                        {
                                            "id": tc.id,
                                            "type": tc.type,
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments
                                            }
                                        }
                                        for tc in (choice.message.tool_calls or [])
                                    ]
                                },
                                "finish_reason": choice.finish_reason
                            }
                            for choice in response.choices
                        ]
                    }
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise Exception(f"LLM request failed after {self.max_retries} attempts: {str(e)}")
    
    def stream_complete(self, messages: List[Dict], tools: Optional[List[Dict]] = None):
        """Stream completion responses"""
        return self.complete(messages, tools, stream=True)


# ============================================================================
# Tool Registry
# ============================================================================

class Tool_Registry:
    """Manages tool registration and execution"""
    
    def __init__(self):
        self.tools: Dict[str, Tool_Definition] = {}
    
    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable
    ):
        """
        Register a tool with the registry
        
        Args:
            name: Tool name (must be unique)
            description: Tool description for LLM
            parameters: JSON schema for tool parameters
            function: Python function to execute
        """
        if name in self.tools:
            raise ValueError(f"Tool '{name}' already registered")
        
        tool_def = Tool_Definition(
            name=name,
            description=description,
            parameters=parameters,
            function=function
        )
        self.tools[name] = tool_def
    
    def get_tool(self, name: str) -> Optional[Tool_Definition]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())
    
    def get_tools_for_llm(self) -> List[Dict]:
        """Convert tools to OpenAI function calling format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools.values()
        ]
    
    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool with given arguments
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result as string
        """
        tool = self.get_tool(name)
        if not tool:
            return json.dumps({"error": f"Tool '{name}' not found"})
        
        try:
            # Validate arguments against schema (simplified)
            result = tool.function(**arguments)
            
            # Convert result to string
            if isinstance(result, (dict, list)):
                return json.dumps(result)
            elif isinstance(result, str):
                return result
            else:
                return str(result)
        except Exception as e:
            error_msg = f"Error executing tool '{name}': {str(e)}"
            return json.dumps({"error": error_msg})


# ============================================================================
# Built-in Tools
# ============================================================================

def calculator_tool(expression: str) -> str:
    """Evaluate a mathematical expression safely"""
    try:
        # Only allow safe mathematical operations
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def current_time_tool() -> str:
    """Get the current date and time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def web_search_tool(query: str) -> str:
    """Mock web search tool (returns simulated results)"""
    # In a real implementation, this would call a search API
    return json.dumps({
        "query": query,
        "results": [
            {
                "title": f"Result about {query}",
                "url": f"https://example.com/{query.replace(' ', '-')}",
                "snippet": f"This is a mock search result for '{query}'"
            }
        ],
        "note": "This is a mock search tool - implement with real API in production"
    })


def file_reader_tool(filepath: str) -> str:
    """Read contents of a text file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return json.dumps({
            "filepath": filepath,
            "content": content,
            "success": True
        })
    except FileNotFoundError:
        return json.dumps({
            "filepath": filepath,
            "error": "File not found",
            "success": False
        })
    except Exception as e:
        return json.dumps({
            "filepath": filepath,
            "error": str(e),
            "success": False
        })


# ============================================================================
# Memory Manager
# ============================================================================

class Memory_Manager:
    """Manages conversation history and memory"""
    
    def __init__(self, max_messages: int = 50, enable_summarization: bool = False):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.enable_summarization = enable_summarization
        self.summary: Optional[str] = None
    
    def add_message(self, role: str, content: str, tool_call_id: Optional[str] = None):
        """Add a message to the conversation history"""
        message = Message(role=role, content=content, tool_call_id=tool_call_id)
        self.messages.append(message)
        
        # Enforce sliding window
        if len(self.messages) > self.max_messages:
            if self.enable_summarization:
                self._summarize_old_messages()
            else:
                # Remove oldest messages (keep system message if present)
                system_msgs = [m for m in self.messages if m.role == "system"]
                other_msgs = [m for m in self.messages if m.role != "system"]
                self.messages = system_msgs + other_msgs[-(self.max_messages - len(system_msgs)):]
    
    def get_history(self) -> List[Message]:
        """Get full conversation history"""
        return self.messages.copy()
    
    def get_history_for_llm(self) -> List[Dict]:
        """Get conversation history in OpenAI API format"""
        return [msg.to_dict() for msg in self.messages]
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.summary = None
    
    def _summarize_old_messages(self):
        """Summarize old messages to save space (simplified implementation)"""
        # In a full implementation, this would use an LLM to summarize
        # For now, we just keep a simple note
        if len(self.messages) > self.max_messages:
            old_count = len(self.messages) - self.max_messages
            self.summary = f"Previous {old_count} messages summarized"
            # Remove oldest non-system messages
            system_msgs = [m for m in self.messages if m.role == "system"]
            other_msgs = [m for m in self.messages if m.role != "system"]
            self.messages = system_msgs + other_msgs[-(self.max_messages - len(system_msgs)):]


# ============================================================================
# Prompt Manager
# ============================================================================

class Prompt_Manager:
    """Manages prompt construction and context assembly"""
    
    def __init__(self, system_prompt: Optional[str] = None):
        self.base_system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Default system prompt"""
        return """You are a helpful AI assistant with access to various tools.
You can use tools to help answer questions and complete tasks.
When you need to use a tool, call it with the appropriate parameters.
Always provide clear and helpful responses to users."""
    
    def build_system_prompt(self, available_tools: List[str]) -> str:
        """Build system prompt with context about available tools"""
        prompt = self.base_system_prompt
        if available_tools:
            tool_list = ", ".join(available_tools)
            prompt += f"\n\nAvailable tools: {tool_list}"
        return prompt
    
    def assemble_messages(
        self,
        memory: Memory_Manager,
        user_input: str,
        available_tools: List[str]
    ) -> List[Dict]:
        """
        Assemble complete message list for LLM
        
        Args:
            memory: Memory manager with conversation history
            user_input: Current user input
            available_tools: List of available tool names
            
        Returns:
            List of messages in OpenAI format
        """
        messages = []
        
        # Add system prompt if not already in memory
        has_system = any(m.role == "system" for m in memory.get_history())
        if not has_system:
            system_prompt = self.build_system_prompt(available_tools)
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        history = memory.get_history_for_llm()
        messages.extend(history)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def update_system_prompt(self, new_prompt: str):
        """Update the base system prompt"""
        self.base_system_prompt = new_prompt


# ============================================================================
# Agent Class
# ============================================================================

class Agent:
    """Main agent class implementing perception-reasoning-action loop"""
    
    def __init__(
        self,
        llm_client: LLM_Client,
        tool_registry: Tool_Registry,
        memory_manager: Memory_Manager,
        prompt_manager: Prompt_Manager,
        max_iterations: int = 10
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.memory = memory_manager
        self.prompt_manager = prompt_manager
        self.max_iterations = max_iterations
    
    def process(self, user_input: str) -> str:
        """
        Process user input through the agent loop
        
        Args:
            user_input: User's input message
            
        Returns:
            Agent's response
        """
        # Add user message to memory
        self.memory.add_message("user", user_input)
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            try:
                # Assemble messages for LLM
                available_tools = self.tool_registry.list_tools()
                messages = self.prompt_manager.assemble_messages(
                    self.memory,
                    user_input if iteration == 1 else "",
                    available_tools
                )
                
                # Get tools in LLM format
                tools = self.tool_registry.get_tools_for_llm() if available_tools else None
                
                # Call LLM (Reasoning)
                response = self.llm_client.complete(messages, tools)
                
                if not response.get("choices"):
                    return "Error: No response from LLM"
                
                choice = response["choices"][0]
                message = choice["message"]
                
                # Check if LLM wants to call tools
                tool_calls = message.get("tool_calls", [])
                
                if tool_calls:
                    # Execute tools (Action)
                    for tool_call in tool_calls:
                        tool_name = tool_call["function"]["name"]
                        tool_args = json.loads(tool_call["function"]["arguments"])
                        
                        # Execute tool
                        tool_result = self.tool_registry.execute_tool(tool_name, tool_args)
                        
                        # Add tool call and result to memory
                        self.memory.add_message(
                            "assistant",
                            "",
                            tool_call_id=tool_call["id"]
                        )
                        self.memory.add_message(
                            "tool",
                            tool_result,
                            tool_call_id=tool_call["id"]
                        )
                    
                    # Continue loop to get final response
                    continue
                
                else:
                    # LLM generated a text response
                    content = message.get("content", "")
                    
                    # Add assistant response to memory
                    self.memory.add_message("assistant", content)
                    
                    return content
            
            except Exception as e:
                error_msg = f"Error in agent loop: {str(e)}"
                print(f"Agent error: {error_msg}")
                print(traceback.format_exc())
                return f"I encountered an error: {error_msg}"
        
        return "Error: Maximum iterations reached. The agent may be stuck in a loop."
    
    def reset(self):
        """Reset agent state (clear memory)"""
        self.memory.clear()
    
    def get_conversation_history(self) -> List[Message]:
        """Get current conversation history"""
        return self.memory.get_history()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function demonstrating the custom agent"""
    
    # Configuration
    API_KEY = "your-openai-api-key-here"  # Replace with your API key
    MODEL = "gpt-4"
    
    print("=" * 70)
    print("Custom Agent Framework - Single Agent Demo")
    print("=" * 70)
    print()
    
    # Initialize components
    print("Initializing components...")
    
    # LLM Client
    llm_client = LLM_Client(api_key=API_KEY, model=MODEL, temperature=0.7)
    
    # Tool Registry
    tool_registry = Tool_Registry()
    
    # Register built-in tools
    tool_registry.register(
        name="calculator",
        description="Performs mathematical calculations. Input should be a valid mathematical expression.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2+2', '10*5', 'sqrt(16)')"
                }
            },
            "required": ["expression"]
        },
        function=calculator_tool
    )
    
    tool_registry.register(
        name="current_time",
        description="Gets the current date and time",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        },
        function=current_time_tool
    )
    
    tool_registry.register(
        name="web_search",
        description="Searches the web for information. Note: This is a mock implementation.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        },
        function=web_search_tool
    )
    
    tool_registry.register(
        name="file_reader",
        description="Reads the contents of a text file",
        parameters={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["filepath"]
        },
        function=file_reader_tool
    )
    
    print(f"Registered {len(tool_registry.list_tools())} tools: {', '.join(tool_registry.list_tools())}")
    
    # Memory Manager
    memory_manager = Memory_Manager(max_messages=50, enable_summarization=False)
    
    # Prompt Manager
    prompt_manager = Prompt_Manager()
    
    # Create Agent
    agent = Agent(
        llm_client=llm_client,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        prompt_manager=prompt_manager,
        max_iterations=10
    )
    
    print("Agent initialized successfully!")
    print()
    
    # Sample queries
    sample_queries = [
        "What is 15 multiplied by 23?",
        "What time is it now?",
        "Can you search for information about Python programming?",
        "Calculate the result of (100 + 50) / 2",
    ]
    
    print("Running sample queries...")
    print("-" * 70)
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 70)
        
        try:
            response = agent.process(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print()
    
    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)
    
    # Show conversation history
    print("\nConversation History:")
    print("-" * 70)
    for msg in agent.get_conversation_history():
        role = msg.role.upper()
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"[{role}]: {content}")


if __name__ == "__main__":
    # Note: Replace API_KEY in main() with your actual OpenAI API key
    # Or set it as an environment variable and read it here
    import os
    
    # Try to get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        # Update the main function to use environment variable
        print("Found OPENAI_API_KEY in environment")
    
    main()
