"""
AutoGen Multi-Agent Chat Examples

This module demonstrates various multi-agent conversation patterns using AutoGen:
1. Two-Agent Chat: Basic conversation between AssistantAgent and UserProxyAgent
2. Group Chat: Multiple agents collaborating in a group conversation
3. Function Registration: Registering and using custom functions
4. Nested Chat: Agents initiating sub-conversations
5. Human-in-the-Loop: Different input modes for UserProxyAgent
6. Custom Speaker Selection: Controlling group chat flow
"""

import os
from typing import Dict, Any, List, Optional
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager


# ============================================================================
# Configuration
# ============================================================================

def get_llm_config() -> Dict[str, Any]:
    """
    Get LLM configuration for agents.
    Modify this to use your preferred LLM provider and API keys.
    """
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    
    config_list = [
        {
            "model": "gpt-4",
            "api_key": api_key,
            "base_url": None,
            "api_type": "open_ai"
        }
    ]
    
    return {
        "config_list": config_list,
        "temperature": 0.7,
        "timeout": 120,
        "max_tokens": 2000
    }


# ============================================================================
# Example 1: Two-Agent Chat
# ============================================================================

def example_two_agent_chat():
    """
    Basic two-agent conversation pattern.
    AssistantAgent provides answers, UserProxyAgent executes code and manages interaction.
    """
    print("\n" + "="*80)
    print("Example 1: Two-Agent Chat")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    # Create AssistantAgent
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful AI assistant. You can help with problem solving, "
                      "code generation, and answering questions. When asked to write code, "
                      "provide complete, runnable code blocks.",
        llm_config=llm_config
    )
    
    # Create UserProxyAgent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    # Start conversation
    task = """
    Write a Python function that calculates the factorial of a number.
    Then test it with the number 5 and show the result.
    """
    
    print(f"Task: {task}\n")
    print("Starting conversation...\n")
    
    user_proxy.initiate_chat(
        assistant,
        message=task
    )
    
    print("\n" + "-"*80)
    print("Two-Agent Chat Complete")
    print("-"*80 + "\n")


# ============================================================================
# Example 2: Group Chat
# ============================================================================

def example_group_chat():
    """
    Multiple agents collaborating in a group conversation.
    GroupChatManager orchestrates the conversation flow.
    """
    print("\n" + "="*80)
    print("Example 2: Group Chat")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    # Create multiple specialized agents
    coder = AssistantAgent(
        name="coder",
        system_message="You are a Python programming expert. You write clean, efficient code. "
                      "Focus on code quality and best practices.",
        llm_config=llm_config
    )
    
    reviewer = AssistantAgent(
        name="reviewer",
        system_message="You are a code reviewer. You analyze code for bugs, performance issues, "
                      "and style problems. Provide constructive feedback.",
        llm_config=llm_config
    )
    
    tester = AssistantAgent(
        name="tester",
        system_message="You are a QA tester. You think about edge cases, test scenarios, "
                      "and potential failures. Suggest comprehensive tests.",
        llm_config=llm_config
    )
    
    # Create UserProxyAgent for code execution
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    # Create GroupChat
    groupchat = GroupChat(
        agents=[coder, reviewer, tester, user_proxy],
        messages=[],
        max_round=12
    )
    
    # Create GroupChatManager
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
        system_message="You are managing a group chat with a coder, reviewer, and tester. "
                      "Coordinate their collaboration to produce high-quality code."
    )
    
    # Start group conversation
    task = """
    Create a Python class for a simple calculator that supports:
    - Addition
    - Subtraction
    - Multiplication
    - Division
    
    The coder should write the code, the reviewer should review it,
    and the tester should suggest test cases.
    """
    
    print(f"Task: {task}\n")
    print("Starting group chat...\n")
    
    user_proxy.initiate_chat(
        manager,
        message=task
    )
    
    print("\n" + "-"*80)
    print("Group Chat Complete")
    print("-"*80 + "\n")


# ============================================================================
# Example 3: Function Registration
# ============================================================================

def example_function_registration():
    """
    Register custom functions for agents to use.
    Functions can be called by AssistantAgent and executed by UserProxyAgent.
    """
    print("\n" + "="*80)
    print("Example 3: Function Registration")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    # Create agents
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant with access to various tools. "
                      "Use the available functions to help solve problems.",
        llm_config=llm_config
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    # Register functions
    
    @user_proxy.register_for_execution(name="calculate")
    @assistant.register_for_llm(name="calculate", description="Calculate a mathematical expression")
    def calculate(expression: str) -> float:
        """
        Calculate a mathematical expression safely.
        
        Args:
            expression: A valid Python mathematical expression (e.g., "2 + 2", "10 * 5")
            
        Returns:
            The result of the calculation
        """
        try:
            # Safe evaluation of mathematical expressions
            allowed_chars = set("0123456789+-*/()., ")
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Expression contains invalid characters")
            result = eval(expression)
            return float(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    @user_proxy.register_for_execution(name="get_weather")
    @assistant.register_for_llm(name="get_weather", description="Get weather information for a city")
    def get_weather(city: str) -> str:
        """
        Get weather information for a city.
        
        Args:
            city: Name of the city
            
        Returns:
            Weather information string
        """
        # Mock weather data
        weather_data = {
            "New York": "Sunny, 72°F",
            "London": "Cloudy, 60°F",
            "Tokyo": "Rainy, 68°F",
            "Paris": "Partly cloudy, 65°F"
        }
        return weather_data.get(city, f"Weather data not available for {city}")
    
    @user_proxy.register_for_execution(name="format_text")
    @assistant.register_for_llm(name="format_text", description="Format text in different styles")
    def format_text(text: str, style: str = "uppercase") -> str:
        """
        Format text in different styles.
        
        Args:
            text: Text to format
            style: Format style (uppercase, lowercase, title, reverse)
            
        Returns:
            Formatted text
        """
        styles = {
            "uppercase": text.upper(),
            "lowercase": text.lower(),
            "title": text.title(),
            "reverse": text[::-1]
        }
        return styles.get(style, text)
    
    # Start conversation
    task = """
    Use the available functions to:
    1. Calculate: (15 * 7) + (23 / 4)
    2. Get weather for New York
    3. Format the text "hello world" in uppercase
    
    Show the results of each function call.
    """
    
    print(f"Task: {task}\n")
    print("Starting conversation with registered functions...\n")
    
    user_proxy.initiate_chat(
        assistant,
        message=task
    )
    
    print("\n" + "-"*80)
    print("Function Registration Example Complete")
    print("-"*80 + "\n")


# ============================================================================
# Example 4: Nested Chat
# ============================================================================

def example_nested_chat():
    """
    Demonstrate nested conversations where an agent initiates a sub-conversation.
    """
    print("\n" + "="*80)
    print("Example 4: Nested Chat")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    # Create main agent
    coordinator = AssistantAgent(
        name="coordinator",
        system_message="You are a project coordinator. When you need specialized help, "
                      "you can initiate nested conversations with specialist agents. "
                      "After getting help, summarize the results.",
        llm_config=llm_config
    )
    
    # Create specialist agent
    math_specialist = AssistantAgent(
        name="math_specialist",
        system_message="You are a mathematics expert. You solve complex mathematical problems "
                      "and explain your reasoning step by step.",
        llm_config=llm_config
    )
    
    # Create UserProxyAgent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=15,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    # Task that requires nested conversation
    task = """
    I need help with a complex problem:
    
    1. First, work with the math specialist to solve: 
       "Find the sum of all prime numbers between 1 and 100"
    
    2. Then, create a Python function to verify the result
    
    3. Finally, summarize what was accomplished
    
    Use nested conversations to collaborate with the specialist.
    """
    
    print(f"Task: {task}\n")
    print("Starting conversation with nested chat capability...\n")
    
    # The coordinator can initiate nested chats
    # In practice, you would implement custom logic to trigger nested chats
    # For this example, we'll show the pattern
    
    user_proxy.initiate_chat(
        coordinator,
        message=task
    )
    
    # Demonstrate nested chat manually
    print("\n--- Initiating Nested Chat with Math Specialist ---\n")
    
    nested_task = "Find the sum of all prime numbers between 1 and 100. Show your work."
    
    user_proxy.initiate_chat(
        math_specialist,
        message=nested_task,
        clear_history=False
    )
    
    print("\n--- Returning to Coordinator ---\n")
    
    user_proxy.send(
        coordinator,
        message="The math specialist has provided the solution. Now create a Python function to verify it."
    )
    
    print("\n" + "-"*80)
    print("Nested Chat Example Complete")
    print("-"*80 + "\n")


# ============================================================================
# Example 5: Human-in-the-Loop
# ============================================================================

def example_human_in_the_loop():
    """
    Demonstrate different human input modes:
    - ALWAYS: Request input after every response
    - NEVER: No human input
    - TERMINATE: Request input only at termination
    """
    print("\n" + "="*80)
    print("Example 5: Human-in-the-Loop")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant that helps with various tasks.",
        llm_config=llm_config
    )
    
    # Example 5a: NEVER mode (fully autonomous)
    print("--- Mode: NEVER (Fully Autonomous) ---\n")
    
    user_proxy_never = UserProxyAgent(
        name="user_proxy_never",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    task_never = "Write a Python function to reverse a string and test it with 'Hello World'"
    
    print(f"Task: {task_never}\n")
    print("Running in NEVER mode (no human input)...\n")
    
    user_proxy_never.initiate_chat(
        assistant,
        message=task_never
    )
    
    print("\n" + "-"*40 + "\n")
    
    # Example 5b: TERMINATE mode (human input at end)
    print("--- Mode: TERMINATE (Human Input at End) ---\n")
    
    user_proxy_terminate = UserProxyAgent(
        name="user_proxy_terminate",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=10,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    task_terminate = """
    Create a Python script that:
    1. Reads a list of numbers
    2. Calculates the mean, median, and mode
    3. Displays the results
    
    Test it with: [1, 2, 3, 4, 5, 5, 6, 7, 8, 9]
    """
    
    print(f"Task: {task_terminate}\n")
    print("Running in TERMINATE mode (human input requested at completion)...\n")
    print("(In interactive mode, you would be prompted for input here)\n")
    
    # Note: In actual usage, TERMINATE mode will prompt for input
    # For this example, we'll simulate it
    user_proxy_terminate.initiate_chat(
        assistant,
        message=task_terminate
    )
    
    print("\n" + "-"*80)
    print("Human-in-the-Loop Example Complete")
    print("Note: ALWAYS mode requires interactive execution to demonstrate")
    print("-"*80 + "\n")


# ============================================================================
# Example 6: Custom Speaker Selection
# ============================================================================

def example_custom_speaker_selection():
    """
    Custom speaker selection logic for group chats.
    Control which agent speaks next based on custom rules.
    """
    print("\n" + "="*80)
    print("Example 6: Custom Speaker Selection")
    print("="*80 + "\n")
    
    llm_config = get_llm_config()
    
    # Create specialized agents
    planner = AssistantAgent(
        name="planner",
        system_message="You are a project planner. You break down tasks into steps and create plans.",
        llm_config=llm_config
    )
    
    implementer = AssistantAgent(
        name="implementer",
        system_message="You are a code implementer. You write code based on plans and specifications.",
        llm_config=llm_config
    )
    
    validator = AssistantAgent(
        name="validator",
        system_message="You are a validator. You check if implementations meet requirements.",
        llm_config=llm_config
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )
    
    # Custom speaker selection function
    def custom_speaker_selection_func(
        last_speaker: ConversableAgent,
        selector: ConversableAgent,
        agents: List[ConversableAgent]
    ) -> ConversableAgent:
        """
        Custom logic: planner -> implementer -> validator -> planner (cycle)
        """
        agent_names = [agent.name for agent in agents if agent.name != "user_proxy"]
        
        if last_speaker.name == "planner":
            next_name = "implementer"
        elif last_speaker.name == "implementer":
            next_name = "validator"
        elif last_speaker.name == "validator":
            next_name = "planner"
        else:
            # Default: start with planner
            next_name = "planner"
        
        # Find and return the agent
        for agent in agents:
            if agent.name == next_name:
                return agent
        
        # Fallback: return first non-user_proxy agent
        return agents[0] if agents[0].name != "user_proxy" else agents[1]
    
    # Create GroupChat with custom speaker selection
    groupchat = GroupChat(
        agents=[planner, implementer, validator, user_proxy],
        messages=[],
        max_round=12,
        speaker_selection_method=custom_speaker_selection_func
    )
    
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
        system_message="You manage a group chat with planner, implementer, and validator. "
                      "Follow the custom speaker selection order."
    )
    
    task = """
    Create a Python function to find the longest common subsequence (LCS) of two strings.
    
    Workflow:
    1. Planner: Create a plan for implementing LCS
    2. Implementer: Write the code
    3. Validator: Verify the implementation
    4. Test with strings "ABCDGH" and "AEDFHR"
    """
    
    print(f"Task: {task}\n")
    print("Starting group chat with custom speaker selection...\n")
    print("Speaker order: planner -> implementer -> validator -> planner (cycle)\n")
    
    user_proxy.initiate_chat(
        manager,
        message=task
    )
    
    print("\n" + "-"*80)
    print("Custom Speaker Selection Example Complete")
    print("-"*80 + "\n")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Run all examples.
    Comment out examples you don't want to run.
    """
    print("\n" + "#"*80)
    print("# AutoGen Multi-Agent Chat Examples")
    print("#"*80)
    
    # Run examples
    try:
        example_two_agent_chat()
    except Exception as e:
        print(f"Error in example_two_agent_chat: {e}\n")
    
    try:
        example_group_chat()
    except Exception as e:
        print(f"Error in example_group_chat: {e}\n")
    
    try:
        example_function_registration()
    except Exception as e:
        print(f"Error in example_function_registration: {e}\n")
    
    try:
        example_nested_chat()
    except Exception as e:
        print(f"Error in example_nested_chat: {e}\n")
    
    try:
        example_human_in_the_loop()
    except Exception as e:
        print(f"Error in example_human_in_the_loop: {e}\n")
    
    try:
        example_custom_speaker_selection()
    except Exception as e:
        print(f"Error in example_custom_speaker_selection: {e}\n")
    
    print("\n" + "#"*80)
    print("# All Examples Complete")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
