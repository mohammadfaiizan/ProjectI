#!/usr/bin/env python3
"""
ReAct Pattern: Reasoning and Acting in AI Agents
==============================================

WHAT IS THE PROBLEM?
==================
Traditional AI systems either:
1. Only reason (think) but can't act in the real world
2. Only act but don't explain their reasoning
3. Make decisions without showing their thought process

This makes it hard to:
- Trust AI decisions
- Debug when things go wrong
- Understand how AI reaches conclusions
- Build educational AI that can teach

REAL WORLD EXAMPLE:
=================
Imagine you're a research assistant helping a student with homework:

Student: "What's the population of Tokyo in 2023 and how does it compare to New York?"

Bad AI: "Tokyo has 37 million, NYC has 8 million" (no reasoning shown)

ReAct AI: 
- THOUGHT: "I need to search for current population data for both cities"
- ACTION: Search for "Tokyo population 2023"
- OBSERVATION: "Found: Tokyo metropolitan area has 37.2 million people"
- THOUGHT: "Now I need NYC data to compare"
- ACTION: Search for "New York City population 2023" 
- OBSERVATION: "Found: NYC has 8.3 million people"
- THOUGHT: "Now I can compare and give a complete answer"
- FINAL ANSWER: "Tokyo (37.2M) is about 4.5 times larger than NYC (8.3M)"

THE REACT ALGORITHM:
==================
1. THINK about what you need to do
2. CHOOSE an action to take
3. OBSERVE the results
4. REPEAT until you have enough information
5. GIVE final answer with reasoning

PSEUDO CODE:
===========
while not solved:
    thought = think_about_current_situation()
    action = decide_what_to_do(thought)
    observation = execute_action(action)
    if observation.contains_answer():
        break
return final_answer_with_reasoning()

WHY IS THIS USEFUL?
==================
- Makes AI transparent and trustworthy
- Easy to debug when wrong
- Great for education and explanation
- Handles complex multi-step problems
- Can use external tools and APIs
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class StepType(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"

@dataclass
class ReActStep:
    """One step in the ReAct reasoning process"""
    step_number: int
    step_type: StepType
    content: str
    confidence: float = 0.8

class SimpleReActAgent:
    """
    A simple ReAct agent that solves problems step by step
    
    EXAMPLE USAGE:
    =============
    agent = SimpleReActAgent()
    result = await agent.solve("What's 15% of 240?")
    
    This will show each step:
    - Thought: I need to calculate 15% of 240
    - Action: Calculate 240 * 0.15  
    - Observation: Result is 36
    - Final Answer: 15% of 240 is 36
    """
    
    def __init__(self):
        self.steps_history = []
        self.tools = {
            "calculator": self.calculator_tool,
            "web_search": self.web_search_tool,
            "text_analyzer": self.text_analyzer_tool
        }
    
    async def solve(self, problem: str) -> Dict[str, Any]:
        """
        Main method to solve any problem using ReAct pattern
        
        Args:
            problem: The question or task to solve
            
        Returns:
            Complete solution with step-by-step reasoning
        """
        print(f"\n🎯 PROBLEM: {problem}")
        print("=" * 50)
        
        self.steps_history = []
        max_steps = 10  # Prevent infinite loops
        
        # Keep reasoning until we have an answer
        for step_num in range(1, max_steps + 1):
            
            # STEP 1: THINK about current situation
            thought = await self.think(problem, step_num)
            self.add_step(step_num, StepType.THOUGHT, thought)
            print(f"💭 THOUGHT {step_num}: {thought}")
            
            # Check if we're ready for final answer
            if "final answer" in thought.lower() or "conclude" in thought.lower():
                final_answer = await self.generate_final_answer(problem)
                self.add_step(step_num, StepType.FINAL_ANSWER, final_answer)
                print(f"✅ FINAL ANSWER: {final_answer}")
                break
            
            # STEP 2: CHOOSE an action
            action = await self.choose_action(thought, problem)
            self.add_step(step_num, StepType.ACTION, action)
            print(f"🔧 ACTION {step_num}: {action}")
            
            # STEP 3: OBSERVE results
            observation = await self.execute_action(action)
            self.add_step(step_num, StepType.OBSERVATION, observation)
            print(f"👁️ OBSERVATION {step_num}: {observation}")
            
            # Small delay to make it readable
            await asyncio.sleep(0.1)
        
        return {
            "problem": problem,
            "steps": self.steps_history,
            "total_steps": len(self.steps_history),
            "solved": any(step.step_type == StepType.FINAL_ANSWER for step in self.steps_history)
        }
    
    async def think(self, problem: str, step_number: int) -> str:
        """
        Generate reasoning about what to do next
        
        This simulates human-like thinking about the problem
        """
        context = self.get_current_context()
        
        # First step - analyze the problem
        if step_number == 1:
            if any(word in problem.lower() for word in ["calculate", "math", "+", "-", "*", "/", "%"]):
                return "I need to solve a math problem. Let me identify the calculation needed."
            elif any(word in problem.lower() for word in ["search", "find", "what is", "who is", "when"]):
                return "This is a knowledge question. I need to search for information."
            elif any(word in problem.lower() for word in ["analyze", "compare", "summarize"]):
                return "This requires analysis. Let me break down what needs to be examined."
            else:
                return "Let me understand what type of problem this is and what approach to take."
        
        # Later steps - build on what we know
        elif step_number == 2:
            return "Based on my analysis, let me take the appropriate action to gather information."
        elif step_number == 3:
            if context and "result" in context.lower():
                return "I have some results. Let me check if this answers the question or if I need more information."
            else:
                return "Let me try a different approach to get the information I need."
        else:
            return "Let me review what I've learned and see if I can provide a complete answer now."
    
    async def choose_action(self, thought: str, problem: str) -> str:
        """
        Decide what action to take based on current thinking
        """
        thought_lower = thought.lower()
        problem_lower = problem.lower()
        
        # Math problems
        if any(word in thought_lower for word in ["calculate", "math", "solve"]):
            # Extract numbers and operators from problem
            import re
            math_expression = self.extract_math_expression(problem)
            return f"Use calculator to compute: {math_expression}"
        
        # Search problems  
        elif any(word in thought_lower for word in ["search", "find", "information"]):
            search_terms = self.extract_search_terms(problem)
            return f"Search the web for: {search_terms}"
        
        # Analysis problems
        elif any(word in thought_lower for word in ["analyze", "examine", "compare"]):
            return f"Analyze the text: {problem}"
        
        else:
            return f"Research the topic: {problem}"
    
    async def execute_action(self, action: str) -> str:
        """
        Actually perform the chosen action
        """
        action_lower = action.lower()
        
        # Calculator actions
        if "calculator" in action_lower or "compute" in action_lower:
            # Extract the calculation part
            if ":" in action:
                calculation = action.split(":", 1)[1].strip()
                return await self.tools["calculator"](calculation)
        
        # Search actions
        elif "search" in action_lower:
            if ":" in action:
                search_query = action.split(":", 1)[1].strip()
                return await self.tools["web_search"](search_query)
        
        # Analysis actions
        elif "analyze" in action_lower:
            if ":" in action:
                text_to_analyze = action.split(":", 1)[1].strip()
                return await self.tools["text_analyzer"](text_to_analyze)
        
        return "Action completed, but no specific result obtained."
    
    async def generate_final_answer(self, problem: str) -> str:
        """
        Create the final answer based on all observations
        """
        # Collect all observations
        observations = [
            step.content for step in self.steps_history 
            if step.step_type == StepType.OBSERVATION
        ]
        
        if not observations:
            return "I couldn't find enough information to answer this question."
        
        # Combine observations into a coherent answer
        combined_info = " ".join(observations)
        
        return f"Based on my research and calculations: {combined_info}"
    
    # TOOL IMPLEMENTATIONS
    # ===================
    
    async def calculator_tool(self, expression: str) -> str:
        """Simple calculator tool"""
        try:
            # Clean up the expression
            cleaned = expression.replace(" ", "")
            
            # Handle percentage calculations
            if "%" in cleaned:
                if "of" in expression.lower():
                    # Handle "X% of Y" format
                    parts = expression.lower().split("of")
                    if len(parts) == 2:
                        percent_part = parts[0].replace("%", "").strip()
                        number_part = parts[1].strip()
                        try:
                            percent = float(percent_part)
                            number = float(number_part)
                            result = (percent / 100) * number
                            return f"Calculation: {percent}% of {number} = {result}"
                        except:
                            pass
            
            # Handle basic math expressions
            # Only allow safe characters
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in cleaned):
                result = eval(cleaned)
                return f"Calculation result: {cleaned} = {result}"
            else:
                return f"Cannot calculate '{expression}' - contains invalid characters"
                
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    async def web_search_tool(self, query: str) -> str:
        """Simulated web search tool"""
        # Simulate search delay
        await asyncio.sleep(0.1)
        
        # Simple knowledge base for demo
        knowledge = {
            "tokyo population": "Tokyo metropolitan area has approximately 37.2 million people as of 2023",
            "new york population": "New York City has approximately 8.3 million people as of 2023",
            "python programming": "Python is a high-level programming language created by Guido van Rossum in 1991",
            "ai agents": "AI agents are autonomous software entities that can perceive, reason, and act in their environment",
            "machine learning": "Machine learning is a subset of AI that enables computers to learn without explicit programming"
        }
        
        query_lower = query.lower()
        
        # Find matching knowledge
        for key, value in knowledge.items():
            if any(word in query_lower for word in key.split()):
                return f"Search result: {value}"
        
        return f"Search completed for '{query}' - found general information available"
    
    async def text_analyzer_tool(self, text: str) -> str:
        """Simple text analysis tool"""
        word_count = len(text.split())
        char_count = len(text)
        
        # Simple sentiment analysis
        positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "horrible", "worst"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return f"Text analysis: {word_count} words, {char_count} characters, sentiment appears {sentiment}"
    
    # HELPER METHODS
    # =============
    
    def extract_math_expression(self, problem: str) -> str:
        """Extract mathematical expression from problem text"""
        import re
        
        # Look for percentage calculations
        percent_match = re.search(r'(\d+)%\s+of\s+(\d+)', problem)
        if percent_match:
            return f"{percent_match.group(1)}% of {percent_match.group(2)}"
        
        # Look for basic math expressions
        math_match = re.search(r'[\d\+\-\*/\(\)\s]+', problem)
        if math_match:
            return math_match.group().strip()
        
        return problem
    
    def extract_search_terms(self, problem: str) -> str:
        """Extract search terms from problem"""
        # Remove question words
        stop_words = ["what", "is", "the", "how", "when", "where", "why", "who"]
        words = problem.lower().split()
        search_words = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(search_words[:4])  # Top 4 relevant words
    
    def get_current_context(self) -> str:
        """Get summary of what we know so far"""
        observations = [
            step.content for step in self.steps_history 
            if step.step_type == StepType.OBSERVATION
        ]
        return " ".join(observations)
    
    def add_step(self, step_number: int, step_type: StepType, content: str):
        """Add a step to our reasoning history"""
        step = ReActStep(step_number, step_type, content)
        self.steps_history.append(step)

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_math_problem():
    """Demo: Solving a math problem"""
    print("\n" + "="*60)
    print("DEMO 1: MATH PROBLEM")
    print("="*60)
    
    agent = SimpleReActAgent()
    await agent.solve("What is 15% of 240?")

async def demo_research_question():
    """Demo: Answering a research question"""
    print("\n" + "="*60)
    print("DEMO 2: RESEARCH QUESTION")  
    print("="*60)
    
    agent = SimpleReActAgent()
    await agent.solve("What is Python programming language?")

async def demo_comparison_question():
    """Demo: Complex comparison question"""
    print("\n" + "="*60)
    print("DEMO 3: COMPARISON QUESTION")
    print("="*60)
    
    agent = SimpleReActAgent()
    await agent.solve("How does Tokyo population compare to New York?")

async def main():
    """
    Run all demonstrations to show ReAct pattern in action
    
    WHAT YOU'LL LEARN:
    ================
    1. How AI can show its reasoning process
    2. Why step-by-step thinking is important
    3. How to combine thinking with actions
    4. How to handle different types of problems
    5. Why transparency matters in AI systems
    
    REAL WORLD APPLICATIONS:
    =======================
    - Educational AI tutors that explain their reasoning
    - Research assistants that show their work
    - Debugging tools that trace AI decision making  
    - Customer service bots that explain their answers
    - Medical AI that shows diagnostic reasoning
    """
    
    print("🚀 ReAct Pattern Demonstration")
    print("This shows how AI agents can think step-by-step like humans!")
    
    await demo_math_problem()
    await demo_research_question() 
    await demo_comparison_question()
    
    print("\n" + "="*60)
    print("🎓 WHAT WE LEARNED:")
    print("="*60)
    print("✅ ReAct makes AI thinking transparent")
    print("✅ Each step builds on the previous one")
    print("✅ Actions provide new information")
    print("✅ Observations guide next steps")
    print("✅ Final answers are well-reasoned")
    print("\n🔧 TRY IT YOURSELF:")
    print("- Modify the tools to add new capabilities")
    print("- Add more complex reasoning patterns")
    print("- Connect to real APIs instead of simulated ones")
    print("- Use with actual LLMs for better reasoning")

if __name__ == "__main__":
    asyncio.run(main())
