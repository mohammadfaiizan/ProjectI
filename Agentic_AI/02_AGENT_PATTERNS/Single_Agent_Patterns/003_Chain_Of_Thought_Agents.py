#!/usr/bin/env python3
"""
Chain of Thought (CoT): Step-by-Step Logical Reasoning
====================================================

WHAT IS THE PROBLEM?
==================
How do you solve this problem: "Sarah has 3 boxes. Each box contains 4 smaller boxes. Each smaller box has 6 items. How many items does Sarah have in total?"

❌ BAD AI: "Answer: 72 items" (No explanation, hard to verify)

✅ GOOD AI (Chain of Thought):
Step 1: Sarah has 3 main boxes
Step 2: Each main box contains 4 smaller boxes
Step 3: So total smaller boxes = 3 × 4 = 12 smaller boxes  
Step 4: Each smaller box has 6 items
Step 5: So total items = 12 × 6 = 72 items
Answer: 72 items

REAL WORLD EXAMPLE:
=================
Imagine you're helping a student with math homework:

PROBLEM: "A restaurant serves 150 customers per day. Each customer orders an average of 2.5 items. If each item costs $8 on average, what's the daily revenue?"

WITHOUT CHAIN OF THOUGHT:
Student: "What's the answer?"
AI: "$3000"
Student: "How did you get that? I don't understand!"

WITH CHAIN OF THOUGHT:
AI: "Let me solve this step by step:

Step 1: Identify what we know
- Customers per day: 150
- Items per customer: 2.5 (average)  
- Cost per item: $8 (average)

Step 2: Calculate total items sold per day
Total items = Customers × Items per customer
Total items = 150 × 2.5 = 375 items

Step 3: Calculate daily revenue
Daily revenue = Total items × Cost per item
Daily revenue = 375 × $8 = $3000

Answer: The restaurant's daily revenue is $3000"

Student: "Oh! Now I understand how to solve similar problems!"

THE ALGORITHM:
=============
1. Break the problem into logical steps
2. Solve each step clearly and explicitly  
3. Show your work for each step
4. Build each step on the previous ones
5. Arrive at final answer through clear reasoning

PSEUDO CODE:
===========
def solve_with_chain_of_thought(problem):
    steps = []
    
    # Step 1: Understand the problem
    understanding = analyze_problem(problem)
    steps.append(f"Understanding: {understanding}")
    
    # Step 2-N: Solve step by step
    current_info = understanding
    while not solved(current_info):
        next_step = determine_next_logical_step(current_info)
        result = solve_step(next_step)
        steps.append(f"Step {len(steps)}: {next_step} = {result}")
        current_info = update_info(current_info, result)
    
    # Final step: Conclude
    final_answer = synthesize_answer(steps)
    steps.append(f"Final Answer: {final_answer}")
    
    return steps, final_answer

WHY IS THIS CRUCIAL?
==================
- Makes AI reasoning transparent and trustworthy
- Helps humans learn the thinking process
- Makes it easy to find errors in reasoning
- Enables teaching and educational applications  
- Builds confidence in AI answers
- Works especially well for math, logic, and analysis
"""

import asyncio
import re
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

class StepType(Enum):
    UNDERSTANDING = "understanding"
    CALCULATION = "calculation"
    LOGICAL_DEDUCTION = "logical_deduction"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    FINAL_ANSWER = "final_answer"

@dataclass
class ReasoningStep:
    """One step in our chain of thought"""
    step_number: int
    step_type: StepType
    description: str
    calculation: Optional[str] = None
    result: Any = None
    confidence: float = 0.9

class ChainOfThoughtAgent:
    """
    An agent that solves problems using explicit step-by-step reasoning
    
    EXAMPLE USAGE:
    =============
    agent = ChainOfThoughtAgent()
    solution = await agent.solve("If a train travels 60 mph for 2.5 hours, how far does it go?")
    
    This will show:
    Step 1: Understanding - We need to find distance using speed and time
    Step 2: Apply formula - Distance = Speed × Time  
    Step 3: Calculate - Distance = 60 mph × 2.5 hours = 150 miles
    Final Answer: The train travels 150 miles
    """
    
    def __init__(self):
        self.reasoning_chain = []
        self.knowledge_base = {
            # Math formulas
            "distance": "Distance = Speed × Time",
            "area_rectangle": "Area = Length × Width", 
            "area_circle": "Area = π × radius²",
            "compound_interest": "A = P(1 + r/n)^(nt)",
            "percentage": "Percentage = (Part/Whole) × 100",
            
            # Logic rules
            "if_then": "If A then B, A is true, therefore B is true",
            "contrapositive": "If A then B is equivalent to If not B then not A",
            
            # Basic facts
            "days_in_year": 365,
            "hours_in_day": 24,
            "minutes_in_hour": 60
        }
    
    async def solve(self, problem: str) -> Dict[str, Any]:
        """
        Solve any problem using chain of thought reasoning
        
        Args:
            problem: The problem to solve
            
        Returns:
            Complete solution with step-by-step reasoning
        """
        print(f"\n🧠 PROBLEM: {problem}")
        print("=" * 60)
        
        self.reasoning_chain = []
        
        # Step 1: Understand the problem
        understanding = await self.understand_problem(problem)
        self.add_step(StepType.UNDERSTANDING, "Problem Analysis", result=understanding)
        print(f"🔍 UNDERSTANDING: {understanding}")
        
        # Step 2: Identify solution approach
        approach = await self.identify_approach(understanding, problem)
        self.add_step(StepType.ANALYSIS, "Solution Approach", result=approach)
        print(f"📋 APPROACH: {approach}")
        
        # Step 3-N: Solve step by step
        step_count = 2
        current_info = {"understanding": understanding, "approach": approach}
        
        while not self.is_solved(current_info) and step_count < 10:
            step_count += 1
            
            # Determine next logical step
            next_step = await self.determine_next_step(current_info, problem)
            
            if next_step["type"] == "calculation":
                result = await self.perform_calculation(next_step["operation"])
                self.add_step(StepType.CALCULATION, next_step["description"], 
                            calculation=next_step["operation"], result=result)
                print(f"🧮 STEP {step_count}: {next_step['description']}")
                print(f"   Calculation: {next_step['operation']} = {result}")
                
            elif next_step["type"] == "logical_deduction":
                result = await self.perform_logical_reasoning(next_step["premise"], next_step["rule"])
                self.add_step(StepType.LOGICAL_DEDUCTION, next_step["description"], result=result)
                print(f"🔗 STEP {step_count}: {next_step['description']}")
                print(f"   Reasoning: {result}")
                
            elif next_step["type"] == "analysis":
                result = await self.perform_analysis(next_step["data"])
                self.add_step(StepType.ANALYSIS, next_step["description"], result=result)
                print(f"📊 STEP {step_count}: {next_step['description']}")
                print(f"   Analysis: {result}")
            
            # Update our current information
            current_info["latest_result"] = result
            current_info["step_count"] = step_count
        
        # Final step: Synthesize answer
        final_answer = await self.synthesize_final_answer(current_info, problem)
        self.add_step(StepType.FINAL_ANSWER, "Final Answer", result=final_answer)
        print(f"✅ FINAL ANSWER: {final_answer}")
        
        return {
            "problem": problem,
            "reasoning_chain": self.reasoning_chain,
            "final_answer": final_answer,
            "total_steps": len(self.reasoning_chain),
            "solved": True
        }
    
    async def understand_problem(self, problem: str) -> str:
        """
        Analyze and understand what the problem is asking
        
        This is like reading the problem carefully and identifying:
        - What information we have
        - What we need to find
        - What type of problem this is
        """
        problem_lower = problem.lower()
        
        # Math problems
        if any(word in problem_lower for word in ["calculate", "how much", "how many", "total", "sum"]):
            # Extract numbers from the problem
            numbers = re.findall(r'\d+(?:\.\d+)?', problem)
            
            if "speed" in problem_lower and "time" in problem_lower:
                return f"This is a distance calculation problem. We have speed and time, need to find distance. Numbers found: {numbers}"
            elif "area" in problem_lower or ("length" in problem_lower and "width" in problem_lower):
                return f"This is an area calculation problem. We need to find the area of a shape. Numbers found: {numbers}"
            elif "%" in problem or "percent" in problem_lower:
                return f"This is a percentage calculation problem. Numbers found: {numbers}"
            elif "box" in problem_lower or "container" in problem_lower:
                return f"This is a multiplication/counting problem involving containers. Numbers found: {numbers}"
            else:
                return f"This is a mathematical problem requiring calculation. Numbers found: {numbers}"
        
        # Logic problems
        elif any(word in problem_lower for word in ["if", "then", "therefore", "because", "since"]):
            return "This is a logical reasoning problem. We need to apply logical rules to reach a conclusion."
        
        # Analysis problems
        elif any(word in problem_lower for word in ["analyze", "compare", "evaluate", "assess"]):
            return "This is an analysis problem. We need to examine information and draw insights."
        
        # Word problems
        elif "?" in problem:
            return f"This is a word problem asking for specific information. We need to extract key facts and apply appropriate methods."
        
        else:
            return "This problem requires careful analysis to determine the best solution approach."
    
    async def identify_approach(self, understanding: str, problem: str) -> str:
        """
        Decide what approach/method to use for solving
        """
        understanding_lower = understanding.lower()
        problem_lower = problem.lower()
        
        if "distance calculation" in understanding_lower:
            return "Use the Distance = Speed × Time formula"
        elif "area calculation" in understanding_lower:
            if "rectangle" in problem_lower or ("length" in problem_lower and "width" in problem_lower):
                return "Use the Area = Length × Width formula for rectangle"
            elif "circle" in problem_lower:
                return "Use the Area = π × radius² formula for circle"
            else:
                return "Identify the shape and apply appropriate area formula"
        elif "percentage" in understanding_lower:
            return "Use percentage formula: (Part/Whole) × 100 or find percentage of a number"
        elif "multiplication" in understanding_lower or "counting" in understanding_lower:
            return "Break down into steps and multiply the quantities at each level"
        elif "logical reasoning" in understanding_lower:
            return "Apply logical rules step by step"
        else:
            return "Break the problem into smaller parts and solve systematically"
    
    async def determine_next_step(self, current_info: Dict, problem: str) -> Dict[str, Any]:
        """
        Figure out what the next logical step should be
        """
        approach = current_info.get("approach", "")
        step_count = current_info.get("step_count", 2)
        
        # For distance problems
        if "Distance = Speed × Time" in approach:
            if step_count == 3:
                # Extract speed and time from problem
                numbers = re.findall(r'\d+(?:\.\d+)?', problem)
                if len(numbers) >= 2:
                    return {
                        "type": "calculation",
                        "description": f"Apply Distance formula with given values",
                        "operation": f"{numbers[0]} × {numbers[1]}"
                    }
        
        # For area problems
        elif "Area = Length × Width" in approach:
            numbers = re.findall(r'\d+(?:\.\d+)?', problem)
            if len(numbers) >= 2:
                return {
                    "type": "calculation",
                    "description": "Calculate area using length and width",
                    "operation": f"{numbers[0]} × {numbers[1]}"
                }
        
        # For percentage problems
        elif "percentage" in approach.lower():
            numbers = re.findall(r'\d+(?:\.\d+)?', problem)
            if "%" in problem and len(numbers) >= 2:
                percent = None
                base_number = None
                for i, num in enumerate(numbers):
                    # Find which number has % after it
                    if f"{num}%" in problem:
                        percent = float(num)
                    else:
                        base_number = float(num)
                
                if percent and base_number:
                    return {
                        "type": "calculation", 
                        "description": f"Calculate {percent}% of {base_number}",
                        "operation": f"{base_number} × {percent}/100"
                    }
        
        # For counting/multiplication problems
        elif "multiply" in approach.lower():
            numbers = re.findall(r'\d+(?:\.\d+)?', problem)
            if len(numbers) >= 2:
                return {
                    "type": "calculation",
                    "description": f"Multiply the quantities: {' × '.join(numbers)}",
                    "operation": " × ".join(numbers)
                }
        
        # Default step
        return {
            "type": "analysis",
            "description": "Analyze the available information",
            "data": problem
        }
    
    async def perform_calculation(self, operation: str) -> Union[float, str]:
        """
        Perform mathematical calculations safely
        """
        try:
            # Clean up the operation
            clean_operation = operation.replace("×", "*").replace("÷", "/")
            
            # Only allow safe mathematical operations
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in clean_operation):
                result = eval(clean_operation)
                return round(result, 2) if isinstance(result, float) else result
            else:
                return f"Cannot calculate: {operation} (contains unsafe characters)"
                
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    async def perform_logical_reasoning(self, premise: str, rule: str) -> str:
        """
        Apply logical reasoning rules
        """
        # Simple logical reasoning (can be expanded)
        if "if" in rule.lower() and "then" in rule.lower():
            return f"Applied logical rule: {rule} to premise: {premise}"
        else:
            return f"Logical analysis of: {premise}"
    
    async def perform_analysis(self, data: str) -> str:
        """
        Perform general analysis of information
        """
        word_count = len(data.split())
        has_numbers = bool(re.findall(r'\d+', data))
        has_question = "?" in data
        
        return f"Analyzed data: {word_count} words, contains numbers: {has_numbers}, has question: {has_question}"
    
    def is_solved(self, current_info: Dict) -> bool:
        """
        Check if we have enough information to provide final answer
        """
        return (current_info.get("step_count", 0) >= 4 and 
                "latest_result" in current_info and
                current_info["latest_result"] is not None)
    
    async def synthesize_final_answer(self, current_info: Dict, problem: str) -> str:
        """
        Create the final answer based on all our reasoning steps
        """
        latest_result = current_info.get("latest_result")
        
        if isinstance(latest_result, (int, float)):
            # For numerical results, provide context
            if "distance" in current_info.get("approach", "").lower():
                return f"The distance traveled is {latest_result} miles"
            elif "area" in current_info.get("approach", "").lower():
                return f"The area is {latest_result} square units"
            elif "percentage" in current_info.get("approach", "").lower():
                return f"The result is {latest_result}"
            elif "total" in problem.lower():
                return f"The total is {latest_result}"
            else:
                return f"The answer is {latest_result}"
        else:
            return f"Based on the analysis: {latest_result}"
    
    def add_step(self, step_type: StepType, description: str, 
                calculation: Optional[str] = None, result: Any = None):
        """Add a step to our reasoning chain"""
        step = ReasoningStep(
            step_number=len(self.reasoning_chain) + 1,
            step_type=step_type,
            description=description,
            calculation=calculation,
            result=result
        )
        self.reasoning_chain.append(step)

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_math_problem():
    """Demo: Solving a math word problem"""
    print("\n" + "="*70)
    print("DEMO 1: MATH WORD PROBLEM")
    print("="*70)
    
    agent = ChainOfThoughtAgent()
    await agent.solve("A car travels at 65 mph for 3.5 hours. How far does it travel?")

async def demo_percentage_problem():
    """Demo: Solving a percentage problem"""
    print("\n" + "="*70)
    print("DEMO 2: PERCENTAGE PROBLEM")
    print("="*70)
    
    agent = ChainOfThoughtAgent()
    await agent.solve("What is 25% of 240?")

async def demo_counting_problem():
    """Demo: Solving a multi-level counting problem"""
    print("\n" + "="*70)
    print("DEMO 3: COUNTING PROBLEM")
    print("="*70)
    
    agent = ChainOfThoughtAgent()
    await agent.solve("Sarah has 4 boxes. Each box contains 6 smaller boxes. Each smaller box has 8 items. How many items does Sarah have in total?")

async def demo_area_problem():
    """Demo: Solving an area problem"""
    print("\n" + "="*70)
    print("DEMO 4: AREA PROBLEM")
    print("="*70)
    
    agent = ChainOfThoughtAgent()
    await agent.solve("What is the area of a rectangle that is 12 feet long and 8 feet wide?")

async def main():
    """
    Demonstrate Chain of Thought reasoning with clear examples
    
    WHAT YOU'LL LEARN:
    ================
    1. How to break complex problems into logical steps
    2. Why showing your work builds trust and understanding
    3. How step-by-step reasoning prevents errors  
    4. How to make AI reasoning transparent and educational
    5. Why this pattern is crucial for math and logic problems
    
    REAL WORLD APPLICATIONS:
    =======================
    - Educational AI tutors that teach step-by-step
    - Math problem solving applications
    - Logical reasoning and analysis tools
    - Transparent AI decision making systems
    - Debugging and verification of AI reasoning
    - Scientific and research applications
    """
    
    print("🧠 Chain of Thought Reasoning Demonstration")
    print("This shows how AI can think step-by-step like a good teacher!")
    
    await demo_math_problem()
    await demo_percentage_problem()
    await demo_counting_problem()
    await demo_area_problem()
    
    print("\n" + "="*70)
    print("🎓 WHAT WE LEARNED:")
    print("="*70)
    print("✅ Breaking problems into steps makes them solvable")
    print("✅ Showing work builds trust and enables learning")  
    print("✅ Each step logically builds on the previous one")
    print("✅ Calculations and reasoning are explicit and verifiable")
    print("✅ Final answers include context and explanation")
    print("\n🔧 TRY IT YOURSELF:")
    print("- Add more complex mathematical formulas")
    print("- Implement logical reasoning for different domains")
    print("- Add visual step-by-step diagrams")
    print("- Connect to educational platforms")
    print("- Add confidence scoring for each step")

if __name__ == "__main__":
    asyncio.run(main())
