#!/usr/bin/env python3
"""
Self Reflection Agents: Learning from Mistakes and Improving
==========================================================

WHAT IS THE PROBLEM?
==================
Most systems make mistakes and never learn from them. They repeat the same errors over and over.

Example: Bad GPS Navigation System
- Takes you the same bad route every day
- Never learns that this route has traffic jams
- Never improves its recommendations
- Keeps making the same routing mistakes

REAL WORLD EXAMPLE:
=================
Imagine you're learning to play chess:

WITHOUT SELF REFLECTION:
- Play a game and lose
- Start next game without thinking about mistakes
- Make the same mistakes repeatedly
- Never improve your gameplay

WITH SELF REFLECTION:
- Play a game and lose
- REFLECT: "What went wrong? I lost my queen early"
- ANALYZE: "I moved my queen too aggressively without protection"
- LEARN: "Next time, develop pieces before attacking"
- IMPROVE: Apply this lesson in the next game
- RESULT: Gradually get better at chess

THE ALGORITHM:
=============
1. EXECUTE: Perform a task or solve a problem
2. EVALUATE: Assess how well you performed
3. REFLECT: Identify what went wrong and what went right
4. LEARN: Extract general lessons from the experience
5. ADAPT: Modify your approach for future similar tasks
6. REPEAT: Apply improved approach to next task

PSEUDO CODE:
===========
def self_reflecting_agent():
    experience_memory = []
    learned_rules = []
    
    while True:
        # Execute task
        result = execute_task(current_task)
        
        # Evaluate performance
        performance = evaluate_result(result, expected_outcome)
        
        # Reflect on what happened
        reflection = analyze_experience(result, performance, context)
        
        # Learn from reflection
        new_lessons = extract_lessons(reflection)
        learned_rules.extend(new_lessons)
        
        # Store experience for future reference
        experience_memory.append({
            'task': current_task,
            'result': result,
            'reflection': reflection,
            'lessons': new_lessons
        })
        
        # Adapt approach for next time
        approach = adapt_approach(learned_rules, next_task)

WHY IS THIS CRUCIAL?
==================
- Prevents repeating the same mistakes
- Enables continuous improvement over time
- Builds up wisdom from experience
- Makes agents more reliable and effective
- Enables learning without explicit training
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class PerformanceLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    FAILED = "failed"

class LessonType(Enum):
    STRATEGY = "strategy"
    APPROACH = "approach"
    MISTAKE_AVOIDANCE = "mistake_avoidance"
    OPTIMIZATION = "optimization"
    GENERAL_PRINCIPLE = "general_principle"

@dataclass
class TaskExecution:
    """Record of executing a specific task"""
    task_id: str
    task_type: str
    task_description: str
    approach_used: str
    result: Any
    success: bool
    execution_time: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class PerformanceEvaluation:
    """Evaluation of how well a task was performed"""
    execution_id: str
    performance_level: PerformanceLevel
    success_rate: float
    efficiency_score: float
    quality_score: float
    areas_for_improvement: List[str]
    what_went_well: List[str]

@dataclass
class Reflection:
    """Analysis of what happened and why"""
    execution_id: str
    what_happened: str
    why_it_happened: str
    what_could_be_better: str
    root_causes: List[str]
    contributing_factors: List[str]

@dataclass
class Lesson:
    """A learned principle that can be applied to future tasks"""
    lesson_id: str
    lesson_type: LessonType
    content: str
    confidence: float
    applicable_to: List[str]  # Task types this applies to
    learned_from: str  # Which execution taught this
    times_applied: int = 0
    success_when_applied: int = 0

class SelfReflectionAgent:
    """
    An agent that learns from its experiences through self-reflection
    
    EXAMPLE USAGE:
    =============
    agent = SelfReflectionAgent()
    
    # Agent performs tasks and learns from each one
    result1 = await agent.perform_task("solve_math_problem", "Calculate 15% of 240")
    result2 = await agent.perform_task("solve_math_problem", "Calculate 20% of 150")
    
    # Agent gets better over time by learning from mistakes
    agent.show_learning_progress()
    """
    
    def __init__(self):
        self.execution_history: List[TaskExecution] = []
        self.evaluation_history: List[PerformanceEvaluation] = []
        self.reflection_history: List[Reflection] = []
        self.learned_lessons: List[Lesson] = []
        self.current_strategies: Dict[str, str] = {}
        
        # Initialize with some basic problem-solving strategies
        self.current_strategies = {
            "math_problem": "break_into_steps",
            "research_task": "search_and_synthesize",
            "planning_task": "top_down_breakdown",
            "creative_task": "brainstorm_then_refine"
        }
    
    async def perform_task(self, task_type: str, task_description: str) -> Dict[str, Any]:
        """
        Perform a task while learning from the experience
        
        This is the main method that executes, evaluates, reflects, and learns
        """
        print(f"\nTASK: {task_description}")
        print("=" * 50)
        
        # Step 1: Choose approach based on learned lessons
        approach = self.choose_approach(task_type, task_description)
        print(f"APPROACH: Using '{approach}' strategy")
        
        # Step 2: Execute the task
        execution = await self.execute_task(task_type, task_description, approach)
        self.execution_history.append(execution)
        print(f"EXECUTION: {'SUCCESS' if execution.success else 'FAILED'} in {execution.execution_time:.2f}s")
        
        # Step 3: Evaluate performance
        evaluation = self.evaluate_performance(execution)
        self.evaluation_history.append(evaluation)
        print(f"EVALUATION: {evaluation.performance_level.value} (Quality: {evaluation.quality_score:.2f})")
        
        # Step 4: Reflect on what happened
        reflection = await self.reflect_on_experience(execution, evaluation)
        self.reflection_history.append(reflection)
        print(f"REFLECTION: {reflection.what_happened}")
        
        # Step 5: Learn lessons for the future
        new_lessons = self.extract_lessons(execution, evaluation, reflection)
        self.learned_lessons.extend(new_lessons)
        if new_lessons:
            print(f"LEARNED: {len(new_lessons)} new lessons")
            for lesson in new_lessons:
                print(f"  - {lesson.content}")
        
        # Step 6: Update strategies for future tasks
        self.update_strategies(task_type, execution, evaluation, new_lessons)
        
        return {
            "task_description": task_description,
            "result": execution.result,
            "success": execution.success,
            "performance_level": evaluation.performance_level.value,
            "lessons_learned": len(new_lessons),
            "total_lessons": len(self.learned_lessons)
        }
    
    def choose_approach(self, task_type: str, task_description: str) -> str:
        """
        Choose the best approach based on past experience and learned lessons
        """
        # Check if we have learned lessons for this type of task
        relevant_lessons = [
            lesson for lesson in self.learned_lessons
            if task_type in lesson.applicable_to and lesson.confidence > 0.6
        ]
        
        if relevant_lessons:
            # Use the most confident lesson
            best_lesson = max(relevant_lessons, key=lambda l: l.confidence)
            return f"learned_strategy_from_{best_lesson.lesson_id}"
        
        # Use default strategy for this task type
        return self.current_strategies.get(task_type, "general_problem_solving")
    
    async def execute_task(self, task_type: str, task_description: str, approach: str) -> TaskExecution:
        """
        Actually execute the task using the chosen approach
        """
        start_time = time.time()
        task_id = f"task_{len(self.execution_history) + 1}"
        
        try:
            # Simulate different types of task execution
            if task_type == "math_problem":
                result = await self.solve_math_problem(task_description, approach)
                success = "error" not in str(result).lower()
                
            elif task_type == "research_task":
                result = await self.perform_research(task_description, approach)
                success = len(str(result)) > 50  # Reasonable amount of content
                
            elif task_type == "planning_task":
                result = await self.create_plan(task_description, approach)
                success = isinstance(result, dict) and "steps" in result
                
            elif task_type == "creative_task":
                result = await self.generate_creative_content(task_description, approach)
                success = len(str(result)) > 20
                
            else:
                result = await self.handle_generic_task(task_description, approach)
                success = result is not None
            
            execution_time = time.time() - start_time
            
            return TaskExecution(
                task_id=task_id,
                task_type=task_type,
                task_description=task_description,
                approach_used=approach,
                result=result,
                success=success,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TaskExecution(
                task_id=task_id,
                task_type=task_type,
                task_description=task_description,
                approach_used=approach,
                result=f"Error: {str(e)}",
                success=False,
                execution_time=execution_time
            )
    
    async def solve_math_problem(self, problem: str, approach: str) -> str:
        """Solve mathematical problems with different approaches"""
        await asyncio.sleep(0.1)  # Simulate thinking time
        
        if "%" in problem:
            # Percentage calculation
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', problem)
            if len(numbers) >= 2:
                if "learned_strategy" in approach:
                    # Use improved method from learning
                    return f"Step 1: Identify {numbers[0]}% of {numbers[1]}\nStep 2: Calculate {numbers[0]}/100 * {numbers[1]} = {float(numbers[0])/100 * float(numbers[1])}"
                else:
                    # Basic method
                    result = float(numbers[0])/100 * float(numbers[1])
                    return f"{numbers[0]}% of {numbers[1]} = {result}"
        
        return "Mathematical solution provided"
    
    async def perform_research(self, topic: str, approach: str) -> str:
        """Research topics with different approaches"""
        await asyncio.sleep(0.2)
        
        if "learned_strategy" in approach:
            return f"Comprehensive research on {topic}: Found multiple sources, cross-referenced information, identified key insights and potential applications."
        else:
            return f"Basic research on {topic}: Found general information about the topic."
    
    async def create_plan(self, goal: str, approach: str) -> Dict[str, Any]:
        """Create plans with different approaches"""
        await asyncio.sleep(0.15)
        
        if "learned_strategy" in approach:
            return {
                "goal": goal,
                "steps": ["Detailed analysis", "Resource identification", "Timeline creation", "Risk assessment", "Implementation plan"],
                "quality": "high_detail"
            }
        else:
            return {
                "goal": goal,
                "steps": ["Step 1", "Step 2", "Step 3"],
                "quality": "basic"
            }
    
    async def generate_creative_content(self, prompt: str, approach: str) -> str:
        """Generate creative content with different approaches"""
        await asyncio.sleep(0.1)
        
        if "learned_strategy" in approach:
            return f"Creative response to '{prompt}': Multiple innovative ideas with detailed exploration and practical applications."
        else:
            return f"Creative response to '{prompt}': Basic creative ideas."
    
    async def handle_generic_task(self, description: str, approach: str) -> str:
        """Handle any other type of task"""
        await asyncio.sleep(0.1)
        return f"Completed task: {description}"
    
    def evaluate_performance(self, execution: TaskExecution) -> PerformanceEvaluation:
        """
        Evaluate how well the task was performed
        """
        # Base evaluation on success, time, and result quality
        if not execution.success:
            performance_level = PerformanceLevel.FAILED
            success_rate = 0.0
            quality_score = 0.2
        else:
            # Evaluate based on execution time and result content
            if execution.execution_time < 0.1:
                efficiency_score = 1.0
            elif execution.execution_time < 0.2:
                efficiency_score = 0.8
            else:
                efficiency_score = 0.6
            
            # Evaluate quality based on result content
            result_str = str(execution.result)
            if len(result_str) > 100 and "detailed" in result_str.lower():
                quality_score = 0.9
            elif len(result_str) > 50:
                quality_score = 0.7
            else:
                quality_score = 0.5
            
            # Determine overall performance
            overall_score = (efficiency_score + quality_score) / 2
            if overall_score >= 0.9:
                performance_level = PerformanceLevel.EXCELLENT
            elif overall_score >= 0.7:
                performance_level = PerformanceLevel.GOOD
            elif overall_score >= 0.5:
                performance_level = PerformanceLevel.AVERAGE
            else:
                performance_level = PerformanceLevel.POOR
            
            success_rate = 1.0
        
        # Identify areas for improvement
        areas_for_improvement = []
        what_went_well = []
        
        if execution.execution_time > 0.2:
            areas_for_improvement.append("execution_speed")
        else:
            what_went_well.append("efficient_execution")
        
        if "error" in str(execution.result).lower():
            areas_for_improvement.append("accuracy")
        else:
            what_went_well.append("accurate_result")
        
        if len(str(execution.result)) < 50:
            areas_for_improvement.append("result_detail")
        else:
            what_went_well.append("detailed_result")
        
        return PerformanceEvaluation(
            execution_id=execution.task_id,
            performance_level=performance_level,
            success_rate=success_rate,
            efficiency_score=efficiency_score if execution.success else 0.0,
            quality_score=quality_score,
            areas_for_improvement=areas_for_improvement,
            what_went_well=what_went_well
        )
    
    async def reflect_on_experience(self, execution: TaskExecution, evaluation: PerformanceEvaluation) -> Reflection:
        """
        Reflect deeply on what happened and why
        """
        # Analyze what happened
        if execution.success:
            what_happened = f"Successfully completed {execution.task_type} with {evaluation.performance_level.value} performance"
        else:
            what_happened = f"Failed to complete {execution.task_type} due to execution issues"
        
        # Analyze why it happened
        if "learned_strategy" in execution.approach_used:
            why_it_happened = "Used learned strategy from previous experiences"
        else:
            why_it_happened = "Used default strategy without learned optimizations"
        
        # Identify what could be better
        what_could_be_better = ""
        if "execution_speed" in evaluation.areas_for_improvement:
            what_could_be_better += "Could optimize for faster execution. "
        if "accuracy" in evaluation.areas_for_improvement:
            what_could_be_better += "Could improve accuracy and error handling. "
        if "result_detail" in evaluation.areas_for_improvement:
            what_could_be_better += "Could provide more detailed and comprehensive results. "
        
        if not what_could_be_better:
            what_could_be_better = "Performance was satisfactory with minimal room for improvement"
        
        # Identify root causes and contributing factors
        root_causes = []
        contributing_factors = []
        
        if not execution.success:
            root_causes.append("insufficient_error_handling")
            contributing_factors.append("lack_of_validation")
        
        if execution.execution_time > 0.2:
            root_causes.append("inefficient_algorithm")
            contributing_factors.append("non_optimized_approach")
        
        return Reflection(
            execution_id=execution.task_id,
            what_happened=what_happened,
            why_it_happened=why_it_happened,
            what_could_be_better=what_could_be_better,
            root_causes=root_causes,
            contributing_factors=contributing_factors
        )
    
    def extract_lessons(self, execution: TaskExecution, evaluation: PerformanceEvaluation, reflection: Reflection) -> List[Lesson]:
        """
        Extract actionable lessons from the experience
        """
        lessons = []
        lesson_counter = len(self.learned_lessons)
        
        # Learn from successful approaches
        if execution.success and evaluation.performance_level in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD]:
            if "learned_strategy" in execution.approach_used:
                # Reinforce that learned strategies work well
                lesson = Lesson(
                    lesson_id=f"lesson_{lesson_counter + len(lessons) + 1}",
                    lesson_type=LessonType.STRATEGY,
                    content=f"Learned strategies for {execution.task_type} continue to be effective",
                    confidence=0.8,
                    applicable_to=[execution.task_type],
                    learned_from=execution.task_id
                )
                lessons.append(lesson)
        
        # Learn from mistakes
        if not execution.success or evaluation.performance_level == PerformanceLevel.POOR:
            lesson = Lesson(
                lesson_id=f"lesson_{lesson_counter + len(lessons) + 1}",
                lesson_type=LessonType.MISTAKE_AVOIDANCE,
                content=f"Avoid approach '{execution.approach_used}' for {execution.task_type} as it leads to poor results",
                confidence=0.7,
                applicable_to=[execution.task_type],
                learned_from=execution.task_id
            )
            lessons.append(lesson)
        
        # Learn optimizations
        if "execution_speed" in evaluation.areas_for_improvement:
            lesson = Lesson(
                lesson_id=f"lesson_{lesson_counter + len(lessons) + 1}",
                lesson_type=LessonType.OPTIMIZATION,
                content=f"For {execution.task_type}, prioritize faster execution methods",
                confidence=0.6,
                applicable_to=[execution.task_type],
                learned_from=execution.task_id
            )
            lessons.append(lesson)
        
        # Learn general principles
        if len(self.execution_history) >= 3:
            # Look for patterns across multiple executions
            recent_executions = self.execution_history[-3:]
            if all(ex.success for ex in recent_executions):
                lesson = Lesson(
                    lesson_id=f"lesson_{lesson_counter + len(lessons) + 1}",
                    lesson_type=LessonType.GENERAL_PRINCIPLE,
                    content="Consistent success indicates current strategies are working well",
                    confidence=0.8,
                    applicable_to=["all"],
                    learned_from="pattern_analysis"
                )
                lessons.append(lesson)
        
        return lessons
    
    def update_strategies(self, task_type: str, execution: TaskExecution, evaluation: PerformanceEvaluation, new_lessons: List[Lesson]):
        """
        Update strategies based on new lessons learned
        """
        # Update strategy if we learned something important
        for lesson in new_lessons:
            if (lesson.lesson_type == LessonType.MISTAKE_AVOIDANCE and 
                evaluation.performance_level == PerformanceLevel.FAILED):
                # Change strategy to avoid repeating mistakes
                self.current_strategies[task_type] = "improved_error_handling"
            
            elif (lesson.lesson_type == LessonType.OPTIMIZATION and
                  "execution_speed" in evaluation.areas_for_improvement):
                # Update strategy to focus on speed
                self.current_strategies[task_type] = "speed_optimized"
    
    def show_learning_progress(self) -> None:
        """
        Display the agent's learning progress and insights
        """
        print("\nLEARNING PROGRESS REPORT")
        print("=" * 40)
        print(f"Total tasks completed: {len(self.execution_history)}")
        print(f"Success rate: {sum(1 for ex in self.execution_history if ex.success) / len(self.execution_history) * 100:.1f}%" if self.execution_history else "0%")
        print(f"Lessons learned: {len(self.learned_lessons)}")
        
        if self.learned_lessons:
            print("\nKEY LESSONS:")
            for lesson in self.learned_lessons[-3:]:  # Show last 3 lessons
                print(f"- {lesson.content}")
        
        print(f"\nCURRENT STRATEGIES:")
        for task_type, strategy in self.current_strategies.items():
            print(f"- {task_type}: {strategy}")

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_math_learning():
    """Demo: Agent learning to solve math problems better"""
    print("\nDEMO 1: LEARNING MATH PROBLEM SOLVING")
    print("=" * 50)
    
    agent = SelfReflectionAgent()
    
    # Give the agent several math problems to learn from
    problems = [
        "Calculate 15% of 240",
        "Calculate 20% of 150", 
        "Calculate 25% of 80"
    ]
    
    for i, problem in enumerate(problems):
        print(f"\n--- Attempt {i+1} ---")
        await agent.perform_task("math_problem", problem)
    
    agent.show_learning_progress()

async def demo_research_improvement():
    """Demo: Agent learning to research topics better"""
    print("\nDEMO 2: LEARNING RESEARCH SKILLS")
    print("=" * 50)
    
    agent = SelfReflectionAgent()
    
    # Give the agent research tasks to learn from
    topics = [
        "Artificial Intelligence applications",
        "Machine Learning algorithms",
        "Python programming best practices"
    ]
    
    for i, topic in enumerate(topics):
        print(f"\n--- Research {i+1} ---")
        await agent.perform_task("research_task", topic)
    
    agent.show_learning_progress()

async def main():
    """
    Demonstrate Self Reflection Agents learning from experience
    
    WHAT YOU'LL LEARN:
    ================
    1. How agents can evaluate their own performance
    2. How to extract lessons from successes and failures
    3. How to adapt strategies based on experience
    4. How self-reflection enables continuous improvement
    5. How agents can become more effective over time
    
    REAL WORLD APPLICATIONS:
    =======================
    - Personal AI assistants that get better at helping you
    - Customer service bots that learn from interactions
    - Educational AI that adapts to student needs
    - Recommendation systems that improve over time
    - Any AI system that needs to adapt and improve
    """
    
    print("SELF REFLECTION AGENTS DEMONSTRATION")
    print("This shows how AI can learn from its mistakes and improve!")
    
    await demo_math_learning()
    await demo_research_improvement()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Self-reflection enables continuous improvement")
    print("✓ Agents can learn from both successes and failures")
    print("✓ Performance evaluation guides learning")
    print("✓ Lessons learned improve future performance")
    print("✓ Adaptation makes agents more effective over time")
    print("\nTRY IT YOURSELF:")
    print("- Add more sophisticated performance metrics")
    print("- Implement different learning strategies")
    print("- Add memory consolidation for long-term learning")
    print("- Create specialized reflection for different domains")

if __name__ == "__main__":
    asyncio.run(main())
