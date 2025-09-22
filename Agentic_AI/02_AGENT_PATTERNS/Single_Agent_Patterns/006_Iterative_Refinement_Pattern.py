#!/usr/bin/env python3
"""
Iterative Refinement Pattern: Making Things Better Step by Step
=============================================================

WHAT IS THE PROBLEM?
==================
Most people try to create perfect solutions on the first try and fail because:

Example: Writing a Perfect Essay
- Try to write a perfect essay in one draft
- Get stuck because nothing sounds good enough
- Spend hours on the first paragraph
- Give up because it's not perfect
- End up with nothing

REAL WORLD EXAMPLE:
=================
How do professional writers actually work?

BAD APPROACH (Perfectionist):
- Write perfect first sentence
- Write perfect second sentence
- Get stuck because third sentence isn't perfect
- Delete everything and start over
- Repeat until deadline, submit nothing

GOOD APPROACH (Iterative Refinement):
Draft 1: "AI is cool. It can do many things. It helps people."
Draft 2: "Artificial Intelligence is powerful technology. It can automate tasks and solve complex problems. It helps people be more productive."
Draft 3: "Artificial Intelligence represents a transformative technology that can automate routine tasks and solve complex problems, ultimately helping people become more productive and focus on creative work."
Draft 4: Polish grammar, add examples, improve flow
Final: Professional-quality writing

THE ALGORITHM:
=============
1. CREATE: Make a rough first version (don't worry about perfection)
2. EVALUATE: Identify what's wrong and what could be better
3. REFINE: Make specific improvements to address issues
4. TEST: Check if the improvements actually work
5. REPEAT: Continue until good enough or time runs out

PSEUDO CODE:
===========
def iterative_refinement(initial_problem):
    solution = create_rough_solution(initial_problem)
    
    for iteration in range(max_iterations):
        # Evaluate current solution
        issues = identify_problems(solution)
        improvements = identify_improvements(solution)
        
        # If good enough, stop
        if quality_sufficient(solution, issues):
            break
        
        # Refine solution
        solution = apply_improvements(solution, improvements)
        solution = fix_issues(solution, issues)
        
        # Test the refined solution
        new_quality = test_solution(solution)
        
        # If not improving, try different approach
        if new_quality <= previous_quality:
            solution = try_alternative_approach(solution)
    
    return solution

WHY IS THIS POWERFUL?
===================
- Overcomes perfectionist paralysis
- Enables continuous improvement
- Works for complex problems that can't be solved perfectly first try
- Allows learning and adaptation during the process
- Produces better final results than trying to be perfect immediately
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class RefinementType(Enum):
    CONTENT_IMPROVEMENT = "content_improvement"
    STRUCTURE_OPTIMIZATION = "structure_optimization"
    CLARITY_ENHANCEMENT = "clarity_enhancement"
    ACCURACY_CORRECTION = "accuracy_correction"
    COMPLETENESS_ADDITION = "completeness_addition"
    EFFICIENCY_OPTIMIZATION = "efficiency_optimization"

class QualityMetric(Enum):
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    EFFICIENCY = "efficiency"
    USEFULNESS = "usefulness"

@dataclass
class QualityAssessment:
    """Assessment of solution quality across different metrics"""
    accuracy_score: float
    completeness_score: float
    clarity_score: float
    efficiency_score: float
    usefulness_score: float
    overall_score: float
    identified_issues: List[str]
    improvement_suggestions: List[str]

@dataclass
class RefinementIteration:
    """Record of one refinement iteration"""
    iteration_number: int
    solution_before: Any
    solution_after: Any
    refinements_applied: List[str]
    quality_before: QualityAssessment
    quality_after: QualityAssessment
    improvement_achieved: float
    time_spent: float

class IterativeRefinementAgent:
    """
    An agent that creates solutions through iterative refinement
    
    EXAMPLE USAGE:
    =============
    agent = IterativeRefinementAgent()
    
    # Agent will create initial solution, then refine it step by step
    final_solution = await agent.solve_iteratively("Write a guide for learning Python")
    
    # You can see each iteration and how the solution improved
    agent.show_refinement_process()
    """
    
    def __init__(self, max_iterations: int = 5, quality_threshold: float = 0.8):
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.refinement_history: List[RefinementIteration] = []
        self.current_solution = None
        
        # Different refinement strategies
        self.refinement_strategies = {
            RefinementType.CONTENT_IMPROVEMENT: self.improve_content,
            RefinementType.STRUCTURE_OPTIMIZATION: self.optimize_structure,
            RefinementType.CLARITY_ENHANCEMENT: self.enhance_clarity,
            RefinementType.ACCURACY_CORRECTION: self.correct_accuracy,
            RefinementType.COMPLETENESS_ADDITION: self.add_completeness,
            RefinementType.EFFICIENCY_OPTIMIZATION: self.optimize_efficiency
        }
    
    async def solve_iteratively(self, problem: str, problem_type: str = "general") -> Dict[str, Any]:
        """
        Solve a problem through iterative refinement
        
        Args:
            problem: The problem to solve
            problem_type: Type of problem (writing, coding, planning, etc.)
            
        Returns:
            Final refined solution with improvement history
        """
        print(f"PROBLEM: {problem}")
        print("=" * 60)
        
        self.refinement_history = []
        
        # Step 1: Create initial rough solution
        print("ITERATION 0: Creating initial solution...")
        initial_solution = await self.create_initial_solution(problem, problem_type)
        self.current_solution = initial_solution
        
        # Assess initial quality
        initial_quality = await self.assess_quality(initial_solution, problem, problem_type)
        print(f"Initial quality: {initial_quality.overall_score:.2f}")
        print(f"Issues found: {initial_quality.identified_issues}")
        
        # Step 2: Iteratively refine
        for iteration in range(1, self.max_iterations + 1):
            print(f"\nITERATION {iteration}: Refining solution...")
            
            # Check if quality is good enough
            if initial_quality.overall_score >= self.quality_threshold:
                print("Quality threshold reached - stopping refinement")
                break
            
            # Perform one refinement iteration
            iteration_result = await self.perform_refinement_iteration(
                iteration, problem, problem_type, initial_quality
            )
            
            self.refinement_history.append(iteration_result)
            self.current_solution = iteration_result.solution_after
            initial_quality = iteration_result.quality_after
            
            print(f"Quality improved from {iteration_result.quality_before.overall_score:.2f} to {iteration_result.quality_after.overall_score:.2f}")
            print(f"Refinements: {iteration_result.refinements_applied}")
            
            # If no improvement, try different approach or stop
            if iteration_result.improvement_achieved <= 0.01:
                print("Minimal improvement - considering alternative approaches")
                break
        
        # Final result
        final_quality = initial_quality
        print(f"\nFINAL RESULT:")
        print(f"Quality: {final_quality.overall_score:.2f}")
        print(f"Iterations: {len(self.refinement_history)}")
        
        return {
            "problem": problem,
            "final_solution": self.current_solution,
            "final_quality": final_quality.overall_score,
            "iterations_performed": len(self.refinement_history),
            "total_improvement": (final_quality.overall_score - 
                                self.refinement_history[0].quality_before.overall_score 
                                if self.refinement_history else 0),
            "refinement_history": self.refinement_history
        }
    
    async def create_initial_solution(self, problem: str, problem_type: str) -> Any:
        """
        Create a rough initial solution - don't worry about perfection
        """
        await asyncio.sleep(0.1)  # Simulate thinking time
        
        if problem_type == "writing" or "write" in problem.lower():
            # Create basic writing structure
            return {
                "title": self.extract_topic(problem),
                "sections": [
                    {"heading": "Introduction", "content": "Basic introduction to the topic."},
                    {"heading": "Main Content", "content": "Main information about the topic."},
                    {"heading": "Conclusion", "content": "Summary and conclusion."}
                ],
                "word_count": 50
            }
        
        elif problem_type == "coding" or "code" in problem.lower():
            # Create basic code structure
            return {
                "functions": ["main_function()"],
                "comments": "# Basic implementation",
                "error_handling": False,
                "documentation": "Basic function",
                "complexity": "simple"
            }
        
        elif problem_type == "planning" or "plan" in problem.lower():
            # Create basic plan structure
            return {
                "goal": problem,
                "steps": [
                    "Step 1: Basic planning",
                    "Step 2: Implementation", 
                    "Step 3: Completion"
                ],
                "timeline": "rough estimate",
                "resources": "basic resources"
            }
        
        else:
            # Generic solution
            return {
                "approach": "basic approach to solve the problem",
                "details": f"Simple solution for: {problem}",
                "completeness": "minimal"
            }
    
    async def assess_quality(self, solution: Any, problem: str, problem_type: str) -> QualityAssessment:
        """
        Assess the quality of current solution across multiple metrics
        """
        await asyncio.sleep(0.05)  # Simulate assessment time
        
        # Assess different quality metrics
        accuracy_score = self.assess_accuracy(solution, problem)
        completeness_score = self.assess_completeness(solution, problem_type)
        clarity_score = self.assess_clarity(solution)
        efficiency_score = self.assess_efficiency(solution)
        usefulness_score = self.assess_usefulness(solution, problem)
        
        # Calculate overall score
        overall_score = (accuracy_score + completeness_score + clarity_score + 
                        efficiency_score + usefulness_score) / 5
        
        # Identify issues and improvements
        issues = []
        improvements = []
        
        if accuracy_score < 0.7:
            issues.append("accuracy_low")
            improvements.append("verify_facts_and_correctness")
        
        if completeness_score < 0.7:
            issues.append("incomplete_content")
            improvements.append("add_missing_information")
        
        if clarity_score < 0.7:
            issues.append("unclear_presentation")
            improvements.append("improve_structure_and_explanation")
        
        if efficiency_score < 0.7:
            issues.append("inefficient_approach")
            improvements.append("optimize_for_efficiency")
        
        if usefulness_score < 0.7:
            issues.append("limited_usefulness")
            improvements.append("add_practical_value")
        
        return QualityAssessment(
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            clarity_score=clarity_score,
            efficiency_score=efficiency_score,
            usefulness_score=usefulness_score,
            overall_score=overall_score,
            identified_issues=issues,
            improvement_suggestions=improvements
        )
    
    def assess_accuracy(self, solution: Any, problem: str) -> float:
        """Assess how accurate the solution is"""
        if isinstance(solution, dict):
            # Check if solution addresses the problem
            solution_str = str(solution).lower()
            problem_words = problem.lower().split()
            
            relevance = sum(1 for word in problem_words if word in solution_str) / len(problem_words)
            return min(0.3 + relevance * 0.7, 1.0)  # Base score + relevance bonus
        return 0.5
    
    def assess_completeness(self, solution: Any, problem_type: str) -> float:
        """Assess how complete the solution is"""
        if isinstance(solution, dict):
            if problem_type == "writing":
                sections = solution.get("sections", [])
                return min(len(sections) / 5, 1.0)  # More sections = more complete
            elif problem_type == "coding":
                features = sum(1 for key in ["functions", "error_handling", "documentation"] 
                             if solution.get(key))
                return features / 3
            else:
                return min(len(solution) / 4, 1.0)  # More attributes = more complete
        return 0.3
    
    def assess_clarity(self, solution: Any) -> float:
        """Assess how clear and well-structured the solution is"""
        if isinstance(solution, dict):
            # Check for structure and organization
            has_structure = any(key in solution for key in ["sections", "steps", "functions"])
            has_details = any(len(str(value)) > 20 for value in solution.values())
            
            clarity_score = 0.3  # Base score
            if has_structure:
                clarity_score += 0.4
            if has_details:
                clarity_score += 0.3
            
            return clarity_score
        return 0.4
    
    def assess_efficiency(self, solution: Any) -> float:
        """Assess how efficient the solution approach is"""
        if isinstance(solution, dict):
            # Simple heuristic: more complex is less efficient initially
            complexity = len(str(solution))
            if complexity < 200:
                return 0.8  # Simple and efficient
            elif complexity < 500:
                return 0.6  # Moderate complexity
            else:
                return 0.4  # High complexity
        return 0.5
    
    def assess_usefulness(self, solution: Any, problem: str) -> float:
        """Assess how useful the solution is for solving the problem"""
        if isinstance(solution, dict):
            # Check for practical elements
            practical_elements = sum(1 for key in solution.keys() 
                                   if key in ["steps", "examples", "resources", "functions"])
            return min(practical_elements / 3, 1.0)
        return 0.4
    
    async def perform_refinement_iteration(self, iteration: int, problem: str, 
                                         problem_type: str, current_quality: QualityAssessment) -> RefinementIteration:
        """
        Perform one iteration of refinement
        """
        start_time = time.time()
        solution_before = self.current_solution.copy() if isinstance(self.current_solution, dict) else self.current_solution
        
        # Identify what refinements to apply
        refinement_types = self.choose_refinements(current_quality)
        
        # Apply refinements
        refined_solution = self.current_solution
        applied_refinements = []
        
        for refinement_type in refinement_types:
            if refinement_type in self.refinement_strategies:
                refined_solution = await self.refinement_strategies[refinement_type](
                    refined_solution, problem, current_quality
                )
                applied_refinements.append(refinement_type.value)
        
        # Assess quality after refinement
        new_quality = await self.assess_quality(refined_solution, problem, problem_type)
        
        improvement = new_quality.overall_score - current_quality.overall_score
        time_spent = time.time() - start_time
        
        return RefinementIteration(
            iteration_number=iteration,
            solution_before=solution_before,
            solution_after=refined_solution,
            refinements_applied=applied_refinements,
            quality_before=current_quality,
            quality_after=new_quality,
            improvement_achieved=improvement,
            time_spent=time_spent
        )
    
    def choose_refinements(self, quality: QualityAssessment) -> List[RefinementType]:
        """
        Choose which refinements to apply based on quality assessment
        """
        refinements = []
        
        # Prioritize the most needed improvements
        if quality.accuracy_score < 0.7:
            refinements.append(RefinementType.ACCURACY_CORRECTION)
        
        if quality.completeness_score < 0.7:
            refinements.append(RefinementType.COMPLETENESS_ADDITION)
        
        if quality.clarity_score < 0.7:
            refinements.append(RefinementType.CLARITY_ENHANCEMENT)
        
        if quality.efficiency_score < 0.7:
            refinements.append(RefinementType.EFFICIENCY_OPTIMIZATION)
        
        # Always try content improvement
        refinements.append(RefinementType.CONTENT_IMPROVEMENT)
        
        return refinements[:3]  # Limit to 3 refinements per iteration
    
    # REFINEMENT STRATEGIES
    # ====================
    
    async def improve_content(self, solution: Any, problem: str, quality: QualityAssessment) -> Any:
        """Improve the content quality of the solution"""
        await asyncio.sleep(0.05)
        
        if isinstance(solution, dict):
            if "sections" in solution:
                # Improve writing content
                for section in solution["sections"]:
                    if len(section["content"]) < 50:
                        section["content"] = f"Detailed explanation of {section['heading'].lower()}. This section provides comprehensive information, examples, and practical insights to help understand the topic better."
                solution["word_count"] = sum(len(section["content"]) for section in solution["sections"])
            
            elif "steps" in solution:
                # Improve planning content
                for i, step in enumerate(solution["steps"]):
                    if len(step) < 30:
                        solution["steps"][i] = f"Detailed {step} with specific actions, timelines, and expected outcomes"
        
        return solution
    
    async def optimize_structure(self, solution: Any, problem: str, quality: QualityAssessment) -> Any:
        """Optimize the structure and organization of the solution"""
        await asyncio.sleep(0.05)
        
        if isinstance(solution, dict):
            if "sections" in solution and len(solution["sections"]) < 4:
                # Add more structured sections
                solution["sections"].append({
                    "heading": "Examples and Applications", 
                    "content": "Practical examples and real-world applications."
                })
                solution["sections"].append({
                    "heading": "Best Practices", 
                    "content": "Recommended best practices and tips."
                })
        
        return solution
    
    async def enhance_clarity(self, solution: Any, problem: str, quality: QualityAssessment) -> Any:
        """Enhance clarity and readability of the solution"""
        await asyncio.sleep(0.05)
        
        if isinstance(solution, dict):
            # Add clear formatting and structure
            if "sections" in solution:
                for section in solution["sections"]:
                    if ":" not in section["content"]:
                        section["content"] = f"Overview: {section['content']} Key points: Important aspects to consider. Summary: Main takeaways."
        
        return solution
    
    async def correct_accuracy(self, solution: Any, problem: str, quality: QualityAssessment) -> Any:
        """Correct accuracy issues in the solution"""
        await asyncio.sleep(0.05)
        
        if isinstance(solution, dict):
            # Add verification and fact-checking
            if "accuracy_verified" not in solution:
                solution["accuracy_verified"] = True
                solution["fact_checked"] = "Information verified against reliable sources"
        
        return solution
    
    async def add_completeness(self, solution: Any, problem: str, quality: QualityAssessment) -> Any:
        """Add missing information to make solution more complete"""
        await asyncio.sleep(0.05)
        
        if isinstance(solution, dict):
            # Add missing components
            if "examples" not in solution:
                solution["examples"] = ["Example 1: Practical demonstration", "Example 2: Real-world application"]
            
            if "resources" not in solution:
                solution["resources"] = ["Additional reading materials", "Useful tools and references"]
        
        return solution
    
    async def optimize_efficiency(self, solution: Any, problem: str, quality: QualityAssessment) -> Any:
        """Optimize the solution for better efficiency"""
        await asyncio.sleep(0.05)
        
        if isinstance(solution, dict):
            # Add efficiency improvements
            if "optimizations" not in solution:
                solution["optimizations"] = "Streamlined approach for better efficiency and faster results"
        
        return solution
    
    def extract_topic(self, problem: str) -> str:
        """Extract the main topic from the problem description"""
        if "write" in problem.lower():
            # Extract what to write about
            words = problem.lower().split()
            if "about" in words:
                about_index = words.index("about")
                if about_index + 1 < len(words):
                    return " ".join(words[about_index + 1:]).title()
            elif "for" in words:
                for_index = words.index("for")
                if for_index + 1 < len(words):
                    return " ".join(words[for_index + 1:]).title()
        
        return problem.title()
    
    def show_refinement_process(self) -> None:
        """
        Display the complete refinement process
        """
        print("\nREFINEMENT PROCESS SUMMARY")
        print("=" * 40)
        
        if not self.refinement_history:
            print("No refinements performed")
            return
        
        initial_quality = self.refinement_history[0].quality_before.overall_score
        final_quality = self.refinement_history[-1].quality_after.overall_score
        
        print(f"Initial quality: {initial_quality:.2f}")
        print(f"Final quality: {final_quality:.2f}")
        print(f"Total improvement: {final_quality - initial_quality:.2f}")
        print(f"Iterations: {len(self.refinement_history)}")
        
        print("\nITERATION DETAILS:")
        for iteration in self.refinement_history:
            print(f"Iteration {iteration.iteration_number}:")
            print(f"  Quality: {iteration.quality_before.overall_score:.2f} → {iteration.quality_after.overall_score:.2f}")
            print(f"  Improvements: {iteration.refinements_applied}")
            print(f"  Time: {iteration.time_spent:.2f}s")

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_writing_refinement():
    """Demo: Refining a writing task iteratively"""
    print("\nDEMO 1: ITERATIVE WRITING REFINEMENT")
    print("=" * 50)
    
    agent = IterativeRefinementAgent(max_iterations=4)
    
    result = await agent.solve_iteratively(
        "Write a comprehensive guide for beginners learning Python programming",
        "writing"
    )
    
    print(f"\nFinal solution has {len(result['final_solution']['sections'])} sections")
    print(f"Quality improved by {result['total_improvement']:.2f} points")
    
    agent.show_refinement_process()

async def demo_planning_refinement():
    """Demo: Refining a planning task iteratively"""
    print("\nDEMO 2: ITERATIVE PLANNING REFINEMENT")
    print("=" * 50)
    
    agent = IterativeRefinementAgent(max_iterations=3)
    
    result = await agent.solve_iteratively(
        "Create a detailed plan for organizing a tech conference",
        "planning"
    )
    
    print(f"\nFinal plan has {len(result['final_solution']['steps'])} steps")
    print(f"Planning quality: {result['final_quality']:.2f}")
    
    agent.show_refinement_process()

async def main():
    """
    Demonstrate Iterative Refinement Pattern with practical examples
    
    WHAT YOU'LL LEARN:
    ================
    1. How to overcome perfectionist paralysis by starting rough
    2. How to systematically identify and fix problems
    3. How to improve solutions through multiple iterations
    4. How to balance quality improvement with time constraints
    5. How iterative approaches lead to better final results
    
    REAL WORLD APPLICATIONS:
    =======================
    - Content creation and writing improvement
    - Software development and code refinement
    - Product design and user experience optimization
    - Business strategy development and planning
    - Creative projects and artistic works
    - Academic research and paper writing
    """
    
    print("ITERATIVE REFINEMENT PATTERN DEMONSTRATION")
    print("This shows how to create better solutions through step-by-step improvement!")
    
    await demo_writing_refinement()
    await demo_planning_refinement()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Start with rough solutions, don't aim for perfection initially")
    print("✓ Systematically identify problems and areas for improvement")
    print("✓ Apply focused refinements in each iteration")
    print("✓ Track quality improvements to guide the process")
    print("✓ Iterative approaches often produce better results than one-shot attempts")
    print("\nTRY IT YOURSELF:")
    print("- Apply to your writing, coding, or planning projects")
    print("- Add more sophisticated quality metrics")
    print("- Implement domain-specific refinement strategies")
    print("- Add collaborative refinement with multiple agents")

if __name__ == "__main__":
    asyncio.run(main())
