#!/usr/bin/env python3
"""
Plan and Execute Pattern: Think First, Act Later
==============================================

WHAT IS THE PROBLEM?
==================
Imagine you need to organize a birthday party. What do most people do wrong?

❌ BAD APPROACH:
- Just start doing things randomly
- Buy decorations first, then realize you don't know how many guests
- Order cake before deciding on the theme
- Send invitations last minute
- Everything becomes chaotic and stressful

✅ GOOD APPROACH (Plan and Execute):
- PLAN FIRST: Make a complete plan with all steps
- THEN EXECUTE: Follow the plan step by step
- MONITOR: Check progress and adjust if needed

REAL WORLD EXAMPLE:
=================
Let's say you want to "Learn Python Programming":

WITHOUT PLAN AND EXECUTE:
- Start random tutorial
- Get confused about concepts
- Jump between different resources
- Give up after a week

WITH PLAN AND EXECUTE:

PLANNING PHASE:
1. Break down into phases: Basics → OOP → Projects → Advanced
2. Identify resources needed: Books, online courses, practice projects
3. Set timeline: 2 weeks per phase = 8 weeks total
4. Plan milestones: Complete one project per phase

EXECUTION PHASE:
1. Week 1-2: Learn syntax and basic concepts ✓
2. Week 3-4: Learn functions and data structures ✓  
3. Week 5-6: Learn OOP concepts ✓
4. Week 7-8: Build a complete project ✓

MONITORING:
- Check progress weekly
- Adjust timeline if needed
- Replan if stuck on something

THE ALGORITHM:
=============
PHASE 1: PLANNING
1. Analyze the goal
2. Break into smaller tasks
3. Identify dependencies (what must be done first)
4. Estimate time and resources
5. Create execution schedule

PHASE 2: EXECUTION  
1. Execute tasks in planned order
2. Monitor progress
3. Handle failures/roadblocks
4. Replan if major issues occur

PSEUDO CODE:
===========
goal = "Learn Python Programming"

# PLANNING PHASE
plan = create_detailed_plan(goal)
tasks = break_into_tasks(plan)
schedule = create_schedule(tasks)

# EXECUTION PHASE
for task in schedule:
    result = execute_task(task)
    if result.failed and task.is_critical:
        new_plan = replan(goal, failed_task)
        schedule = update_schedule(new_plan)
    monitor_progress()

return final_result

WHY IS THIS BETTER?
==================
- Avoids random, chaotic work
- Identifies problems early in planning
- Makes progress measurable
- Handles failures gracefully
- Scales to complex projects
- Reduces stress and overwhelm
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """A single task in our plan"""
    id: str
    name: str
    description: str
    priority: TaskPriority
    estimated_hours: float
    dependencies: List[str] = field(default_factory=list)  # Tasks that must complete first
    status: TaskStatus = TaskStatus.PENDING
    actual_hours: float = 0.0
    result: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class ExecutionPlan:
    """Complete plan for achieving a goal"""
    goal: str
    phases: List[str]
    tasks: List[Task]
    total_estimated_hours: float
    created_at: datetime = field(default_factory=datetime.now)

class PlanAndExecuteAgent:
    """
    An agent that plans completely before executing
    
    EXAMPLE USAGE:
    =============
    agent = PlanAndExecuteAgent()
    result = await agent.achieve_goal("Learn to cook Italian food")
    
    This will:
    1. Create a detailed plan with phases and tasks
    2. Execute each task in the right order
    3. Handle problems and adjust as needed
    4. Give you a complete learning plan
    """
    
    def __init__(self):
        self.current_plan: Optional[ExecutionPlan] = None
        self.execution_log = []
        
        # Available "tools" for different types of tasks
        self.task_executors = {
            "research": self.execute_research_task,
            "practice": self.execute_practice_task,
            "create": self.execute_creation_task,
            "learn": self.execute_learning_task,
            "setup": self.execute_setup_task
        }
    
    async def achieve_goal(self, goal: str) -> Dict[str, Any]:
        """
        Main method: Plan and execute to achieve any goal
        
        Args:
            goal: What you want to accomplish
            
        Returns:
            Complete results with planning and execution details
        """
        print(f"\n🎯 GOAL: {goal}")
        print("=" * 60)
        
        # PHASE 1: PLANNING
        print("📋 PHASE 1: CREATING DETAILED PLAN")
        print("-" * 40)
        
        self.current_plan = await self.create_plan(goal)
        self.display_plan()
        
        # PHASE 2: EXECUTION
        print("\n🚀 PHASE 2: EXECUTING PLAN")
        print("-" * 40)
        
        execution_result = await self.execute_plan()
        
        # PHASE 3: RESULTS
        print("\n📊 PHASE 3: FINAL RESULTS")
        print("-" * 40)
        
        return self.generate_final_report(execution_result)
    
    async def create_plan(self, goal: str) -> ExecutionPlan:
        """
        Create a detailed plan for achieving the goal
        
        This is the PLANNING phase - we think through everything first
        """
        print(f"🤔 Analyzing goal: {goal}")
        
        # Step 1: Identify what type of goal this is
        goal_type = self.classify_goal(goal)
        print(f"📂 Goal type: {goal_type}")
        
        # Step 2: Break into logical phases
        phases = self.identify_phases(goal, goal_type)
        print(f"📝 Identified phases: {phases}")
        
        # Step 3: Create detailed tasks for each phase
        all_tasks = []
        for phase_idx, phase in enumerate(phases):
            phase_tasks = self.create_phase_tasks(phase, phase_idx, goal_type)
            all_tasks.extend(phase_tasks)
        
        # Step 4: Set up dependencies between tasks
        self.setup_task_dependencies(all_tasks)
        
        # Step 5: Calculate total time estimate
        total_hours = sum(task.estimated_hours for task in all_tasks)
        
        plan = ExecutionPlan(
            goal=goal,
            phases=phases,
            tasks=all_tasks,
            total_estimated_hours=total_hours
        )
        
        print(f"⏱️  Total estimated time: {total_hours} hours")
        print(f"📋 Created {len(all_tasks)} tasks across {len(phases)} phases")
        
        return plan
    
    def classify_goal(self, goal: str) -> str:
        """Figure out what type of goal this is"""
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ["learn", "study", "understand", "master"]):
            return "learning"
        elif any(word in goal_lower for word in ["build", "create", "make", "develop"]):
            return "creation"
        elif any(word in goal_lower for word in ["organize", "plan", "arrange", "setup"]):
            return "organization"
        elif any(word in goal_lower for word in ["research", "investigate", "analyze", "find"]):
            return "research"
        else:
            return "general"
    
    def identify_phases(self, goal: str, goal_type: str) -> List[str]:
        """Break the goal into logical phases"""
        
        if goal_type == "learning":
            if "programming" in goal.lower() or "coding" in goal.lower():
                return [
                    "Foundation Setup",
                    "Basic Concepts",
                    "Practical Application", 
                    "Advanced Topics",
                    "Real Projects"
                ]
            elif "language" in goal.lower():
                return [
                    "Basic Vocabulary",
                    "Grammar Rules",
                    "Conversation Practice",
                    "Advanced Fluency"
                ]
            elif "cooking" in goal.lower():
                return [
                    "Kitchen Setup",
                    "Basic Techniques",
                    "Simple Recipes",
                    "Complex Dishes",
                    "Menu Planning"
                ]
            else:
                return [
                    "Research and Planning",
                    "Foundation Learning",
                    "Practice and Application",
                    "Mastery and Refinement"
                ]
        
        elif goal_type == "creation":
            return [
                "Planning and Design",
                "Resource Gathering",
                "Core Development",
                "Testing and Refinement",
                "Finalization"
            ]
        
        elif goal_type == "organization":
            return [
                "Assessment and Planning",
                "Preparation",
                "Implementation",
                "Monitoring and Adjustment"
            ]
        
        else:
            return [
                "Analysis and Planning",
                "Information Gathering",
                "Implementation",
                "Review and Optimization"
            ]
    
    def create_phase_tasks(self, phase: str, phase_idx: int, goal_type: str) -> List[Task]:
        """Create specific tasks for each phase"""
        tasks = []
        base_id = f"phase_{phase_idx}_task"
        
        if phase == "Foundation Setup" or phase == "Kitchen Setup":
            tasks = [
                Task(f"{base_id}_1", "Environment Setup", 
                     f"Set up necessary tools and environment for {phase}", 
                     TaskPriority.HIGH, 2.0),
                Task(f"{base_id}_2", "Resource Collection", 
                     "Gather all needed materials and resources", 
                     TaskPriority.MEDIUM, 1.5),
                Task(f"{base_id}_3", "Initial Configuration", 
                     "Configure tools and workspace for optimal learning", 
                     TaskPriority.MEDIUM, 1.0)
            ]
        
        elif "Basic" in phase:
            tasks = [
                Task(f"{base_id}_1", "Concept Introduction", 
                     f"Learn fundamental concepts of {phase}", 
                     TaskPriority.HIGH, 3.0),
                Task(f"{base_id}_2", "Guided Practice", 
                     "Practice with examples and tutorials", 
                     TaskPriority.HIGH, 4.0),
                Task(f"{base_id}_3", "Knowledge Verification", 
                     "Test understanding with exercises", 
                     TaskPriority.MEDIUM, 2.0)
            ]
        
        elif "Practice" in phase or "Application" in phase:
            tasks = [
                Task(f"{base_id}_1", "Hands-on Exercises", 
                     f"Complete practical exercises for {phase}", 
                     TaskPriority.HIGH, 5.0),
                Task(f"{base_id}_2", "Mini Projects", 
                     "Work on small projects to apply knowledge", 
                     TaskPriority.HIGH, 6.0),
                Task(f"{base_id}_3", "Skill Assessment", 
                     "Evaluate progress and identify gaps", 
                     TaskPriority.MEDIUM, 1.5)
            ]
        
        elif "Advanced" in phase:
            tasks = [
                Task(f"{base_id}_1", "Advanced Study", 
                     f"Deep dive into advanced {phase} topics", 
                     TaskPriority.MEDIUM, 4.0),
                Task(f"{base_id}_2", "Complex Projects", 
                     "Work on challenging real-world projects", 
                     TaskPriority.HIGH, 8.0),
                Task(f"{base_id}_3", "Mastery Validation", 
                     "Demonstrate mastery through comprehensive project", 
                     TaskPriority.HIGH, 6.0)
            ]
        
        else:  # Generic phase
            tasks = [
                Task(f"{base_id}_1", f"{phase} - Research", 
                     f"Research and plan for {phase}", 
                     TaskPriority.MEDIUM, 2.0),
                Task(f"{base_id}_2", f"{phase} - Implementation", 
                     f"Execute main activities of {phase}", 
                     TaskPriority.HIGH, 4.0),
                Task(f"{base_id}_3", f"{phase} - Review", 
                     f"Review and consolidate {phase} outcomes", 
                     TaskPriority.MEDIUM, 1.0)
            ]
        
        return tasks
    
    def setup_task_dependencies(self, tasks: List[Task]) -> None:
        """Set up which tasks must be completed before others"""
        # Simple rule: tasks in later phases depend on previous phase completion
        # and within each phase, tasks are sequential
        
        for i, task in enumerate(tasks):
            if i > 0:
                # Each task depends on the previous one (simplified)
                prev_task = tasks[i-1]
                task.dependencies.append(prev_task.id)
    
    def display_plan(self) -> None:
        """Show the complete plan in a readable format"""
        if not self.current_plan:
            return
        
        print(f"\n📋 COMPLETE EXECUTION PLAN")
        print(f"Goal: {self.current_plan.goal}")
        print(f"Total Tasks: {len(self.current_plan.tasks)}")
        print(f"Estimated Time: {self.current_plan.total_estimated_hours} hours")
        print()
        
        current_phase = ""
        for task in self.current_plan.tasks:
            # Group by phase (simplified by looking at task ID)
            phase_num = task.id.split('_')[1]
            if phase_num != current_phase:
                current_phase = phase_num
                phase_idx = int(phase_num)
                if phase_idx < len(self.current_plan.phases):
                    print(f"\n🔸 PHASE {phase_idx + 1}: {self.current_plan.phases[phase_idx]}")
                    print("-" * 30)
            
            priority_emoji = "🔴" if task.priority == TaskPriority.CRITICAL else "🟡" if task.priority == TaskPriority.HIGH else "🟢"
            print(f"  {priority_emoji} {task.name} ({task.estimated_hours}h)")
            print(f"     {task.description}")
            if task.dependencies:
                print(f"     Depends on: {', '.join(task.dependencies)}")
            print()
    
    async def execute_plan(self) -> Dict[str, Any]:
        """
        Execute the plan step by step
        
        This is the EXECUTION phase - we follow our plan
        """
        if not self.current_plan:
            return {"error": "No plan to execute"}
        
        start_time = time.time()
        completed_tasks = 0
        failed_tasks = 0
        
        print("🚀 Starting plan execution...")
        
        # Execute tasks in dependency order
        for task in self.current_plan.tasks:
            # Check if dependencies are satisfied
            if not self.are_dependencies_satisfied(task):
                print(f"⏸️  Skipping {task.name} - dependencies not met")
                task.status = TaskStatus.BLOCKED
                continue
            
            # Execute the task
            print(f"\n🔧 Executing: {task.name}")
            task.status = TaskStatus.IN_PROGRESS
            
            execution_start = time.time()
            
            try:
                # Determine task type and execute
                task_type = self.determine_task_type(task)
                executor = self.task_executors.get(task_type, self.execute_generic_task)
                
                result = await executor(task)
                
                execution_time = time.time() - execution_start
                task.actual_hours = execution_time / 3600  # Convert to hours
                task.status = TaskStatus.COMPLETED
                task.result = result
                completed_tasks += 1
                
                print(f"✅ Completed: {task.name} ({execution_time:.1f}s)")
                print(f"   Result: {result}")
                
            except Exception as e:
                execution_time = time.time() - execution_start
                task.actual_hours = execution_time / 3600
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                failed_tasks += 1
                
                print(f"❌ Failed: {task.name} - {str(e)}")
                
                # Decide whether to continue or replan
                if task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
                    print("🔄 Critical task failed - considering replanning...")
                    # In a real system, you might replan here
        
        total_time = time.time() - start_time
        
        return {
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "total_tasks": len(self.current_plan.tasks),
            "execution_time_seconds": total_time,
            "success_rate": completed_tasks / len(self.current_plan.tasks) if self.current_plan.tasks else 0
        }
    
    def are_dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies are completed"""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            dep_task = next((t for t in self.current_plan.tasks if t.id == dep_id), None)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def determine_task_type(self, task: Task) -> str:
        """Figure out what type of task this is"""
        name_lower = task.name.lower()
        desc_lower = task.description.lower()
        
        if any(word in name_lower + desc_lower for word in ["research", "study", "learn", "understand"]):
            return "research"
        elif any(word in name_lower + desc_lower for word in ["practice", "exercise", "apply"]):
            return "practice"
        elif any(word in name_lower + desc_lower for word in ["create", "build", "develop", "make"]):
            return "create"
        elif any(word in name_lower + desc_lower for word in ["setup", "install", "configure"]):
            return "setup"
        else:
            return "learn"
    
    # TASK EXECUTORS - These simulate actually doing the work
    # =====================================================
    
    async def execute_research_task(self, task: Task) -> str:
        """Execute research/learning tasks"""
        await asyncio.sleep(0.2)  # Simulate work
        return f"Research completed for {task.name}: Found comprehensive information and resources"
    
    async def execute_practice_task(self, task: Task) -> str:
        """Execute practice/exercise tasks"""
        await asyncio.sleep(0.3)  # Simulate work
        return f"Practice session completed for {task.name}: Skills improved through hands-on exercises"
    
    async def execute_creation_task(self, task: Task) -> str:
        """Execute creation/building tasks"""
        await asyncio.sleep(0.4)  # Simulate work
        return f"Creation task completed for {task.name}: Successfully built/created the required deliverable"
    
    async def execute_learning_task(self, task: Task) -> str:
        """Execute general learning tasks"""
        await asyncio.sleep(0.2)  # Simulate work
        return f"Learning completed for {task.name}: New concepts understood and knowledge integrated"
    
    async def execute_setup_task(self, task: Task) -> str:
        """Execute setup/configuration tasks"""
        await asyncio.sleep(0.1)  # Simulate work
        return f"Setup completed for {task.name}: Environment configured and ready for use"
    
    async def execute_generic_task(self, task: Task) -> str:
        """Execute any other type of task"""
        await asyncio.sleep(0.2)  # Simulate work
        return f"Task completed: {task.name}"
    
    def generate_final_report(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        if not self.current_plan:
            return {"error": "No plan was executed"}
        
        # Calculate actual vs estimated time
        total_actual_hours = sum(task.actual_hours for task in self.current_plan.tasks)
        estimated_hours = self.current_plan.total_estimated_hours
        time_variance = ((total_actual_hours - estimated_hours) / estimated_hours * 100) if estimated_hours > 0 else 0
        
        # Identify completed phases
        completed_phases = []
        for i, phase in enumerate(self.current_plan.phases):
            phase_tasks = [t for t in self.current_plan.tasks if f"phase_{i}_" in t.id]
            if all(t.status == TaskStatus.COMPLETED for t in phase_tasks):
                completed_phases.append(phase)
        
        report = {
            "goal": self.current_plan.goal,
            "overall_success": execution_result["success_rate"] >= 0.8,
            "completion_rate": f"{execution_result['success_rate']:.1%}",
            "tasks_completed": execution_result["completed_tasks"],
            "tasks_failed": execution_result["failed_tasks"],
            "total_tasks": execution_result["total_tasks"],
            "time_estimates": {
                "estimated_hours": estimated_hours,
                "actual_hours": total_actual_hours,
                "variance_percent": f"{time_variance:+.1f}%"
            },
            "phases_completed": completed_phases,
            "execution_time_minutes": execution_result["execution_time_seconds"] / 60
        }
        
        # Display the report
        print(f"🎯 GOAL: {report['goal']}")
        print(f"📊 SUCCESS RATE: {report['completion_rate']}")
        print(f"✅ TASKS COMPLETED: {report['tasks_completed']}/{report['total_tasks']}")
        print(f"⏱️  TIME: {report['time_estimates']['actual_hours']:.1f}h (estimated: {report['time_estimates']['estimated_hours']:.1f}h)")
        print(f"📈 TIME VARIANCE: {report['time_estimates']['variance_percent']}")
        print(f"🏁 PHASES COMPLETED: {len(report['phases_completed'])}/{len(self.current_plan.phases)}")
        
        if report["phases_completed"]:
            print("   Completed phases:")
            for phase in report["phases_completed"]:
                print(f"   ✅ {phase}")
        
        return report

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_learning_goal():
    """Demo: Learning a new skill"""
    print("\n" + "="*70)
    print("DEMO 1: LEARNING GOAL - Master Python Programming")
    print("="*70)
    
    agent = PlanAndExecuteAgent()
    await agent.achieve_goal("Learn Python programming from beginner to intermediate")

async def demo_cooking_goal():
    """Demo: Learning to cook"""
    print("\n" + "="*70)
    print("DEMO 2: COOKING GOAL - Learn Italian Cuisine")  
    print("="*70)
    
    agent = PlanAndExecuteAgent()
    await agent.achieve_goal("Learn to cook authentic Italian food")

async def demo_creation_goal():
    """Demo: Building something"""
    print("\n" + "="*70)
    print("DEMO 3: CREATION GOAL - Build a Personal Website")
    print("="*70)
    
    agent = PlanAndExecuteAgent()
    await agent.achieve_goal("Build a professional personal website with portfolio")

async def main():
    """
    Demonstrate Plan and Execute pattern with real examples
    
    WHAT YOU'LL LEARN:
    ================
    1. Why planning before acting is crucial
    2. How to break complex goals into manageable tasks
    3. How to handle dependencies between tasks
    4. How to monitor progress and adapt plans
    5. Why this pattern prevents chaos and overwhelm
    
    REAL WORLD APPLICATIONS:
    =======================
    - Personal learning and skill development
    - Project management in any field
    - Event planning and organization
    - Software development projects
    - Career planning and goal achievement
    - Habit formation and lifestyle changes
    """
    
    print("🚀 Plan and Execute Pattern Demonstration")
    print("This shows how to achieve complex goals systematically!")
    
    await demo_learning_goal()
    await demo_cooking_goal()
    await demo_creation_goal()
    
    print("\n" + "="*70)
    print("🎓 WHAT WE LEARNED:")
    print("="*70)
    print("✅ Planning prevents chaos and random effort")
    print("✅ Breaking goals into phases makes them manageable")
    print("✅ Dependencies ensure logical task ordering")
    print("✅ Monitoring helps identify problems early")
    print("✅ Structured approach increases success rate")
    print("\n🔧 TRY IT YOURSELF:")
    print("- Plan your next learning goal using this pattern")
    print("- Add replanning logic for when critical tasks fail")
    print("- Create visual progress tracking")
    print("- Add resource estimation and management")
    print("- Connect to real project management tools")

if __name__ == "__main__":
    asyncio.run(main())
