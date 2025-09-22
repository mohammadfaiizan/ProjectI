#!/usr/bin/env python3
"""
Goal Oriented Agents: Working Backwards from Desired Outcomes
============================================================

WHAT IS THE PROBLEM?
==================
Most people work forward from where they are, not backward from where they want to be.

Example: Getting a Job
FORWARD THINKING (Often Fails):
- "I'll update my resume"
- "I'll apply to some jobs"
- "I'll see what happens"
- Result: Random applications, no clear direction, frustration

REAL WORLD EXAMPLE:
=================
How do successful people actually achieve goals?

GOAL-ORIENTED APPROACH:
Goal: "Get a software engineering job at Google within 6 months"

WORKING BACKWARDS:
Step 6: Get job offer from Google
Step 5: Pass Google technical interviews
Step 4: Get interview invitation from Google
Step 3: Have resume that stands out to Google recruiters
Step 2: Build projects that demonstrate Google-level skills
Step 1: Learn specific technologies Google uses

WORKING FORWARDS FROM STEP 1:
Week 1-4: Learn Google's tech stack (Go, Kubernetes, etc.)
Week 5-8: Build impressive projects using these technologies
Week 9-12: Optimize resume with Google keywords and format
Week 13-16: Apply strategically, network with Google employees
Week 17-20: Prepare intensively for technical interviews
Week 21-24: Interview and negotiate offer

THE ALGORITHM:
=============
1. DEFINE: Clearly specify the desired end goal
2. DECOMPOSE: Break goal into required sub-goals
3. PRIORITIZE: Order sub-goals by importance and dependencies
4. PLAN: Create action plan working backwards from goal
5. EXECUTE: Work through the plan systematically
6. MONITOR: Track progress toward goal and adjust as needed

PSEUDO CODE:
===========
def goal_oriented_agent(final_goal):
    # Define clear goal with success criteria
    goal = define_clear_goal(final_goal)
    
    # Work backwards to identify required sub-goals
    sub_goals = decompose_goal(goal)
    
    # Create execution plan
    plan = create_action_plan(sub_goals)
    
    # Execute plan while monitoring progress
    while not goal_achieved(goal):
        current_step = get_next_action(plan)
        result = execute_action(current_step)
        
        progress = assess_progress_toward_goal(goal, result)
        
        if progress.blocked:
            plan = replan_around_obstacle(plan, progress.obstacle)
        
        if progress.off_track:
            plan = adjust_plan_to_goal(plan, goal)
    
    return success_result

WHY IS THIS POWERFUL?
===================
- Ensures all effort is directed toward the actual goal
- Prevents wasted time on irrelevant activities  
- Makes complex goals achievable through systematic approach
- Enables course correction when getting off track
- Maximizes likelihood of actually achieving desired outcomes
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

class GoalStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"

class GoalPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ActionType(Enum):
    RESEARCH = "research"
    SKILL_BUILDING = "skill_building"
    CREATION = "creation"
    NETWORKING = "networking"
    APPLICATION = "application"
    OPTIMIZATION = "optimization"

@dataclass
class Goal:
    """Represents a specific goal with clear success criteria"""
    id: str
    description: str
    success_criteria: List[str]
    deadline: Optional[datetime]
    priority: GoalPriority
    status: GoalStatus = GoalStatus.NOT_STARTED
    progress_percentage: float = 0.0
    parent_goal_id: Optional[str] = None
    sub_goal_ids: List[str] = field(default_factory=list)

@dataclass
class Action:
    """Represents a specific action to take toward a goal"""
    id: str
    description: str
    action_type: ActionType
    goal_id: str
    estimated_effort: float  # hours
    dependencies: List[str] = field(default_factory=list)
    completed: bool = False
    result: Optional[str] = None

@dataclass
class ProgressAssessment:
    """Assessment of progress toward a goal"""
    goal_id: str
    current_progress: float
    on_track: bool
    obstacles: List[str]
    next_actions: List[str]
    estimated_completion: Optional[datetime]

class GoalOrientedAgent:
    """
    An agent that works systematically toward achieving specific goals
    
    EXAMPLE USAGE:
    =============
    agent = GoalOrientedAgent()
    
    # Set a clear goal
    goal_id = agent.set_goal("Learn machine learning and build ML portfolio")
    
    # Agent will work backwards to create plan
    await agent.work_toward_goal(goal_id)
    
    # Agent systematically executes plan while monitoring progress
    """
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.actions: Dict[str, Action] = {}
        self.execution_history: List[Dict] = []
        
        # Goal decomposition strategies
        self.decomposition_strategies = {
            "learning": self.decompose_learning_goal,
            "career": self.decompose_career_goal,
            "project": self.decompose_project_goal,
            "business": self.decompose_business_goal,
            "health": self.decompose_health_goal,
            "creative": self.decompose_creative_goal
        }
        
        # Action planning strategies
        self.planning_strategies = {
            "skill_building": self.plan_skill_building,
            "project_creation": self.plan_project_creation,
            "job_search": self.plan_job_search,
            "business_launch": self.plan_business_launch
        }
    
    def set_goal(self, goal_description: str, deadline: Optional[str] = None, 
                priority: GoalPriority = GoalPriority.MEDIUM) -> str:
        """
        Set a clear goal with specific success criteria
        
        Args:
            goal_description: What you want to achieve
            deadline: When you want to achieve it (optional)
            priority: How important this goal is
            
        Returns:
            Goal ID for tracking
        """
        goal_id = f"goal_{len(self.goals) + 1}"
        
        # Parse deadline if provided
        deadline_dt = None
        if deadline:
            try:
                deadline_dt = datetime.strptime(deadline, "%Y-%m-%d")
            except:
                deadline_dt = datetime.now() + timedelta(days=90)  # Default 3 months
        
        # Generate success criteria based on goal description
        success_criteria = self.generate_success_criteria(goal_description)
        
        goal = Goal(
            id=goal_id,
            description=goal_description,
            success_criteria=success_criteria,
            deadline=deadline_dt,
            priority=priority
        )
        
        self.goals[goal_id] = goal
        
        print(f"GOAL SET: {goal_description}")
        print(f"Goal ID: {goal_id}")
        print(f"Success Criteria:")
        for criterion in success_criteria:
            print(f"  - {criterion}")
        
        return goal_id
    
    async def work_toward_goal(self, goal_id: str) -> Dict[str, Any]:
        """
        Work systematically toward achieving the specified goal
        
        This is the main method that:
        1. Decomposes the goal into sub-goals
        2. Creates action plan working backwards
        3. Executes actions while monitoring progress
        """
        if goal_id not in self.goals:
            return {"error": f"Goal {goal_id} not found"}
        
        goal = self.goals[goal_id]
        print(f"\nWORKING TOWARD GOAL: {goal.description}")
        print("=" * 60)
        
        # Step 1: Decompose goal into manageable sub-goals
        sub_goals = await self.decompose_goal(goal)
        print(f"DECOMPOSED INTO {len(sub_goals)} SUB-GOALS:")
        for sub_goal in sub_goals:
            print(f"  - {sub_goal.description}")
        
        # Step 2: Create action plan working backwards from goal
        action_plan = await self.create_action_plan(goal, sub_goals)
        print(f"\nCREATED ACTION PLAN WITH {len(action_plan)} ACTIONS:")
        for action in action_plan[:3]:  # Show first 3 actions
            print(f"  - {action.description} ({action.action_type.value})")
        
        # Step 3: Execute actions systematically
        execution_results = await self.execute_action_plan(goal, action_plan)
        
        # Step 4: Assess final progress
        final_progress = await self.assess_progress(goal_id)
        
        return {
            "goal_id": goal_id,
            "goal_description": goal.description,
            "sub_goals_created": len(sub_goals),
            "actions_planned": len(action_plan),
            "actions_completed": execution_results["completed_actions"],
            "final_progress": final_progress.current_progress,
            "goal_achieved": final_progress.current_progress >= 0.8,
            "execution_summary": execution_results
        }
    
    def generate_success_criteria(self, goal_description: str) -> List[str]:
        """
        Generate specific, measurable success criteria for a goal
        """
        goal_lower = goal_description.lower()
        criteria = []
        
        if "learn" in goal_lower:
            if "programming" in goal_lower or "coding" in goal_lower:
                criteria = [
                    "Complete at least 3 substantial projects demonstrating the skills",
                    "Pass technical assessments or coding challenges",
                    "Build a portfolio showcasing learned technologies",
                    "Receive positive feedback from experienced developers"
                ]
            elif "machine learning" in goal_lower or "ml" in goal_lower:
                criteria = [
                    "Complete end-to-end ML projects with real datasets",
                    "Understand and implement key ML algorithms",
                    "Deploy ML models to production environment",
                    "Achieve specific accuracy metrics on test datasets"
                ]
            else:
                criteria = [
                    "Demonstrate practical knowledge through projects",
                    "Pass relevant assessments or certifications",
                    "Apply knowledge to real-world scenarios"
                ]
        
        elif "job" in goal_lower or "career" in goal_lower:
            criteria = [
                "Receive and accept job offer meeting salary expectations",
                "Position matches desired role and responsibilities",
                "Company culture and values align with preferences",
                "Demonstrate required skills during interview process"
            ]
        
        elif "build" in goal_lower or "create" in goal_lower:
            criteria = [
                "Complete functional version meeting requirements",
                "Achieve quality standards for intended use",
                "Get positive feedback from target users",
                "Meet performance and reliability expectations"
            ]
        
        else:
            # Generic success criteria
            criteria = [
                "Achieve clearly defined measurable outcomes",
                "Meet quality standards appropriate for the goal",
                "Complete within reasonable time constraints",
                "Demonstrate value and impact of achievement"
            ]
        
        return criteria
    
    async def decompose_goal(self, goal: Goal) -> List[Goal]:
        """
        Break down a high-level goal into manageable sub-goals
        """
        goal_type = self.classify_goal_type(goal.description)
        
        if goal_type in self.decomposition_strategies:
            return await self.decomposition_strategies[goal_type](goal)
        else:
            return await self.decompose_generic_goal(goal)
    
    def classify_goal_type(self, goal_description: str) -> str:
        """Classify what type of goal this is"""
        goal_lower = goal_description.lower()
        
        if any(word in goal_lower for word in ["learn", "study", "master", "understand"]):
            return "learning"
        elif any(word in goal_lower for word in ["job", "career", "promotion", "hire"]):
            return "career"
        elif any(word in goal_lower for word in ["build", "create", "develop", "make"]):
            return "project"
        elif any(word in goal_lower for word in ["business", "startup", "company", "launch"]):
            return "business"
        elif any(word in goal_lower for word in ["health", "fitness", "weight", "exercise"]):
            return "health"
        elif any(word in goal_lower for word in ["write", "art", "music", "creative"]):
            return "creative"
        else:
            return "generic"
    
    async def decompose_learning_goal(self, goal: Goal) -> List[Goal]:
        """Decompose learning-related goals"""
        sub_goals = []
        
        # Foundation sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_foundation",
            description=f"Build foundational knowledge for {goal.description}",
            success_criteria=["Complete basic courses or tutorials", "Understand core concepts"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        # Practice sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_practice",
            description=f"Practice skills through hands-on projects",
            success_criteria=["Complete at least 3 practice projects", "Apply learned concepts"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        # Portfolio sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_portfolio",
            description=f"Build portfolio demonstrating mastery",
            success_criteria=["Create comprehensive portfolio", "Showcase best work"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        # Validation sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_validation",
            description=f"Get external validation of skills",
            success_criteria=["Pass assessments or get feedback", "Demonstrate expertise"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        # Add sub-goals to main goal
        goal.sub_goal_ids = [sg.id for sg in sub_goals]
        
        # Store sub-goals
        for sub_goal in sub_goals:
            self.goals[sub_goal.id] = sub_goal
        
        return sub_goals
    
    async def decompose_career_goal(self, goal: Goal) -> List[Goal]:
        """Decompose career-related goals"""
        sub_goals = []
        
        # Skills sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_skills",
            description="Develop required skills and qualifications",
            success_criteria=["Master key technical skills", "Meet job requirements"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        # Network sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_network",
            description="Build professional network and connections",
            success_criteria=["Connect with industry professionals", "Build relationships"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        # Application sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_application",
            description="Apply strategically to target positions",
            success_criteria=["Submit quality applications", "Get interview invitations"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        # Interview sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_interview",
            description="Excel in interview process",
            success_criteria=["Perform well in interviews", "Receive job offers"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        goal.sub_goal_ids = [sg.id for sg in sub_goals]
        for sub_goal in sub_goals:
            self.goals[sub_goal.id] = sub_goal
        
        return sub_goals
    
    async def decompose_project_goal(self, goal: Goal) -> List[Goal]:
        """Decompose project creation goals"""
        sub_goals = []
        
        # Planning sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_planning",
            description="Plan and design the project",
            success_criteria=["Create detailed project plan", "Define requirements"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        # Development sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_development",
            description="Build the core functionality",
            success_criteria=["Implement main features", "Meet functional requirements"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        # Testing sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_testing",
            description="Test and refine the project",
            success_criteria=["Complete thorough testing", "Fix critical issues"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        # Launch sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_launch",
            description="Deploy and launch the project",
            success_criteria=["Successfully deploy", "Get user feedback"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        goal.sub_goal_ids = [sg.id for sg in sub_goals]
        for sub_goal in sub_goals:
            self.goals[sub_goal.id] = sub_goal
        
        return sub_goals
    
    async def decompose_business_goal(self, goal: Goal) -> List[Goal]:
        """Decompose business-related goals"""
        # Similar structure to other decomposition methods
        return await self.decompose_generic_goal(goal)
    
    async def decompose_health_goal(self, goal: Goal) -> List[Goal]:
        """Decompose health and fitness goals"""
        return await self.decompose_generic_goal(goal)
    
    async def decompose_creative_goal(self, goal: Goal) -> List[Goal]:
        """Decompose creative project goals"""
        return await self.decompose_generic_goal(goal)
    
    async def decompose_generic_goal(self, goal: Goal) -> List[Goal]:
        """Generic goal decomposition"""
        sub_goals = []
        
        # Research sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_research",
            description=f"Research and plan approach for {goal.description}",
            success_criteria=["Complete thorough research", "Create action plan"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        # Execution sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_execution",
            description=f"Execute main activities for {goal.description}",
            success_criteria=["Complete core activities", "Make significant progress"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        # Completion sub-goal
        sub_goals.append(Goal(
            id=f"{goal.id}_completion",
            description=f"Complete and validate {goal.description}",
            success_criteria=["Achieve final outcomes", "Meet success criteria"],
            deadline=goal.deadline,
            priority=goal.priority,
            parent_goal_id=goal.id
        ))
        
        goal.sub_goal_ids = [sg.id for sg in sub_goals]
        for sub_goal in sub_goals:
            self.goals[sub_goal.id] = sub_goal
        
        return sub_goals
    
    async def create_action_plan(self, goal: Goal, sub_goals: List[Goal]) -> List[Action]:
        """
        Create specific action plan working backwards from the goal
        """
        actions = []
        action_counter = 0
        
        # Create actions for each sub-goal
        for sub_goal in sub_goals:
            sub_goal_actions = await self.create_sub_goal_actions(sub_goal, goal)
            actions.extend(sub_goal_actions)
            action_counter += len(sub_goal_actions)
        
        # Order actions by dependencies and priority
        ordered_actions = self.order_actions_by_dependencies(actions)
        
        # Store actions
        for action in ordered_actions:
            self.actions[action.id] = action
        
        return ordered_actions
    
    async def create_sub_goal_actions(self, sub_goal: Goal, main_goal: Goal) -> List[Action]:
        """Create specific actions for a sub-goal"""
        actions = []
        base_id = f"action_{sub_goal.id}"
        
        if "foundation" in sub_goal.description.lower() or "research" in sub_goal.description.lower():
            actions = [
                Action(f"{base_id}_1", "Research best learning resources and materials", 
                      ActionType.RESEARCH, sub_goal.id, 3.0),
                Action(f"{base_id}_2", "Create structured learning plan and timeline", 
                      ActionType.CREATION, sub_goal.id, 2.0),
                Action(f"{base_id}_3", "Complete foundational courses or tutorials", 
                      ActionType.SKILL_BUILDING, sub_goal.id, 20.0)
            ]
        
        elif "practice" in sub_goal.description.lower() or "development" in sub_goal.description.lower():
            actions = [
                Action(f"{base_id}_1", "Identify practice project opportunities", 
                      ActionType.RESEARCH, sub_goal.id, 2.0),
                Action(f"{base_id}_2", "Complete first practice project", 
                      ActionType.CREATION, sub_goal.id, 15.0),
                Action(f"{base_id}_3", "Complete additional practice projects", 
                      ActionType.CREATION, sub_goal.id, 25.0)
            ]
        
        elif "portfolio" in sub_goal.description.lower():
            actions = [
                Action(f"{base_id}_1", "Design portfolio structure and presentation", 
                      ActionType.CREATION, sub_goal.id, 5.0),
                Action(f"{base_id}_2", "Document and showcase completed projects", 
                      ActionType.CREATION, sub_goal.id, 10.0),
                Action(f"{base_id}_3", "Optimize portfolio for target audience", 
                      ActionType.OPTIMIZATION, sub_goal.id, 3.0)
            ]
        
        elif "network" in sub_goal.description.lower():
            actions = [
                Action(f"{base_id}_1", "Identify key people and communities to connect with", 
                      ActionType.RESEARCH, sub_goal.id, 3.0),
                Action(f"{base_id}_2", "Actively engage in professional networking", 
                      ActionType.NETWORKING, sub_goal.id, 10.0),
                Action(f"{base_id}_3", "Build meaningful professional relationships", 
                      ActionType.NETWORKING, sub_goal.id, 15.0)
            ]
        
        elif "application" in sub_goal.description.lower():
            actions = [
                Action(f"{base_id}_1", "Research target companies and positions", 
                      ActionType.RESEARCH, sub_goal.id, 5.0),
                Action(f"{base_id}_2", "Prepare application materials (resume, cover letter)", 
                      ActionType.CREATION, sub_goal.id, 8.0),
                Action(f"{base_id}_3", "Submit strategic applications to target positions", 
                      ActionType.APPLICATION, sub_goal.id, 12.0)
            ]
        
        else:
            # Generic actions
            actions = [
                Action(f"{base_id}_1", f"Plan approach for {sub_goal.description}", 
                      ActionType.RESEARCH, sub_goal.id, 2.0),
                Action(f"{base_id}_2", f"Execute main activities for {sub_goal.description}", 
                      ActionType.CREATION, sub_goal.id, 10.0),
                Action(f"{base_id}_3", f"Complete and validate {sub_goal.description}", 
                      ActionType.OPTIMIZATION, sub_goal.id, 3.0)
            ]
        
        return actions
    
    def order_actions_by_dependencies(self, actions: List[Action]) -> List[Action]:
        """Order actions based on dependencies and logical sequence"""
        # Simple ordering: research first, then skill building, then creation, etc.
        type_order = {
            ActionType.RESEARCH: 1,
            ActionType.SKILL_BUILDING: 2,
            ActionType.CREATION: 3,
            ActionType.NETWORKING: 4,
            ActionType.APPLICATION: 5,
            ActionType.OPTIMIZATION: 6
        }
        
        return sorted(actions, key=lambda a: type_order.get(a.action_type, 99))
    
    async def execute_action_plan(self, goal: Goal, actions: List[Action]) -> Dict[str, Any]:
        """
        Execute the action plan systematically
        """
        print(f"\nEXECUTING ACTION PLAN ({len(actions)} actions)")
        print("-" * 40)
        
        completed_actions = 0
        total_effort = 0
        
        for i, action in enumerate(actions[:5]):  # Execute first 5 actions for demo
            print(f"Action {i+1}: {action.description}")
            
            # Simulate action execution
            result = await self.execute_single_action(action)
            
            if result["success"]:
                action.completed = True
                action.result = result["result"]
                completed_actions += 1
                total_effort += action.estimated_effort
                print(f"  ✓ Completed: {result['result']}")
            else:
                print(f"  ✗ Failed: {result['error']}")
            
            # Brief delay for demonstration
            await asyncio.sleep(0.1)
        
        return {
            "completed_actions": completed_actions,
            "total_actions": len(actions),
            "total_effort_hours": total_effort,
            "completion_rate": completed_actions / len(actions),
            "remaining_actions": len(actions) - completed_actions
        }
    
    async def execute_single_action(self, action: Action) -> Dict[str, Any]:
        """Execute a single action"""
        await asyncio.sleep(0.05)  # Simulate work
        
        # Simulate different success rates based on action type
        success_rates = {
            ActionType.RESEARCH: 0.9,
            ActionType.SKILL_BUILDING: 0.8,
            ActionType.CREATION: 0.7,
            ActionType.NETWORKING: 0.6,
            ActionType.APPLICATION: 0.5,
            ActionType.OPTIMIZATION: 0.8
        }
        
        success_rate = success_rates.get(action.action_type, 0.7)
        
        # Simulate execution
        import random
        if random.random() < success_rate:
            return {
                "success": True,
                "result": f"Successfully completed {action.action_type.value} activity"
            }
        else:
            return {
                "success": False,
                "error": f"Encountered obstacles in {action.action_type.value}"
            }
    
    async def assess_progress(self, goal_id: str) -> ProgressAssessment:
        """Assess current progress toward the goal"""
        goal = self.goals[goal_id]
        
        # Calculate progress based on completed sub-goals and actions
        if goal.sub_goal_ids:
            sub_goal_progress = []
            for sub_goal_id in goal.sub_goal_ids:
                sub_goal = self.goals[sub_goal_id]
                # Calculate sub-goal progress based on related actions
                related_actions = [a for a in self.actions.values() if a.goal_id == sub_goal_id]
                if related_actions:
                    completed_actions = sum(1 for a in related_actions if a.completed)
                    sub_progress = completed_actions / len(related_actions)
                    sub_goal_progress.append(sub_progress)
            
            current_progress = sum(sub_goal_progress) / len(sub_goal_progress) if sub_goal_progress else 0.0
        else:
            current_progress = 0.0
        
        # Update goal progress
        goal.progress_percentage = current_progress * 100
        
        # Determine if on track
        if goal.deadline:
            time_passed = (datetime.now() - goal.deadline + timedelta(days=90)).days / 90
            on_track = current_progress >= time_passed * 0.8
        else:
            on_track = current_progress > 0.1
        
        return ProgressAssessment(
            goal_id=goal_id,
            current_progress=current_progress,
            on_track=on_track,
            obstacles=["Time constraints", "Skill gaps"] if not on_track else [],
            next_actions=[a.description for a in self.actions.values() 
                         if a.goal_id in goal.sub_goal_ids and not a.completed][:3],
            estimated_completion=goal.deadline
        )
    
    def show_goal_progress(self, goal_id: str) -> None:
        """Display progress toward a specific goal"""
        if goal_id not in self.goals:
            print(f"Goal {goal_id} not found")
            return
        
        goal = self.goals[goal_id]
        print(f"\nGOAL PROGRESS: {goal.description}")
        print("=" * 50)
        print(f"Status: {goal.status.value}")
        print(f"Progress: {goal.progress_percentage:.1f}%")
        print(f"Priority: {goal.priority.value}")
        
        if goal.sub_goal_ids:
            print(f"\nSUB-GOALS ({len(goal.sub_goal_ids)}):")
            for sub_goal_id in goal.sub_goal_ids:
                sub_goal = self.goals[sub_goal_id]
                print(f"  - {sub_goal.description}")
        
        # Show next actions
        next_actions = [a for a in self.actions.values() 
                       if a.goal_id in goal.sub_goal_ids and not a.completed]
        if next_actions:
            print(f"\nNEXT ACTIONS:")
            for action in next_actions[:3]:
                print(f"  - {action.description}")

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_learning_goal():
    """Demo: Goal-oriented approach to learning"""
    print("\nDEMO 1: LEARNING GOAL - Master Machine Learning")
    print("=" * 50)
    
    agent = GoalOrientedAgent()
    
    goal_id = agent.set_goal(
        "Learn machine learning and build professional ML portfolio",
        deadline="2024-06-01",
        priority=GoalPriority.HIGH
    )
    
    result = await agent.work_toward_goal(goal_id)
    
    print(f"\nGoal Achievement Summary:")
    print(f"Progress: {result['final_progress']:.1%}")
    print(f"Actions Completed: {result['actions_completed']}/{result['actions_planned']}")
    
    agent.show_goal_progress(goal_id)

async def demo_career_goal():
    """Demo: Goal-oriented approach to career advancement"""
    print("\nDEMO 2: CAREER GOAL - Get Software Engineering Job")
    print("=" * 50)
    
    agent = GoalOrientedAgent()
    
    goal_id = agent.set_goal(
        "Get hired as software engineer at tech company",
        deadline="2024-04-01",
        priority=GoalPriority.CRITICAL
    )
    
    result = await agent.work_toward_goal(goal_id)
    
    print(f"\nCareer Goal Progress:")
    print(f"Sub-goals created: {result['sub_goals_created']}")
    print(f"Action plan: {result['actions_planned']} actions")

async def main():
    """
    Demonstrate Goal Oriented Agents working backward from desired outcomes
    
    WHAT YOU'LL LEARN:
    ================
    1. How to set clear, measurable goals with success criteria
    2. How to work backwards from goals to create action plans
    3. How to systematically decompose complex goals
    4. How to monitor progress and adjust course when needed
    5. How goal-oriented thinking maximizes achievement likelihood
    
    REAL WORLD APPLICATIONS:
    =======================
    - Personal career development and skill building
    - Business strategy and product development
    - Academic and research goal achievement
    - Health and fitness transformation goals
    - Creative project completion
    - Financial and investment planning
    """
    
    print("GOAL ORIENTED AGENTS DEMONSTRATION")
    print("This shows how to achieve complex goals through systematic backward planning!")
    
    await demo_learning_goal()
    await demo_career_goal()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Clear goals with success criteria enable focused effort")
    print("✓ Working backwards reveals necessary steps and dependencies")
    print("✓ Systematic decomposition makes complex goals manageable")
    print("✓ Action plans provide concrete steps toward achievement")
    print("✓ Progress monitoring enables course correction and optimization")
    print("\nTRY IT YOURSELF:")
    print("- Apply to your personal or professional goals")
    print("- Add deadline tracking and urgency management")
    print("- Implement goal prioritization and resource allocation")
    print("- Create collaborative goal achievement for teams")

if __name__ == "__main__":
    asyncio.run(main())
