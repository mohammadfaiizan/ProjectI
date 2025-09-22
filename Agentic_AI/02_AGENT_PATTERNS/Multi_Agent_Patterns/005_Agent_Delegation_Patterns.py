#!/usr/bin/env python3
"""
Agent Delegation Patterns: Effective Task Distribution and Authority Transfer
=============================================================================

WHAT IS THE PROBLEM?
==================
When someone tries to do everything themselves, they become overwhelmed and nothing gets done well. Good leaders know how to delegate effectively.

Example: CEO Doing Everything
BAD APPROACH:
- CEO tries to approve every email, every purchase, every decision
- Employees wait for CEO approval for simple tasks
- CEO works 20 hours/day, still can't keep up
- Important strategic decisions get delayed
- Company moves slowly, employees feel micromanaged
- CEO burns out, company fails to scale

REAL WORLD EXAMPLE:
=================
How does Amazon's Jeff Bezos actually delegate?

EFFECTIVE DELEGATION STRUCTURE:

STRATEGIC DECISIONS (CEO Only):
- Company vision and mission
- Major acquisitions
- Leadership hiring for VPs
- Long-term strategy changes

OPERATIONAL DECISIONS (Delegated to VPs):
- Department budgets under $50M
- Hiring for their department
- Product roadmap execution
- Process improvements

TACTICAL DECISIONS (Delegated to Directors):
- Team budgets under $5M
- Project priorities
- Resource allocation
- Performance management

EXECUTION DECISIONS (Delegated to Managers):
- Daily operations
- Individual task assignments
- Quality control
- Customer issue resolution

DELEGATION PRINCIPLES:
1. Clear Authority: "You can spend up to $10K without approval"
2. Clear Accountability: "You're responsible for team performance"
3. Clear Communication: "Report weekly on progress"
4. Clear Escalation: "Come to me if budget exceeds $50K"

THE ALGORITHM:
=============
1. ANALYZE: Determine what can and should be delegated
2. SELECT: Choose the right agent based on capabilities and capacity
3. DEFINE: Set clear authority boundaries and expectations
4. TRANSFER: Provide necessary context and resources
5. MONITOR: Track progress without micromanaging
6. SUPPORT: Provide help when requested or needed
7. EVALUATE: Assess results and refine delegation approach

PSEUDO CODE:
===========
class DelegationSystem:
    def __init__(self):
        self.delegation_rules = {}  # task_type -> delegation_criteria
        self.authority_matrix = {}  # agent -> authority_levels
        self.monitoring_schedule = {}  # delegated_task -> check_frequency
    
    def delegate_task(self, task, delegating_agent):
        # Determine if task can be delegated
        if not self.can_delegate(task, delegating_agent):
            return delegating_agent.execute_personally(task)
        
        # Find suitable delegate
        suitable_agents = self.find_suitable_delegates(task)
        best_delegate = self.select_best_delegate(suitable_agents, task)
        
        # Transfer authority and context
        delegation = self.create_delegation(task, delegating_agent, best_delegate)
        self.transfer_authority(delegation)
        self.provide_context(delegation)
        
        # Monitor execution
        self.setup_monitoring(delegation)
        
        # Execute with delegation
        result = best_delegate.execute_with_authority(delegation)
        
        # Evaluate and learn
        self.evaluate_delegation_success(delegation, result)
        
        return result

WHY IS THIS CRUCIAL?
===================
- Enables scaling beyond individual capacity
- Develops capabilities in other agents
- Frees high-level agents for strategic work
- Improves overall system efficiency
- Creates redundancy and resilience
- Builds trust and empowerment throughout the organization
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

class AuthorityLevel(Enum):
    EXECUTIVE = "executive"      # Highest level decisions
    MANAGERIAL = "managerial"    # Department/team level
    SUPERVISORY = "supervisory"  # Project/task level
    OPERATIONAL = "operational"  # Day-to-day execution
    LIMITED = "limited"          # Specific constrained tasks

class DelegationType(Enum):
    FULL_AUTHORITY = "full_authority"        # Complete decision-making power
    BOUNDED_AUTHORITY = "bounded_authority"  # Authority within specific limits
    CONSULTATIVE = "consultative"            # Must consult before major decisions
    APPROVAL_REQUIRED = "approval_required"  # Must get approval for key decisions
    EXECUTION_ONLY = "execution_only"        # Just execute, no decision authority

class TaskComplexity(Enum):
    ROUTINE = 1
    STANDARD = 2
    COMPLEX = 3
    CRITICAL = 4
    STRATEGIC = 5

@dataclass
class DelegationAuthority:
    """Defines the authority granted for a delegation"""
    delegation_type: DelegationType
    decision_limits: Dict[str, Any]  # e.g., {"budget": 10000, "timeline": 30}
    approval_requirements: List[str]  # What requires approval
    escalation_triggers: List[str]    # When to escalate back
    reporting_frequency: str          # How often to report progress

@dataclass
class DelegatedTask:
    """Task that has been delegated to another agent"""
    id: str
    original_task: str
    delegated_by: str
    delegated_to: str
    authority_granted: DelegationAuthority
    context_provided: Dict[str, Any]
    deadline: Optional[float] = None
    priority: int = 3
    status: str = "delegated"
    progress_reports: List[Dict[str, Any]] = field(default_factory=list)
    escalations: List[str] = field(default_factory=list)

@dataclass
class AgentCapability:
    """Capability assessment for delegation decisions"""
    skill_level: float  # 0.0 to 1.0
    experience_count: int
    success_rate: float
    current_workload: float
    authority_level: AuthorityLevel
    trustworthiness: float  # 0.0 to 1.0

class DelegatingAgent(ABC):
    """
    Base class for agents that can delegate tasks to others
    """
    
    def __init__(self, agent_id: str, authority_level: AuthorityLevel):
        self.agent_id = agent_id
        self.authority_level = authority_level
        
        # Delegation management
        self.delegation_rules: Dict[str, Dict[str, Any]] = {}
        self.active_delegations: Dict[str, DelegatedTask] = {}
        self.delegation_history: List[DelegatedTask] = []
        
        # Agent assessment
        self.known_agents: Dict[str, AgentCapability] = {}
        
        # Delegation preferences and policies
        self.delegation_preferences = {
            "max_concurrent_delegations": 10,
            "preferred_delegation_types": [DelegationType.BOUNDED_AUTHORITY, DelegationType.CONSULTATIVE],
            "trust_threshold": 0.7,
            "workload_threshold": 0.8
        }
        
        # Performance tracking
        self.delegation_success_rate = 0.0
        self.personal_efficiency_gain = 0.0
        self.team_development_score = 0.0
    
    @abstractmethod
    async def execute_personally(self, task: str) -> Dict[str, Any]:
        """Execute task personally without delegation"""
        pass
    
    @abstractmethod
    def assess_delegation_suitability(self, task: str) -> bool:
        """Assess whether a task is suitable for delegation"""
        pass
    
    async def delegate_task(self, task: str, context: Dict[str, Any] = None, 
                          team: 'DelegationTeam' = None) -> Dict[str, Any]:
        """
        Delegate a task to the most suitable agent
        
        Args:
            task: Description of task to delegate
            context: Additional context and requirements
            team: Team containing potential delegates
            
        Returns:
            Results of the delegated task execution
        """
        
        print(f"\n{self.agent_id} CONSIDERING DELEGATION")
        print(f"Task: {task}")
        print("-" * 50)
        
        # Step 1: Assess if task should be delegated
        should_delegate = self.assess_delegation_suitability(task)
        
        if not should_delegate:
            print("Task not suitable for delegation - executing personally")
            return await self.execute_personally(task)
        
        # Step 2: Find suitable delegates
        suitable_delegates = self.find_suitable_delegates(task, team)
        
        if not suitable_delegates:
            print("No suitable delegates found - executing personally")
            return await self.execute_personally(task)
        
        # Step 3: Select best delegate
        best_delegate = self.select_best_delegate(task, suitable_delegates, team)
        
        if not best_delegate:
            print("Unable to select delegate - executing personally")
            return await self.execute_personally(task)
        
        # Step 4: Create delegation
        delegation = await self.create_delegation(task, best_delegate, context, team)
        
        # Step 5: Transfer task and authority
        result = await self.execute_delegation(delegation, team)
        
        return result
    
    def find_suitable_delegates(self, task: str, team: 'DelegationTeam') -> List[str]:
        """Find agents suitable for delegating this task"""
        
        suitable_delegates = []
        task_requirements = self.analyze_task_requirements(task)
        
        for agent_id, agent in team.agents.items():
            if agent_id == self.agent_id:
                continue  # Can't delegate to self
            
            capability = self.assess_agent_capability(agent, task_requirements, team)
            
            # Check if agent meets minimum requirements
            meets_skill_req = capability.skill_level >= task_requirements.get("min_skill", 0.6)
            meets_authority_req = capability.authority_level.value >= task_requirements.get("min_authority", 1)
            meets_trust_req = capability.trustworthiness >= self.delegation_preferences["trust_threshold"]
            meets_workload_req = capability.current_workload <= self.delegation_preferences["workload_threshold"]
            
            if meets_skill_req and meets_authority_req and meets_trust_req and meets_workload_req:
                suitable_delegates.append(agent_id)
                print(f"  Suitable delegate found: {agent_id} (skill: {capability.skill_level:.2f})")
        
        return suitable_delegates
    
    def analyze_task_requirements(self, task: str) -> Dict[str, Any]:
        """Analyze what the task requires in terms of skills and authority"""
        
        task_lower = task.lower()
        requirements = {
            "min_skill": 0.6,
            "min_authority": 1,
            "complexity": TaskComplexity.STANDARD,
            "required_capabilities": []
        }
        
        # Analyze task complexity
        if any(word in task_lower for word in ["strategic", "critical", "major"]):
            requirements["complexity"] = TaskComplexity.CRITICAL
            requirements["min_skill"] = 0.8
            requirements["min_authority"] = 3
        elif any(word in task_lower for word in ["complex", "advanced", "difficult"]):
            requirements["complexity"] = TaskComplexity.COMPLEX
            requirements["min_skill"] = 0.7
            requirements["min_authority"] = 2
        elif any(word in task_lower for word in ["simple", "routine", "basic"]):
            requirements["complexity"] = TaskComplexity.ROUTINE
            requirements["min_skill"] = 0.5
            requirements["min_authority"] = 1
        
        # Identify required capabilities
        if "development" in task_lower or "coding" in task_lower:
            requirements["required_capabilities"].append("programming")
        if "design" in task_lower:
            requirements["required_capabilities"].append("design")
        if "research" in task_lower:
            requirements["required_capabilities"].append("research")
        if "management" in task_lower:
            requirements["required_capabilities"].append("management")
        
        return requirements
    
    def assess_agent_capability(self, agent: 'DelegatingAgent', requirements: Dict[str, Any], 
                              team: 'DelegationTeam') -> AgentCapability:
        """Assess an agent's capability for a specific task"""
        
        # Get or create capability assessment
        if agent.agent_id not in self.known_agents:
            self.known_agents[agent.agent_id] = AgentCapability(
                skill_level=0.7,  # Default assessment
                experience_count=5,
                success_rate=0.8,
                current_workload=0.5,
                authority_level=agent.authority_level,
                trustworthiness=0.8
            )
        
        capability = self.known_agents[agent.agent_id]
        
        # Update current workload
        capability.current_workload = len(agent.active_delegations) / agent.delegation_preferences["max_concurrent_delegations"]
        
        # Adjust skill level based on required capabilities
        required_caps = requirements.get("required_capabilities", [])
        if required_caps:
            # Check if agent has experience with these capabilities
            has_relevant_experience = any(cap in str(agent.delegation_history) for cap in required_caps)
            if has_relevant_experience:
                capability.skill_level = min(1.0, capability.skill_level + 0.1)
            else:
                capability.skill_level = max(0.3, capability.skill_level - 0.1)
        
        return capability
    
    def select_best_delegate(self, task: str, suitable_delegates: List[str], 
                           team: 'DelegationTeam') -> Optional[str]:
        """Select the best delegate from suitable candidates"""
        
        if not suitable_delegates:
            return None
        
        # Score each candidate
        candidate_scores = {}
        
        for delegate_id in suitable_delegates:
            agent = team.get_agent(delegate_id)
            capability = self.known_agents[delegate_id]
            
            # Calculate composite score
            skill_score = capability.skill_level * 0.4
            trust_score = capability.trustworthiness * 0.3
            availability_score = (1.0 - capability.current_workload) * 0.2
            success_score = capability.success_rate * 0.1
            
            composite_score = skill_score + trust_score + availability_score + success_score
            candidate_scores[delegate_id] = composite_score
        
        # Select highest scoring candidate
        best_delegate = max(candidate_scores.items(), key=lambda x: x[1])
        
        print(f"  Selected delegate: {best_delegate[0]} (score: {best_delegate[1]:.2f})")
        
        return best_delegate[0]
    
    async def create_delegation(self, task: str, delegate_id: str, 
                              context: Dict[str, Any], team: 'DelegationTeam') -> DelegatedTask:
        """Create a formal delegation with appropriate authority"""
        
        task_requirements = self.analyze_task_requirements(task)
        delegate_capability = self.known_agents[delegate_id]
        
        # Determine appropriate delegation type and authority
        delegation_type = self.determine_delegation_type(task_requirements, delegate_capability)
        authority = self.define_delegation_authority(task_requirements, delegation_type)
        
        # Create delegated task
        delegation = DelegatedTask(
            id=f"delegation_{uuid.uuid4().hex[:8]}",
            original_task=task,
            delegated_by=self.agent_id,
            delegated_to=delegate_id,
            authority_granted=authority,
            context_provided=context or {},
            deadline=time.time() + 3600,  # 1 hour default deadline
            priority=task_requirements.get("complexity", TaskComplexity.STANDARD).value
        )
        
        # Record delegation
        self.active_delegations[delegation.id] = delegation
        
        print(f"  Created delegation: {delegation_type.value} authority")
        print(f"  Decision limits: {authority.decision_limits}")
        
        return delegation
    
    def determine_delegation_type(self, requirements: Dict[str, Any], 
                                capability: AgentCapability) -> DelegationType:
        """Determine appropriate delegation type based on task and agent"""
        
        complexity = requirements.get("complexity", TaskComplexity.STANDARD)
        
        # High trust + high skill = full authority for non-critical tasks
        if (capability.trustworthiness >= 0.9 and 
            capability.skill_level >= 0.8 and 
            complexity.value <= TaskComplexity.COMPLEX.value):
            return DelegationType.FULL_AUTHORITY
        
        # Good trust + good skill = bounded authority
        elif (capability.trustworthiness >= 0.7 and 
              capability.skill_level >= 0.7):
            return DelegationType.BOUNDED_AUTHORITY
        
        # Moderate trust/skill = consultative
        elif (capability.trustworthiness >= 0.6 and 
              capability.skill_level >= 0.6):
            return DelegationType.CONSULTATIVE
        
        # Lower trust/skill = approval required
        elif capability.skill_level >= 0.5:
            return DelegationType.APPROVAL_REQUIRED
        
        # Minimal capability = execution only
        else:
            return DelegationType.EXECUTION_ONLY
    
    def define_delegation_authority(self, requirements: Dict[str, Any], 
                                  delegation_type: DelegationType) -> DelegationAuthority:
        """Define specific authority boundaries for the delegation"""
        
        complexity = requirements.get("complexity", TaskComplexity.STANDARD)
        
        if delegation_type == DelegationType.FULL_AUTHORITY:
            return DelegationAuthority(
                delegation_type=delegation_type,
                decision_limits={"budget": 5000, "timeline_days": 14, "team_size": 3},
                approval_requirements=[],
                escalation_triggers=["budget_exceeded", "major_roadblock"],
                reporting_frequency="weekly"
            )
        
        elif delegation_type == DelegationType.BOUNDED_AUTHORITY:
            return DelegationAuthority(
                delegation_type=delegation_type,
                decision_limits={"budget": 2000, "timeline_days": 7, "team_size": 2},
                approval_requirements=["budget_changes", "scope_changes"],
                escalation_triggers=["budget_exceeded", "timeline_risk", "quality_issues"],
                reporting_frequency="bi-weekly"
            )
        
        elif delegation_type == DelegationType.CONSULTATIVE:
            return DelegationAuthority(
                delegation_type=delegation_type,
                decision_limits={"budget": 1000, "timeline_days": 5},
                approval_requirements=["all_major_decisions", "approach_changes"],
                escalation_triggers=["any_uncertainty", "resource_needs"],
                reporting_frequency="daily"
            )
        
        elif delegation_type == DelegationType.APPROVAL_REQUIRED:
            return DelegationAuthority(
                delegation_type=delegation_type,
                decision_limits={"budget": 500, "timeline_days": 3},
                approval_requirements=["all_decisions", "all_actions"],
                escalation_triggers=["immediate_consultation_needed"],
                reporting_frequency="daily"
            )
        
        else:  # EXECUTION_ONLY
            return DelegationAuthority(
                delegation_type=delegation_type,
                decision_limits={"budget": 0, "timeline_days": 1},
                approval_requirements=["everything"],
                escalation_triggers=["any_questions", "any_issues"],
                reporting_frequency="real_time"
            )
    
    async def execute_delegation(self, delegation: DelegatedTask, team: 'DelegationTeam') -> Dict[str, Any]:
        """Execute the delegation by transferring to delegate and monitoring"""
        
        delegate = team.get_agent(delegation.delegated_to)
        if not delegate:
            return {"error": f"Delegate {delegation.delegated_to} not found"}
        
        print(f"  Transferring task to {delegation.delegated_to}")
        
        # Transfer context and authority
        await self.transfer_context_and_authority(delegation, delegate)
        
        # Start monitoring
        monitoring_task = asyncio.create_task(self.monitor_delegation(delegation, team))
        
        # Execute delegated task
        try:
            if hasattr(delegate, 'execute_delegated_task'):
                result = await delegate.execute_delegated_task(delegation)
            else:
                # Fallback to personal execution
                result = await delegate.execute_personally(delegation.original_task)
            
            delegation.status = "completed"
            
            # Update delegate capability based on success
            self.update_delegate_assessment(delegation, result, True)
            
            print(f"  Delegation completed successfully")
            
            return result
            
        except Exception as e:
            delegation.status = "failed"
            
            # Update delegate capability based on failure
            self.update_delegate_assessment(delegation, {"error": str(e)}, False)
            
            print(f"  Delegation failed: {str(e)}")
            
            # Escalate back for personal handling
            return await self.execute_personally(delegation.original_task)
        
        finally:
            # Clean up monitoring
            monitoring_task.cancel()
            
            # Move from active to history
            if delegation.id in self.active_delegations:
                del self.active_delegations[delegation.id]
            self.delegation_history.append(delegation)
    
    async def transfer_context_and_authority(self, delegation: DelegatedTask, 
                                           delegate: 'DelegatingAgent') -> None:
        """Transfer necessary context and authority to delegate"""
        
        # Provide task context
        context_transfer = {
            "task_description": delegation.original_task,
            "context": delegation.context_provided,
            "authority_granted": delegation.authority_granted,
            "deadline": delegation.deadline,
            "priority": delegation.priority,
            "delegating_agent": self.agent_id,
            "success_criteria": "Complete task within authority boundaries"
        }
        
        print(f"    Context transferred: {delegation.authority_granted.delegation_type.value}")
        print(f"    Decision limits: {delegation.authority_granted.decision_limits}")
        
        # Simulate context transfer
        await asyncio.sleep(0.1)
    
    async def monitor_delegation(self, delegation: DelegatedTask, team: 'DelegationTeam') -> None:
        """Monitor delegated task progress"""
        
        reporting_frequency = delegation.authority_granted.reporting_frequency
        
        # Determine monitoring interval
        if reporting_frequency == "real_time":
            interval = 0.5
        elif reporting_frequency == "daily":
            interval = 1.0
        elif reporting_frequency == "bi-weekly":
            interval = 2.0
        else:  # weekly
            interval = 3.0
        
        try:
            while delegation.status in ["delegated", "in_progress"]:
                await asyncio.sleep(interval)
                
                # Check progress
                delegate = team.get_agent(delegation.delegated_to)
                if delegate:
                    # Simulate progress check
                    progress_report = {
                        "timestamp": time.time(),
                        "status": "progressing",
                        "completion_estimate": "on_track",
                        "issues": []
                    }
                    
                    delegation.progress_reports.append(progress_report)
                    
                    # Check for escalation triggers
                    await self.check_escalation_triggers(delegation, progress_report)
                    
        except asyncio.CancelledError:
            pass
    
    async def check_escalation_triggers(self, delegation: DelegatedTask, 
                                      progress_report: Dict[str, Any]) -> None:
        """Check if delegation should be escalated back"""
        
        triggers = delegation.authority_granted.escalation_triggers
        
        # Simulate escalation checks
        if "timeline_risk" in triggers and progress_report.get("completion_estimate") == "delayed":
            await self.escalate_delegation(delegation, "Timeline at risk")
        
        if "quality_issues" in triggers and progress_report.get("quality_concerns"):
            await self.escalate_delegation(delegation, "Quality concerns identified")
    
    async def escalate_delegation(self, delegation: DelegatedTask, reason: str) -> None:
        """Escalate delegation back to delegating agent"""
        
        delegation.escalations.append(reason)
        print(f"    Escalation: {reason}")
        
        # Could trigger intervention or re-delegation
    
    def update_delegate_assessment(self, delegation: DelegatedTask, 
                                 result: Dict[str, Any], success: bool) -> None:
        """Update assessment of delegate based on delegation outcome"""
        
        delegate_id = delegation.delegated_to
        capability = self.known_agents[delegate_id]
        
        # Update success rate
        total_delegations = capability.experience_count + 1
        new_success_rate = (capability.success_rate * capability.experience_count + (1.0 if success else 0.0)) / total_delegations
        
        capability.success_rate = new_success_rate
        capability.experience_count = total_delegations
        
        # Update trust based on outcome
        if success:
            capability.trustworthiness = min(1.0, capability.trustworthiness + 0.05)
            capability.skill_level = min(1.0, capability.skill_level + 0.02)
        else:
            capability.trustworthiness = max(0.3, capability.trustworthiness - 0.1)
            capability.skill_level = max(0.3, capability.skill_level - 0.05)
        
        print(f"    Updated {delegate_id}: trust={capability.trustworthiness:.2f}, skill={capability.skill_level:.2f}")
    
    def get_delegation_summary(self) -> Dict[str, Any]:
        """Get comprehensive delegation performance summary"""
        
        total_delegations = len(self.delegation_history)
        successful_delegations = len([d for d in self.delegation_history if d.status == "completed"])
        
        if total_delegations > 0:
            self.delegation_success_rate = successful_delegations / total_delegations
        
        return {
            "agent_id": self.agent_id,
            "authority_level": self.authority_level.value,
            "total_delegations": total_delegations,
            "active_delegations": len(self.active_delegations),
            "delegation_success_rate": self.delegation_success_rate,
            "known_delegates": len(self.known_agents),
            "delegation_preferences": self.delegation_preferences,
            "most_trusted_delegates": sorted(
                [(agent_id, cap.trustworthiness) for agent_id, cap in self.known_agents.items()],
                key=lambda x: x[1], reverse=True
            )[:3]
        }

# SPECIALIZED DELEGATING AGENTS
# ============================

class ExecutiveDelegator(DelegatingAgent):
    """Executive-level agent that delegates strategic and operational tasks"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AuthorityLevel.EXECUTIVE)
        
        # Executive-specific delegation rules
        self.delegation_rules = {
            "strategic_planning": {"delegate": False, "reason": "executive_responsibility"},
            "operational_oversight": {"delegate": True, "preferred_level": AuthorityLevel.MANAGERIAL},
            "team_management": {"delegate": True, "preferred_level": AuthorityLevel.MANAGERIAL},
            "project_execution": {"delegate": True, "preferred_level": AuthorityLevel.SUPERVISORY}
        }
    
    async def execute_personally(self, task: str) -> Dict[str, Any]:
        """Execute high-level strategic tasks personally"""
        
        print(f"Executive {self.agent_id} executing personally: {task}")
        
        # Simulate executive decision-making
        await asyncio.sleep(1.0)
        
        if "strategic" in task.lower():
            return {
                "strategic_decision": "Executive decision made",
                "impact": "organization_wide",
                "authority_exercised": "executive",
                "task": task,
                "executor": self.agent_id
            }
        else:
            return {
                "executive_action": "High-level oversight applied",
                "decision_quality": "thorough",
                "task": task,
                "executor": self.agent_id
            }
    
    def assess_delegation_suitability(self, task: str) -> bool:
        """Assess if task should be delegated from executive level"""
        
        task_lower = task.lower()
        
        # Never delegate strategic decisions
        if any(word in task_lower for word in ["strategic", "vision", "merger", "acquisition"]):
            return False
        
        # Delegate operational and management tasks
        if any(word in task_lower for word in ["manage", "coordinate", "execute", "implement"]):
            return True
        
        # Delegate routine tasks
        if any(word in task_lower for word in ["routine", "standard", "regular"]):
            return True
        
        return False

class ManagerDelegator(DelegatingAgent):
    """Manager-level agent that delegates team and project tasks"""
    
    def __init__(self, agent_id: str, department: str):
        super().__init__(agent_id, AuthorityLevel.MANAGERIAL)
        self.department = department
        
        # Manager-specific delegation preferences
        self.delegation_preferences.update({
            "max_concurrent_delegations": 8,
            "preferred_delegation_types": [DelegationType.BOUNDED_AUTHORITY, DelegationType.FULL_AUTHORITY],
            "development_focus": True  # Managers focus on developing team members
        })
    
    async def execute_personally(self, task: str) -> Dict[str, Any]:
        """Execute management-level tasks personally"""
        
        print(f"Manager {self.agent_id} ({self.department}) executing: {task}")
        
        await asyncio.sleep(0.8)
        
        return {
            "management_action": "Department-level coordination applied",
            "department": self.department,
            "approach": "collaborative_management",
            "task": task,
            "executor": self.agent_id
        }
    
    def assess_delegation_suitability(self, task: str) -> bool:
        """Assess delegation suitability for management tasks"""
        
        task_lower = task.lower()
        
        # Keep high-level management decisions
        if any(word in task_lower for word in ["budget", "hiring", "strategy", "policy"]):
            return False
        
        # Delegate technical and execution tasks
        if any(word in task_lower for word in ["implement", "develop", "create", "execute"]):
            return True
        
        # Delegate routine operations
        if any(word in task_lower for word in ["routine", "daily", "regular", "standard"]):
            return True
        
        return True  # Managers generally delegate more than executives

class SupervisorDelegator(DelegatingAgent):
    """Supervisor-level agent that delegates individual tasks"""
    
    def __init__(self, agent_id: str, team_specialty: str):
        super().__init__(agent_id, AuthorityLevel.SUPERVISORY)
        self.team_specialty = team_specialty
        
        # Supervisor-specific preferences
        self.delegation_preferences.update({
            "max_concurrent_delegations": 6,
            "preferred_delegation_types": [DelegationType.CONSULTATIVE, DelegationType.APPROVAL_REQUIRED],
            "hands_on_management": True
        })
    
    async def execute_personally(self, task: str) -> Dict[str, Any]:
        """Execute supervisory tasks personally"""
        
        print(f"Supervisor {self.agent_id} ({self.team_specialty}) executing: {task}")
        
        await asyncio.sleep(0.6)
        
        return {
            "supervisory_action": "Direct team leadership applied",
            "specialty": self.team_specialty,
            "approach": "hands_on_guidance",
            "task": task,
            "executor": self.agent_id
        }
    
    def assess_delegation_suitability(self, task: str) -> bool:
        """Assess delegation for supervisor-level tasks"""
        
        task_lower = task.lower()
        
        # Keep quality control and team coordination
        if any(word in task_lower for word in ["review", "coordinate", "quality", "supervise"]):
            return False
        
        # Delegate individual work items
        if any(word in task_lower for word in ["code", "design", "research", "document"]):
            return True
        
        return True

class DelegationTeam:
    """
    Team that supports effective delegation patterns
    
    EXAMPLE USAGE:
    =============
    # Create delegation team
    team = DelegationTeam("delegation_demo")
    
    # Add agents at different levels
    executive = ExecutiveDelegator("ceo")
    manager = ManagerDelegator("eng_manager", "engineering")
    supervisor = SupervisorDelegator("team_lead", "development")
    
    team.add_agent(executive)
    team.add_agent(manager)
    team.add_agent(supervisor)
    
    # Execute with delegation
    result = await executive.delegate_task("Implement new product feature", team=team)
    """
    
    def __init__(self, team_id: str):
        self.team_id = team_id
        self.agents: Dict[str, DelegatingAgent] = {}
        
        # Team delegation metrics
        self.delegation_metrics = {
            "total_delegations": 0,
            "successful_delegations": 0,
            "delegation_depth": 0,  # How many levels deep delegations go
            "efficiency_gain": 0.0
        }
    
    def add_agent(self, agent: DelegatingAgent) -> None:
        """Add agent to the delegation team"""
        self.agents[agent.agent_id] = agent
        print(f"Added {agent.authority_level.value} agent: {agent.agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[DelegatingAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    async def execute_complex_project(self, project_description: str) -> Dict[str, Any]:
        """Execute complex project through delegation hierarchy"""
        
        print(f"\nEXECUTING COMPLEX PROJECT: {project_description}")
        print("=" * 60)
        
        # Find highest authority agent to start delegation chain
        top_agent = max(self.agents.values(), key=lambda a: a.authority_level.value)
        
        start_time = time.time()
        
        # Execute through delegation
        result = await top_agent.delegate_task(project_description, team=self)
        
        execution_time = time.time() - start_time
        
        # Analyze delegation patterns
        delegation_analysis = self.analyze_delegation_patterns()
        
        print(f"\nProject completed in {execution_time:.2f} seconds")
        print(f"Delegation chains used: {delegation_analysis['delegation_chains']}")
        
        return {
            "project_description": project_description,
            "execution_time": execution_time,
            "primary_executor": top_agent.agent_id,
            "delegation_analysis": delegation_analysis,
            "final_result": result
        }
    
    def analyze_delegation_patterns(self) -> Dict[str, Any]:
        """Analyze how delegation worked across the team"""
        
        total_delegations = sum(len(agent.active_delegations) + len(agent.delegation_history) 
                              for agent in self.agents.values())
        
        delegation_by_level = {}
        for agent in self.agents.values():
            level = agent.authority_level.value
            delegations = len(agent.delegation_history)
            delegation_by_level[level] = delegation_by_level.get(level, 0) + delegations
        
        return {
            "total_delegations": total_delegations,
            "delegation_by_level": delegation_by_level,
            "delegation_chains": len([agent for agent in self.agents.values() 
                                    if agent.delegation_history]),
            "average_delegation_success": sum(agent.delegation_success_rate 
                                            for agent in self.agents.values()) / len(self.agents)
        }
    
    def get_team_delegation_summary(self) -> Dict[str, Any]:
        """Get comprehensive team delegation summary"""
        
        agent_summaries = {}
        for agent in self.agents.values():
            agent_summaries[agent.agent_id] = agent.get_delegation_summary()
        
        return {
            "team_id": self.team_id,
            "total_agents": len(self.agents),
            "agent_summaries": agent_summaries,
            "delegation_patterns": self.analyze_delegation_patterns()
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_corporate_delegation():
    """Demo: Corporate hierarchy with effective delegation"""
    print("\nDEMO 1: CORPORATE DELEGATION HIERARCHY")
    print("=" * 60)
    
    # Create corporate team
    team = DelegationTeam("corporate_hierarchy")
    
    # Add corporate hierarchy
    ceo = ExecutiveDelegator("ceo_smith")
    cto = ManagerDelegator("cto_johnson", "technology")
    eng_manager = ManagerDelegator("eng_manager_brown", "engineering")
    team_lead = SupervisorDelegator("team_lead_davis", "development")
    
    team.add_agent(ceo)
    team.add_agent(cto)
    team.add_agent(eng_manager)
    team.add_agent(team_lead)
    
    # Execute strategic initiative through delegation
    result = await team.execute_complex_project("Launch new AI-powered customer service platform")
    
    print(f"\nCorporate Delegation Results:")
    print(f"- Delegation chains: {result['delegation_analysis']['delegation_chains']}")
    print(f"- Total delegations: {result['delegation_analysis']['total_delegations']}")
    print(f"- Average success rate: {result['delegation_analysis']['average_delegation_success']:.2%}")

async def demo_delegation_authority_levels():
    """Demo: Different delegation authority levels and their impact"""
    print("\nDEMO 2: DELEGATION AUTHORITY LEVELS")
    print("=" * 60)
    
    team = DelegationTeam("authority_demo")
    
    # Create agents with different delegation styles
    trusting_manager = ManagerDelegator("trusting_manager", "product")
    careful_manager = ManagerDelegator("careful_manager", "finance")
    
    # Adjust delegation preferences
    trusting_manager.delegation_preferences["trust_threshold"] = 0.6  # More trusting
    careful_manager.delegation_preferences["trust_threshold"] = 0.8   # More careful
    
    team.add_agent(trusting_manager)
    team.add_agent(careful_manager)
    
    # Add some team members with different trust levels
    reliable_worker = SupervisorDelegator("reliable_worker", "development")
    new_worker = SupervisorDelegator("new_worker", "development")
    
    team.add_agent(reliable_worker)
    team.add_agent(new_worker)
    
    # Simulate different trust levels
    trusting_manager.known_agents["reliable_worker"] = AgentCapability(
        skill_level=0.9, experience_count=20, success_rate=0.95,
        current_workload=0.3, authority_level=AuthorityLevel.SUPERVISORY,
        trustworthiness=0.9
    )
    
    careful_manager.known_agents["new_worker"] = AgentCapability(
        skill_level=0.6, experience_count=2, success_rate=0.7,
        current_workload=0.2, authority_level=AuthorityLevel.SUPERVISORY,
        trustworthiness=0.6
    )
    
    # Execute similar tasks with different managers
    print("Trusting manager delegating to reliable worker:")
    result1 = await trusting_manager.delegate_task("Implement complex feature", team=team)
    
    print("\nCareful manager delegating to new worker:")
    result2 = await careful_manager.delegate_task("Implement complex feature", team=team)
    
    print(f"\nAuthority Level Comparison:")
    print(f"- Trusting manager delegation style: More autonomous")
    print(f"- Careful manager delegation style: More oversight")

async def demo_delegation_learning():
    """Demo: How delegation improves over time through learning"""
    print("\nDEMO 3: DELEGATION LEARNING AND IMPROVEMENT")
    print("=" * 60)
    
    team = DelegationTeam("learning_demo")
    
    learning_manager = ManagerDelegator("learning_manager", "operations")
    developing_worker = SupervisorDelegator("developing_worker", "operations")
    
    team.add_agent(learning_manager)
    team.add_agent(developing_worker)
    
    # Initial assessment (low trust)
    learning_manager.known_agents["developing_worker"] = AgentCapability(
        skill_level=0.5, experience_count=1, success_rate=0.6,
        current_workload=0.3, authority_level=AuthorityLevel.SUPERVISORY,
        trustworthiness=0.5
    )
    
    # Execute multiple tasks to show learning
    tasks = [
        "Complete routine data analysis",
        "Prepare monthly operations report", 
        "Coordinate team scheduling",
        "Implement process improvement",
        "Lead quarterly review meeting"
    ]
    
    print("Demonstrating delegation learning over time:")
    
    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}: {task}")
        
        # Show current trust level
        current_trust = learning_manager.known_agents["developing_worker"].trustworthiness
        print(f"Current trust level: {current_trust:.2f}")
        
        # Execute delegation
        result = await learning_manager.delegate_task(task, team=team)
        
        # Show updated trust level
        new_trust = learning_manager.known_agents["developing_worker"].trustworthiness
        print(f"Updated trust level: {new_trust:.2f}")
        
        if new_trust > current_trust:
            print("✓ Trust increased due to successful delegation")
        elif new_trust < current_trust:
            print("✗ Trust decreased due to issues")
        else:
            print("→ Trust level maintained")
    
    # Show final delegation capability
    final_summary = learning_manager.get_delegation_summary()
    print(f"\nFinal Delegation Results:")
    print(f"- Total delegations: {final_summary['total_delegations']}")
    print(f"- Success rate: {final_summary['delegation_success_rate']:.2%}")
    print(f"- Most trusted delegates: {final_summary['most_trusted_delegates']}")

async def main():
    """
    Demonstrate Agent Delegation Patterns for effective task distribution
    
    WHAT YOU'LL LEARN:
    ================
    1. How to assess delegation suitability and select appropriate delegates
    2. How to define authority boundaries and delegation types
    3. How to monitor delegated tasks without micromanaging
    4. How to build trust and capability through successful delegations
    5. How delegation patterns scale organizational effectiveness
    
    REAL WORLD APPLICATIONS:
    =======================
    - Corporate management and organizational hierarchy
    - Project management with distributed teams
    - Government administration and policy implementation
    - Military command and control structures
    - Healthcare systems with specialized roles
    - Educational institutions with administrative delegation
    """
    
    print("AGENT DELEGATION PATTERNS DEMONSTRATION")
    print("This shows how to effectively delegate tasks while maintaining authority and accountability!")
    
    await demo_corporate_delegation()
    await demo_delegation_authority_levels()
    await demo_delegation_learning()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Effective delegation requires matching tasks to agent capabilities")
    print("✓ Authority boundaries prevent delegation failures and conflicts")
    print("✓ Monitoring systems ensure accountability without micromanagement")
    print("✓ Trust builds over time through successful delegation experiences")
    print("✓ Delegation patterns enable organizational scaling and development")
    print("\nTRY IT YOURSELF:")
    print("- Implement domain-specific delegation rules and policies")
    print("- Add delegation performance analytics and optimization")
    print("- Create automated delegation routing based on agent capabilities")
    print("- Build delegation training and capability development systems")

if __name__ == "__main__":
    asyncio.run(main())
