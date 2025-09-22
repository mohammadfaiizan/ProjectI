#!/usr/bin/env python3
"""
Hierarchical Agent Teams: Structured Multi-Level Organization
============================================================

WHAT IS THE PROBLEM?
==================
Large organizations with flat structures become chaotic and inefficient. Everyone tries to communicate with everyone else, decisions take forever, and no one knows who's in charge of what.

REAL WORLD EXAMPLE: How does a tech company organize?
CEO (Strategic) -> CTO (Technology) -> Engineering Manager -> Senior Engineer -> Junior Engineer

THE ALGORITHM:
1. STRUCTURE: Organize agents into clear hierarchical levels
2. DELEGATE: Higher levels assign tasks to lower levels  
3. ESCALATE: Lower levels report problems upward
4. COORDINATE: Same-level agents collaborate directly
5. DECIDE: Each level has clear decision-making authority
6. MONITOR: Higher levels track progress of subordinates

WHY IS THIS POWERFUL?
- Clear authority and accountability at each level
- Efficient delegation and decision-making
- Scalable to very large organizations
- Reduces communication overhead
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class AgentLevel(Enum):
    EXECUTIVE = "executive"
    MANAGER = "manager" 
    TEAM_LEAD = "team_lead"
    WORKER = "worker"

class HierarchicalAgent:
    def __init__(self, agent_id: str, level: AgentLevel, capabilities: List[str]):
        self.agent_id = agent_id
        self.level = level
        self.capabilities = capabilities
        self.supervisor: Optional[str] = None
        self.subordinates: List[str] = []
        self.active_tasks: Dict[str, Any] = {}

    async def execute_task(self, task: str) -> Dict[str, Any]:
        """Execute task at appropriate level"""
        print(f"{self.level.value} {self.agent_id} executing: {task}")
        await asyncio.sleep(0.5)
        
        return {
            "task": task,
            "executor": self.agent_id,
            "level": self.level.value,
            "result": f"Completed {task} at {self.level.value} level"
        }

    async def delegate_task(self, task: str, subordinate_id: str, team: 'HierarchicalTeam') -> Dict[str, Any]:
        """Delegate task to subordinate"""
        subordinate = team.get_agent(subordinate_id)
        if subordinate:
            print(f"{self.agent_id} delegating '{task}' to {subordinate_id}")
            return await subordinate.execute_task(task)
        return {"error": "Subordinate not found"}

class HierarchicalTeam:
    def __init__(self, team_id: str):
        self.team_id = team_id
        self.agents: Dict[str, HierarchicalAgent] = {}
        
    def add_agent(self, agent: HierarchicalAgent) -> None:
        self.agents[agent.agent_id] = agent
        
    def establish_hierarchy(self, supervisor: HierarchicalAgent, subordinate: HierarchicalAgent) -> None:
        subordinate.supervisor = supervisor.agent_id
        supervisor.subordinates.append(subordinate.agent_id)
        
    def get_agent(self, agent_id: str) -> Optional[HierarchicalAgent]:
        return self.agents.get(agent_id)
        
    async def execute_project(self, project: str) -> Dict[str, Any]:
        """Execute project through hierarchy"""
        # Find top-level executive
        executive = next((a for a in self.agents.values() if a.level == AgentLevel.EXECUTIVE), None)
        if executive:
            return await executive.execute_task(project)
        return {"error": "No executive found"}

async def demo_corporate_hierarchy():
    """Demo: Corporate hierarchy execution"""
    print("\nDEMO: CORPORATE HIERARCHICAL ORGANIZATION")
    print("=" * 50)
    
    team = HierarchicalTeam("corp_team")
    
    # Create hierarchy
    ceo = HierarchicalAgent("ceo", AgentLevel.EXECUTIVE, ["strategy"])
    cto = HierarchicalAgent("cto", AgentLevel.MANAGER, ["technology"])
    eng_lead = HierarchicalAgent("eng_lead", AgentLevel.TEAM_LEAD, ["engineering"])
    developer = HierarchicalAgent("developer", AgentLevel.WORKER, ["coding"])
    
    for agent in [ceo, cto, eng_lead, developer]:
        team.add_agent(agent)
    
    # Establish hierarchy
    team.establish_hierarchy(ceo, cto)
    team.establish_hierarchy(cto, eng_lead)
    team.establish_hierarchy(eng_lead, developer)
    
    # Execute project
    result = await team.execute_project("Launch new product")
    print(f"Result: {result}")

async def main():
    print("HIERARCHICAL AGENT TEAMS DEMONSTRATION")
    await demo_corporate_hierarchy()
    
    print("\nKEY BENEFITS:")
    print("✓ Clear authority and accountability")
    print("✓ Efficient delegation mechanisms")  
    print("✓ Scalable organizational structure")
    print("✓ Reduced communication overhead")

if __name__ == "__main__":
    asyncio.run(main())