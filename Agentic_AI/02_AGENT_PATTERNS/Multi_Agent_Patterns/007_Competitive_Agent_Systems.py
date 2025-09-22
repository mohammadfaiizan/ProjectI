#!/usr/bin/env python3
"""
Competitive Agent Systems: Performance Through Competition
=========================================================

WHAT IS THE PROBLEM?
==================
Without competition, systems become complacent and performance stagnates. Competition drives innovation and excellence.

Example: Monopoly vs Competition
BAD APPROACH (Monopoly):
- Single company controls market
- No pressure to improve
- High prices, poor service
- Innovation slows down
- Customers have no alternatives

REAL WORLD EXAMPLE:
=================
How does Uber's driver system work?

COMPETITIVE RIDE-SHARING:
- Multiple drivers compete for rides
- Drivers see ride requests in real-time
- Fastest response gets the ride
- Customer ratings affect future opportunities
- Higher-rated drivers get premium rides
- Competition drives better service

COMPETITION MECHANICS:
1. Performance tracking and ranking
2. Resource allocation based on performance
3. Rewards for winners, penalties for poor performance
4. Fair competition rules and monitoring
5. Continuous performance comparison

THE ALGORITHM:
=============
1. SETUP: Define competition rules and metrics
2. COMPETE: Agents compete for tasks/resources
3. EVALUATE: Measure and compare performance
4. RANK: Order agents by performance
5. REWARD: Allocate resources based on ranking
6. ADAPT: Agents improve to stay competitive

WHY IS THIS POWERFUL?
===================
- Drives continuous improvement and innovation
- Natural selection of best-performing agents
- Efficient resource allocation to top performers
- Self-organizing system optimization
- Motivates agents to exceed standards
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class CompetitionType(Enum):
    SPEED = "speed"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    INNOVATION = "innovation"

class PerformanceMetric(Enum):
    COMPLETION_TIME = "completion_time"
    ACCURACY = "accuracy"
    RESOURCE_USAGE = "resource_usage"
    CUSTOMER_SATISFACTION = "customer_satisfaction"

@dataclass
class CompetitionResult:
    """Result of a competitive task"""
    agent_id: str
    task_id: str
    completion_time: float
    quality_score: float
    resource_efficiency: float
    success: bool
    timestamp: float = field(default_factory=time.time)

@dataclass
class AgentRanking:
    """Agent's current competitive ranking"""
    agent_id: str
    overall_score: float
    rank: int
    wins: int
    competitions: int
    win_rate: float
    recent_performance: List[float]

class CompetitiveAgent:
    """Agent that participates in competitive systems"""
    
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        
        # Performance tracking
        self.competition_history: List[CompetitionResult] = []
        self.current_ranking = AgentRanking(
            agent_id=agent_id,
            overall_score=500.0,  # Starting ELO-like score
            rank=0,
            wins=0,
            competitions=0,
            win_rate=0.0,
            recent_performance=[]
        )
        
        # Competitive behavior
        self.competitiveness = random.uniform(0.6, 1.0)
        self.improvement_rate = random.uniform(0.02, 0.08)
        self.base_performance = random.uniform(0.6, 0.9)
    
    async def compete_for_task(self, task: str, competition_type: CompetitionType) -> CompetitionResult:
        """Compete for a specific task"""
        
        print(f"  {self.agent_id} competing for: {task}")
        
        # Performance varies based on competition type and agent strengths
        if competition_type == CompetitionType.SPEED:
            result = await self.compete_on_speed(task)
        elif competition_type == CompetitionType.QUALITY:
            result = await self.compete_on_quality(task)
        elif competition_type == CompetitionType.EFFICIENCY:
            result = await self.compete_on_efficiency(task)
        else:
            result = await self.compete_on_innovation(task)
        
        # Record competition
        self.competition_history.append(result)
        self.current_ranking.competitions += 1
        
        return result
    
    async def compete_on_speed(self, task: str) -> CompetitionResult:
        """Compete based on completion speed"""
        
        # Simulate variable completion time
        base_time = random.uniform(1.0, 3.0)
        performance_factor = self.base_performance + random.uniform(-0.2, 0.2)
        
        completion_time = base_time / performance_factor
        await asyncio.sleep(completion_time * 0.1)  # Scaled for demo
        
        return CompetitionResult(
            agent_id=self.agent_id,
            task_id=f"speed_task_{int(time.time())}",
            completion_time=completion_time,
            quality_score=random.uniform(0.7, 0.9),
            resource_efficiency=random.uniform(0.6, 0.8),
            success=True
        )
    
    async def compete_on_quality(self, task: str) -> CompetitionResult:
        """Compete based on output quality"""
        
        await asyncio.sleep(random.uniform(0.2, 0.4))
        
        # Quality-focused agents might take longer but produce better results
        quality_focus = random.uniform(0.1, 0.3)
        completion_time = random.uniform(1.5, 2.5) + quality_focus
        quality_score = self.base_performance + quality_focus + random.uniform(-0.1, 0.1)
        
        return CompetitionResult(
            agent_id=self.agent_id,
            task_id=f"quality_task_{int(time.time())}",
            completion_time=completion_time,
            quality_score=min(1.0, quality_score),
            resource_efficiency=random.uniform(0.5, 0.7),
            success=True
        )
    
    async def compete_on_efficiency(self, task: str) -> CompetitionResult:
        """Compete based on resource efficiency"""
        
        await asyncio.sleep(random.uniform(0.15, 0.35))
        
        efficiency = self.base_performance + random.uniform(-0.15, 0.15)
        
        return CompetitionResult(
            agent_id=self.agent_id,
            task_id=f"efficiency_task_{int(time.time())}",
            completion_time=random.uniform(1.0, 2.0),
            quality_score=random.uniform(0.7, 0.85),
            resource_efficiency=min(1.0, efficiency + 0.1),
            success=True
        )
    
    async def compete_on_innovation(self, task: str) -> CompetitionResult:
        """Compete based on innovative approaches"""
        
        await asyncio.sleep(random.uniform(0.3, 0.5))
        
        # Innovation is more variable
        innovation_score = random.uniform(0.4, 1.0)
        
        return CompetitionResult(
            agent_id=self.agent_id,
            task_id=f"innovation_task_{int(time.time())}",
            completion_time=random.uniform(1.5, 3.0),
            quality_score=innovation_score,
            resource_efficiency=random.uniform(0.5, 0.9),
            success=innovation_score > 0.6
        )
    
    def improve_performance(self, competition_result: CompetitionResult, won: bool) -> None:
        """Improve performance based on competition results"""
        
        if won:
            # Winner improves slightly
            self.base_performance = min(1.0, self.base_performance + self.improvement_rate * 0.5)
            self.current_ranking.wins += 1
        else:
            # Loser improves more (learning from failure)
            self.base_performance = min(1.0, self.base_performance + self.improvement_rate)
        
        # Update win rate
        self.current_ranking.win_rate = self.current_ranking.wins / self.current_ranking.competitions
        
        # Track recent performance
        self.current_ranking.recent_performance.append(competition_result.quality_score)
        if len(self.current_ranking.recent_performance) > 5:
            self.current_ranking.recent_performance.pop(0)

class CompetitiveSystem:
    """
    System managing competitive agent interactions
    
    EXAMPLE USAGE:
    =============
    # Create competitive system
    system = CompetitiveSystem("ride_sharing")
    
    # Add competing agents
    for i in range(5):
        agent = CompetitiveAgent(f"driver_{i}", ["driving", "navigation"])
        system.add_agent(agent)
    
    # Run competitions
    result = await system.run_competition("Transport passenger downtown", CompetitionType.SPEED)
    """
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.agents: Dict[str, CompetitiveAgent] = {}
        self.competition_history: List[Dict[str, Any]] = []
        
        # Competition settings
        self.competition_rules = {
            "max_participants": 10,
            "evaluation_criteria": [PerformanceMetric.COMPLETION_TIME, PerformanceMetric.ACCURACY],
            "reward_distribution": "winner_takes_most",  # or "proportional"
            "improvement_incentives": True
        }
    
    def add_agent(self, agent: CompetitiveAgent) -> None:
        """Add agent to competitive system"""
        self.agents[agent.agent_id] = agent
        print(f"Added competitive agent: {agent.agent_id}")
    
    async def run_competition(self, task: str, competition_type: CompetitionType, 
                            participants: List[str] = None) -> Dict[str, Any]:
        """
        Run a competition among agents for a task
        
        Args:
            task: The task to compete for
            competition_type: Type of competition
            participants: List of agent IDs (default: all agents)
            
        Returns:
            Competition results with winner and rankings
        """
        
        print(f"\nRUNNING COMPETITION: {task}")
        print(f"Competition Type: {competition_type.value}")
        print("-" * 50)
        
        # Select participants
        if participants is None:
            participants = list(self.agents.keys())
        
        participating_agents = [self.agents[agent_id] for agent_id in participants if agent_id in self.agents]
        
        if not participating_agents:
            return {"error": "No valid participants"}
        
        # Run competition
        start_time = time.time()
        competition_results = []
        
        # All agents compete simultaneously
        tasks = []
        for agent in participating_agents:
            task_coro = agent.compete_for_task(task, competition_type)
            tasks.append((agent.agent_id, task_coro))
        
        # Execute all competitions
        for agent_id, task_coro in tasks:
            try:
                result = await task_coro
                competition_results.append(result)
            except Exception as e:
                print(f"Agent {agent_id} failed: {e}")
        
        # Evaluate and rank results
        winner, rankings = self.evaluate_competition(competition_results, competition_type)
        
        # Update agent performance and rankings
        self.update_agent_rankings(competition_results, winner, rankings)
        
        # Agents improve based on results
        await self.trigger_improvement_cycle(competition_results, winner)
        
        competition_time = time.time() - start_time
        
        # Record competition
        competition_record = {
            "task": task,
            "type": competition_type.value,
            "participants": participants,
            "winner": winner.agent_id if winner else None,
            "results": competition_results,
            "rankings": rankings,
            "duration": competition_time
        }
        
        self.competition_history.append(competition_record)
        
        print(f"\nCompetition completed in {competition_time:.2f} seconds")
        if winner:
            print(f"Winner: {winner.agent_id}")
        
        return competition_record
    
    def evaluate_competition(self, results: List[CompetitionResult], 
                           competition_type: CompetitionType) -> tuple[Optional[CompetitionResult], List[CompetitionResult]]:
        """Evaluate competition results and determine winner"""
        
        if not results:
            return None, []
        
        # Score results based on competition type
        scored_results = []
        
        for result in results:
            if competition_type == CompetitionType.SPEED:
                # Lower completion time is better
                score = 10.0 / result.completion_time  # Invert time
            elif competition_type == CompetitionType.QUALITY:
                score = result.quality_score * 10.0
            elif competition_type == CompetitionType.EFFICIENCY:
                score = result.resource_efficiency * 10.0
            else:  # INNOVATION
                score = result.quality_score * 10.0  # Innovation measured by quality
            
            scored_results.append((result, score))
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Extract rankings
        rankings = [result for result, score in scored_results]
        winner = rankings[0] if rankings else None
        
        print(f"  Results:")
        for i, (result, score) in enumerate(scored_results[:3]):  # Show top 3
            print(f"    {i+1}. {result.agent_id}: {score:.2f}")
        
        return winner, rankings
    
    def update_agent_rankings(self, results: List[CompetitionResult], 
                            winner: Optional[CompetitionResult], rankings: List[CompetitionResult]) -> None:
        """Update agent rankings based on competition results"""
        
        # Update overall scores using ELO-like rating system
        for i, result in enumerate(rankings):
            agent = self.agents[result.agent_id]
            
            # ELO-like update
            expected_score = 1.0 / (1.0 + 10 ** ((500 - agent.current_ranking.overall_score) / 400))
            actual_score = (len(rankings) - i) / len(rankings)  # Normalized rank position
            
            k_factor = 32  # Learning rate
            score_change = k_factor * (actual_score - expected_score)
            
            agent.current_ranking.overall_score += score_change
            agent.current_ranking.rank = i + 1
        
        # Sort all agents by overall score for global ranking
        all_agents = sorted(self.agents.values(), 
                          key=lambda a: a.current_ranking.overall_score, reverse=True)
        
        for i, agent in enumerate(all_agents):
            agent.current_ranking.rank = i + 1
    
    async def trigger_improvement_cycle(self, results: List[CompetitionResult], 
                                      winner: Optional[CompetitionResult]) -> None:
        """Trigger improvement cycle for all participating agents"""
        
        if not winner:
            return
        
        for result in results:
            agent = self.agents[result.agent_id]
            won = (result.agent_id == winner.agent_id)
            agent.improve_performance(result, won)
    
    async def run_tournament(self, tournament_name: str, tasks: List[str], 
                           competition_type: CompetitionType) -> Dict[str, Any]:
        """Run a tournament with multiple tasks"""
        
        print(f"\nRUNNING TOURNAMENT: {tournament_name}")
        print("=" * 60)
        
        tournament_results = []
        agent_scores = {agent_id: 0 for agent_id in self.agents.keys()}
        
        for i, task in enumerate(tasks):
            print(f"\nRound {i+1}: {task}")
            competition_result = await self.run_competition(task, competition_type)
            tournament_results.append(competition_result)
            
            # Award points based on ranking
            if competition_result.get("rankings"):
                for j, result in enumerate(competition_result["rankings"]):
                    points = len(competition_result["rankings"]) - j
                    agent_scores[result.agent_id] += points
        
        # Determine tournament winner
        tournament_winner = max(agent_scores.items(), key=lambda x: x[1])
        
        print(f"\nTOURNAMENT RESULTS:")
        print(f"Winner: {tournament_winner[0]} with {tournament_winner[1]} points")
        
        # Show final scores
        sorted_scores = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (agent_id, score) in enumerate(sorted_scores[:3]):
            print(f"  {i+1}. {agent_id}: {score} points")
        
        return {
            "tournament_name": tournament_name,
            "winner": tournament_winner[0],
            "final_scores": dict(sorted_scores),
            "competition_results": tournament_results
        }
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current agent leaderboard"""
        
        sorted_agents = sorted(self.agents.values(), 
                             key=lambda a: a.current_ranking.overall_score, reverse=True)
        
        leaderboard = []
        for i, agent in enumerate(sorted_agents):
            leaderboard.append({
                "rank": i + 1,
                "agent_id": agent.agent_id,
                "score": agent.current_ranking.overall_score,
                "wins": agent.current_ranking.wins,
                "competitions": agent.current_ranking.competitions,
                "win_rate": agent.current_ranking.win_rate,
                "recent_avg": (sum(agent.current_ranking.recent_performance) / 
                             len(agent.current_ranking.recent_performance) 
                             if agent.current_ranking.recent_performance else 0.0)
            })
        
        return leaderboard

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_ride_sharing_competition():
    """Demo: Ride-sharing drivers competing for rides"""
    print("\nDEMO 1: RIDE-SHARING DRIVER COMPETITION")
    print("=" * 50)
    
    system = CompetitiveSystem("ride_sharing")
    
    # Add competing drivers
    drivers = ["speedy_sam", "quality_quinn", "efficient_emma", "innovative_ivan", "reliable_rick"]
    for driver in drivers:
        agent = CompetitiveAgent(driver, ["driving", "navigation"])
        system.add_agent(agent)
    
    # Run several ride competitions
    rides = [
        "Airport pickup - rush hour",
        "Downtown to suburbs",
        "Hospital emergency run",
        "Business district pickup"
    ]
    
    for ride in rides:
        await system.run_competition(ride, CompetitionType.SPEED)
        await asyncio.sleep(0.1)  # Brief pause between competitions
    
    # Show final leaderboard
    leaderboard = system.get_leaderboard()
    print(f"\nFINAL DRIVER RANKINGS:")
    for entry in leaderboard[:3]:
        print(f"  {entry['rank']}. {entry['agent_id']}: {entry['score']:.1f} points "
              f"({entry['win_rate']:.1%} win rate)")

async def demo_freelancer_marketplace():
    """Demo: Freelancers competing for projects"""
    print("\nDEMO 2: FREELANCER MARKETPLACE COMPETITION")
    print("=" * 50)
    
    system = CompetitiveSystem("freelancer_market")
    
    # Add freelancers with different specialties
    freelancers = ["dev_alice", "designer_bob", "writer_carol", "analyst_dave"]
    for freelancer in freelancers:
        agent = CompetitiveAgent(freelancer, ["programming", "design", "writing"])
        system.add_agent(agent)
    
    # Run tournament with different project types
    projects = [
        "Build e-commerce website",
        "Design mobile app UI",
        "Write technical documentation",
        "Analyze user data trends"
    ]
    
    tournament_result = await system.run_tournament("Q1 Projects", projects, CompetitionType.QUALITY)
    
    print(f"\nTournament winner gets premium project access!")

async def demo_ai_research_competition():
    """Demo: AI research teams competing for breakthroughs"""
    print("\nDEMO 3: AI RESEARCH TEAM COMPETITION")
    print("=" * 50)
    
    system = CompetitiveSystem("ai_research")
    
    # Add research teams
    teams = ["deepmind_team", "openai_team", "google_team", "meta_team"]
    for team in teams:
        agent = CompetitiveAgent(team, ["research", "machine_learning", "innovation"])
        system.add_agent(agent)
    
    # Research challenges
    challenges = [
        "Breakthrough in language understanding",
        "Efficient training algorithm",
        "Novel architecture design"
    ]
    
    for challenge in challenges:
        await system.run_competition(challenge, CompetitionType.INNOVATION)
    
    # Show research leaderboard
    leaderboard = system.get_leaderboard()
    print(f"\nRESEARCH TEAM RANKINGS:")
    for entry in leaderboard:
        print(f"  {entry['rank']}. {entry['agent_id']}: "
              f"{entry['competitions']} papers, {entry['win_rate']:.1%} breakthrough rate")

async def main():
    """
    Demonstrate Competitive Agent Systems for performance optimization
    
    WHAT YOU'LL LEARN:
    ================
    1. How competition drives continuous improvement and innovation
    2. How to design fair competition rules and evaluation metrics
    3. How to implement ranking systems and performance tracking
    4. How competitive pressure optimizes system performance
    5. How to balance competition with collaboration
    
    REAL WORLD APPLICATIONS:
    =======================
    - Ride-sharing and delivery platforms (Uber, DoorDash)
    - Freelancer marketplaces (Upwork, Fiverr)
    - Trading and financial markets
    - Research and academic competitions
    - Sales teams and performance incentives
    - Gaming and esports tournaments
    """
    
    print("COMPETITIVE AGENT SYSTEMS DEMONSTRATION")
    print("This shows how competition drives performance and innovation!")
    
    await demo_ride_sharing_competition()
    await demo_freelancer_marketplace()
    await demo_ai_research_competition()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Competition drives continuous improvement and excellence")
    print("✓ Fair evaluation metrics ensure legitimate competition")
    print("✓ Ranking systems provide clear performance feedback")
    print("✓ Performance tracking enables data-driven optimization")
    print("✓ Competitive systems naturally evolve toward higher performance")

if __name__ == "__main__":
    asyncio.run(main())
