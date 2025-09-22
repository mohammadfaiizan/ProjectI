#!/usr/bin/env python3
"""
Voting and Decision Making: Democratic Agent Coordination
========================================================

WHAT IS THE PROBLEM?
==================
When multiple agents need to make collective decisions, disagreements can paralyze the system. Voting provides fair, systematic decision-making.

Example: Committee Chaos
BAD APPROACH:
- No clear decision process
- Loudest voices dominate
- Decisions made by whoever speaks first
- No accountability or fairness
- Endless debates without resolution

REAL WORLD EXAMPLE:
=================
How does a corporate board make decisions?

STRUCTURED VOTING PROCESS:
1. PROPOSAL: Someone proposes a decision
2. DISCUSSION: Board members discuss pros/cons
3. VOTING: Each member casts their vote
4. COUNTING: Votes are tallied transparently
5. DECISION: Majority (or supermajority) wins
6. IMPLEMENTATION: Decision is binding on all

THE ALGORITHM:
=============
1. SETUP: Define voting rules and participants
2. PROPOSE: Submit proposals for voting
3. DELIBERATE: Allow discussion period
4. VOTE: Collect votes from all participants
5. COUNT: Tally votes according to rules
6. DECIDE: Determine winning proposal
7. IMPLEMENT: Execute the collective decision

WHY IS THIS POWERFUL?
===================
- Fair representation of all participants
- Clear, transparent decision process
- Democratic legitimacy and buy-in
- Scalable to large groups
- Prevents deadlock and endless debate
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class VotingMethod(Enum):
    SIMPLE_MAJORITY = "simple_majority"
    SUPERMAJORITY = "supermajority"
    UNANIMOUS = "unanimous"
    RANKED_CHOICE = "ranked_choice"
    APPROVAL = "approval"
    WEIGHTED = "weighted"

class VoteType(Enum):
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"

@dataclass
class Proposal:
    """A proposal to be voted on"""
    id: str
    title: str
    description: str
    proposed_by: str
    timestamp: float = field(default_factory=time.time)
    voting_deadline: Optional[float] = None

@dataclass
class Vote:
    """A single vote cast by an agent"""
    voter_id: str
    proposal_id: str
    vote_type: VoteType
    weight: float = 1.0
    timestamp: float = field(default_factory=time.time)
    reasoning: Optional[str] = None

@dataclass
class VotingResult:
    """Result of a voting session"""
    proposal: Proposal
    votes: List[Vote]
    yes_votes: int
    no_votes: int
    abstentions: int
    total_weight: float
    yes_weight: float
    no_weight: float
    passed: bool
    margin: float

class VotingAgent:
    """Agent that participates in voting processes"""
    
    def __init__(self, agent_id: str, voting_weight: float = 1.0, decision_style: str = "analytical"):
        self.agent_id = agent_id
        self.voting_weight = voting_weight
        self.decision_style = decision_style  # analytical, intuitive, conservative, progressive
        
        # Voting behavior
        self.vote_history: List[Vote] = []
        self.proposal_history: List[Proposal] = []
        
        # Decision-making preferences
        self.risk_tolerance = random.uniform(0.3, 0.9)
        self.change_acceptance = random.uniform(0.2, 0.8)
        self.evidence_requirements = random.uniform(0.4, 0.9)
    
    async def evaluate_proposal(self, proposal: Proposal) -> Dict[str, Any]:
        """Evaluate a proposal and form an opinion"""
        
        print(f"  {self.agent_id} evaluating: {proposal.title}")
        
        # Simulate evaluation time
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Evaluation factors
        evaluation = {
            "clarity": random.uniform(0.3, 1.0),
            "feasibility": random.uniform(0.4, 0.9),
            "benefit": random.uniform(0.2, 1.0),
            "risk": random.uniform(0.1, 0.8),
            "cost": random.uniform(0.2, 0.9)
        }
        
        # Decision style influences evaluation
        if self.decision_style == "conservative":
            evaluation["risk"] *= 1.3  # See more risk
            evaluation["benefit"] *= 0.8  # See less benefit
        elif self.decision_style == "progressive":
            evaluation["benefit"] *= 1.2  # See more benefit
            evaluation["risk"] *= 0.7  # See less risk
        
        # Calculate overall score
        overall_score = (
            evaluation["clarity"] * 0.1 +
            evaluation["feasibility"] * 0.2 +
            evaluation["benefit"] * 0.3 -
            evaluation["risk"] * 0.2 -
            evaluation["cost"] * 0.2
        )
        
        evaluation["overall_score"] = overall_score
        return evaluation
    
    async def cast_vote(self, proposal: Proposal, voting_system: 'VotingSystem') -> Vote:
        """Cast vote on a proposal"""
        
        # Evaluate the proposal
        evaluation = await self.evaluate_proposal(proposal)
        
        # Determine vote based on evaluation
        if evaluation["overall_score"] > 0.6:
            vote_type = VoteType.YES
            reasoning = "Proposal shows clear benefits and acceptable risk"
        elif evaluation["overall_score"] < 0.3:
            vote_type = VoteType.NO
            reasoning = "Proposal has significant concerns or insufficient benefit"
        else:
            # Borderline cases depend on agent characteristics
            if self.risk_tolerance < 0.5:
                vote_type = VoteType.NO
                reasoning = "Uncertain outcomes pose too much risk"
            elif random.random() < 0.3:
                vote_type = VoteType.ABSTAIN
                reasoning = "Need more information to make informed decision"
            else:
                vote_type = VoteType.YES
                reasoning = "Potential benefits outweigh concerns"
        
        vote = Vote(
            voter_id=self.agent_id,
            proposal_id=proposal.id,
            vote_type=vote_type,
            weight=self.voting_weight,
            reasoning=reasoning
        )
        
        self.vote_history.append(vote)
        print(f"    {self.agent_id} votes {vote_type.value}: {reasoning}")
        
        return vote
    
    async def propose_idea(self, title: str, description: str) -> Proposal:
        """Propose a new idea for voting"""
        
        proposal = Proposal(
            id=f"prop_{self.agent_id}_{int(time.time())}",
            title=title,
            description=description,
            proposed_by=self.agent_id,
            voting_deadline=time.time() + 300  # 5 minutes
        )
        
        self.proposal_history.append(proposal)
        return proposal

class VotingSystem:
    """
    System for managing democratic decision making
    
    EXAMPLE USAGE:
    =============
    # Create voting system
    system = VotingSystem("corporate_board")
    
    # Add voting members
    for i in range(7):
        agent = VotingAgent(f"board_member_{i}", 1.0)
        system.add_voter(agent)
    
    # Hold vote on proposal
    result = await system.conduct_vote("Approve new product launch")
    """
    
    def __init__(self, system_id: str, voting_method: VotingMethod = VotingMethod.SIMPLE_MAJORITY):
        self.system_id = system_id
        self.voting_method = voting_method
        self.voters: Dict[str, VotingAgent] = {}
        
        # Voting sessions
        self.active_votes: Dict[str, Dict[str, Any]] = {}
        self.completed_votes: List[VotingResult] = []
        
        # Voting rules
        self.quorum_requirement = 0.5  # 50% participation required
        self.supermajority_threshold = 0.67  # 67% for supermajority
        self.discussion_period = 30.0  # seconds
        self.voting_period = 60.0  # seconds
    
    def add_voter(self, agent: VotingAgent) -> None:
        """Add voting agent to the system"""
        self.voters[agent.agent_id] = agent
        print(f"Added voter: {agent.agent_id} (weight: {agent.voting_weight})")
    
    async def conduct_vote(self, proposal_title: str, proposal_description: str = "",
                         proposer_id: str = "system") -> VotingResult:
        """Conduct a complete voting process"""
        
        print(f"\nCONDUCTING VOTE: {proposal_title}")
        print("=" * 50)
        
        # Create proposal
        proposal = Proposal(
            id=f"vote_{int(time.time())}",
            title=proposal_title,
            description=proposal_description,
            proposed_by=proposer_id
        )
        
        # Discussion period
        print(f"Discussion period: {self.discussion_period} seconds")
        await self.discussion_phase(proposal)
        
        # Voting period
        print(f"\nVoting period: {self.voting_period} seconds")
        votes = await self.voting_phase(proposal)
        
        # Count votes and determine result
        result = self.count_votes(proposal, votes)
        
        # Record result
        self.completed_votes.append(result)
        
        print(f"\nVOTING RESULT:")
        print(f"- {result.yes_votes} Yes, {result.no_votes} No, {result.abstentions} Abstain")
        print(f"- Result: {'PASSED' if result.passed else 'FAILED'}")
        print(f"- Margin: {result.margin:.1%}")
        
        return result
    
    async def discussion_phase(self, proposal: Proposal) -> None:
        """Allow discussion period before voting"""
        
        # Simulate discussion where agents evaluate the proposal
        discussion_tasks = []
        for voter in self.voters.values():
            task = voter.evaluate_proposal(proposal)
            discussion_tasks.append(task)
        
        # Wait for all evaluations (discussion)
        await asyncio.gather(*discussion_tasks)
        
        print("Discussion phase completed")
    
    async def voting_phase(self, proposal: Proposal) -> List[Vote]:
        """Collect votes from all participants"""
        
        votes = []
        voting_tasks = []
        
        # All voters cast their votes
        for voter in self.voters.values():
            task = voter.cast_vote(proposal, self)
            voting_tasks.append(task)
        
        # Collect all votes
        votes = await asyncio.gather(*voting_tasks)
        
        print(f"\nVoting completed: {len(votes)} votes cast")
        return votes
    
    def count_votes(self, proposal: Proposal, votes: List[Vote]) -> VotingResult:
        """Count votes and determine outcome"""
        
        yes_votes = 0
        no_votes = 0
        abstentions = 0
        yes_weight = 0.0
        no_weight = 0.0
        total_weight = 0.0
        
        for vote in votes:
            total_weight += vote.weight
            
            if vote.vote_type == VoteType.YES:
                yes_votes += 1
                yes_weight += vote.weight
            elif vote.vote_type == VoteType.NO:
                no_votes += 1
                no_weight += vote.weight
            else:  # ABSTAIN
                abstentions += 1
        
        # Check quorum
        participation_rate = len(votes) / len(self.voters)
        if participation_rate < self.quorum_requirement:
            print(f"Quorum not met: {participation_rate:.1%} < {self.quorum_requirement:.1%}")
            passed = False
            margin = 0.0
        else:
            # Determine if proposal passed based on voting method
            if self.voting_method == VotingMethod.SIMPLE_MAJORITY:
                passed = yes_weight > no_weight
                margin = (yes_weight - no_weight) / total_weight if total_weight > 0 else 0
            elif self.voting_method == VotingMethod.SUPERMAJORITY:
                yes_ratio = yes_weight / total_weight if total_weight > 0 else 0
                passed = yes_ratio >= self.supermajority_threshold
                margin = yes_ratio - self.supermajority_threshold
            elif self.voting_method == VotingMethod.UNANIMOUS:
                passed = no_votes == 0 and abstentions == 0
                margin = 1.0 if passed else -1.0
            else:
                # Default to simple majority
                passed = yes_weight > no_weight
                margin = (yes_weight - no_weight) / total_weight if total_weight > 0 else 0
        
        return VotingResult(
            proposal=proposal,
            votes=votes,
            yes_votes=yes_votes,
            no_votes=no_votes,
            abstentions=abstentions,
            total_weight=total_weight,
            yes_weight=yes_weight,
            no_weight=no_weight,
            passed=passed,
            margin=margin
        )
    
    async def conduct_multiple_votes(self, proposals: List[Tuple[str, str]]) -> List[VotingResult]:
        """Conduct multiple votes in sequence"""
        
        results = []
        
        for title, description in proposals:
            result = await self.conduct_vote(title, description)
            results.append(result)
            
            # Brief pause between votes
            await asyncio.sleep(0.5)
        
        return results
    
    def get_voting_summary(self) -> Dict[str, Any]:
        """Get comprehensive voting system summary"""
        
        total_votes = len(self.completed_votes)
        passed_votes = len([v for v in self.completed_votes if v.passed])
        
        return {
            "system_id": self.system_id,
            "voting_method": self.voting_method.value,
            "total_voters": len(self.voters),
            "total_votes_held": total_votes,
            "votes_passed": passed_votes,
            "pass_rate": passed_votes / total_votes if total_votes > 0 else 0,
            "quorum_requirement": self.quorum_requirement,
            "voter_summaries": {v.agent_id: len(v.vote_history) for v in self.voters.values()}
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_corporate_board_voting():
    """Demo: Corporate board decision making"""
    print("\nDEMO 1: CORPORATE BOARD VOTING")
    print("=" * 50)
    
    # Create board voting system
    board = VotingSystem("corporate_board", VotingMethod.SIMPLE_MAJORITY)
    
    # Add board members with different characteristics
    board_members = [
        ("ceo", 1.0, "progressive"),
        ("cfo", 1.0, "conservative"),
        ("cto", 1.0, "analytical"),
        ("head_sales", 1.0, "progressive"),
        ("head_hr", 1.0, "analytical"),
        ("independent_1", 1.0, "conservative"),
        ("independent_2", 1.0, "analytical")
    ]
    
    for member_id, weight, style in board_members:
        agent = VotingAgent(member_id, weight, style)
        board.add_voter(agent)
    
    # Vote on strategic decisions
    strategic_proposals = [
        ("Acquire competitor for $50M", "Horizontal acquisition to gain market share"),
        ("Launch new product line", "Expand into adjacent market with 18-month timeline"),
        ("Implement remote work policy", "Allow 100% remote work for all employees")
    ]
    
    results = await board.conduct_multiple_votes(strategic_proposals)
    
    print(f"\nBoard Voting Summary:")
    summary = board.get_voting_summary()
    print(f"- Decisions voted on: {summary['total_votes_held']}")
    print(f"- Decisions passed: {summary['votes_passed']}")
    print(f"- Success rate: {summary['pass_rate']:.1%}")

async def demo_technical_committee():
    """Demo: Technical committee standards voting"""
    print("\nDEMO 2: TECHNICAL STANDARDS COMMITTEE")
    print("=" * 50)
    
    # Create technical committee with supermajority requirement
    committee = VotingSystem("tech_committee", VotingMethod.SUPERMAJORITY)
    committee.supermajority_threshold = 0.75  # 75% required
    
    # Add committee members from different organizations
    organizations = ["google", "microsoft", "amazon", "meta", "apple", "netflix", "uber", "tesla"]
    for org in organizations:
        agent = VotingAgent(f"{org}_rep", 1.0, "analytical")
        committee.add_voter(agent)
    
    # Vote on technical standards
    standards_proposals = [
        ("Adopt new security protocol", "Implement enhanced encryption standard"),
        ("Deprecate legacy API", "Phase out support for API v1.0 over 24 months"),
        ("Standardize data format", "Adopt JSON-LD as mandatory data interchange format")
    ]
    
    results = await committee.conduct_multiple_votes(standards_proposals)
    
    print(f"\nTechnical Committee Results:")
    for i, result in enumerate(results):
        proposal = result.proposal
        print(f"- {proposal.title}: {'ADOPTED' if result.passed else 'REJECTED'} "
              f"({result.yes_votes}/{len(committee.voters)} votes)")

async def demo_community_voting():
    """Demo: Community decision making with weighted voting"""
    print("\nDEMO 3: COMMUNITY WEIGHTED VOTING")
    print("=" * 50)
    
    # Create community voting system
    community = VotingSystem("homeowners_association", VotingMethod.WEIGHTED)
    
    # Add community members with voting weights based on property value
    residents = [
        ("resident_1", 1.5, "conservative"),  # Larger property
        ("resident_2", 1.0, "analytical"),
        ("resident_3", 1.0, "progressive"),
        ("resident_4", 2.0, "conservative"),  # Largest property
        ("resident_5", 1.0, "analytical"),
        ("resident_6", 0.5, "progressive"),  # Smaller property
        ("resident_7", 1.0, "conservative"),
        ("resident_8", 1.5, "analytical")
    ]
    
    for resident_id, weight, style in residents:
        agent = VotingAgent(resident_id, weight, style)
        community.add_voter(agent)
    
    # Vote on community issues
    community_proposals = [
        ("Install community pool", "Build swimming pool and recreation area"),
        ("Increase HOA fees by 15%", "Raise monthly fees to fund improvements"),
        ("Implement parking restrictions", "Limit street parking to residents only")
    ]
    
    results = await community.conduct_multiple_votes(community_proposals)
    
    print(f"\nCommunity Voting Summary:")
    for result in results:
        total_weight = sum(agent.voting_weight for agent in community.voters.values())
        yes_percentage = (result.yes_weight / total_weight) * 100
        print(f"- {result.proposal.title}: {yes_percentage:.1f}% support "
              f"({'APPROVED' if result.passed else 'REJECTED'})")

async def main():
    """
    Demonstrate Voting and Decision Making for democratic coordination
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement democratic decision-making processes
    2. How different voting methods affect outcomes
    3. How to handle weighted voting and representation
    4. How to ensure fairness and transparency in collective decisions
    5. How voting systems scale to large groups
    
    REAL WORLD APPLICATIONS:
    =======================
    - Corporate governance and board decisions
    - Technical standards committees and open source projects
    - Government and legislative processes
    - Community and homeowners association decisions
    - Organizational policy making and employee voting
    - Distributed autonomous organizations (DAOs)
    """
    
    print("VOTING AND DECISION MAKING DEMONSTRATION")
    print("This shows how democratic processes enable fair collective decisions!")
    
    await demo_corporate_board_voting()
    await demo_technical_committee()
    await demo_community_voting()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Voting provides fair and transparent decision-making")
    print("✓ Different voting methods suit different organizational needs")
    print("✓ Weighted voting can represent stakeholder interests")
    print("✓ Democratic processes build legitimacy and buy-in")
    print("✓ Structured voting prevents deadlock and enables progress")

if __name__ == "__main__":
    asyncio.run(main())
