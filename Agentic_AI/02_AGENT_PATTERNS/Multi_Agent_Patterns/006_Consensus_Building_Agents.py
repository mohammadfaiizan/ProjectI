#!/usr/bin/env python3
"""
Consensus Building Agents: Collaborative Decision Making
=======================================================

WHAT IS THE PROBLEM?
==================
When a group needs to make decisions together, disagreements and conflicts can paralyze progress. Some people dominate, others withdraw, and the group either can't decide or makes poor decisions.

Example: Committee That Can't Decide
BAD APPROACH:
- 10 people in a meeting, everyone talks at once
- Loudest voices dominate the discussion
- Quiet members don't share their expertise
- No structured process for reaching agreement
- Meetings drag on for hours without decisions
- People leave frustrated, decisions get made poorly or not at all

REAL WORLD EXAMPLE:
=================
How does the Supreme Court actually reach decisions?

STRUCTURED CONSENSUS PROCESS:
1. CASE PRESENTATION: Clear presentation of the issue
2. INDIVIDUAL RESEARCH: Each justice researches independently
3. CONFERENCE DISCUSSION: Structured discussion of viewpoints
4. INITIAL POSITIONS: Each justice states their position
5. DELIBERATION: Justices discuss reasoning and concerns
6. OPINION DRAFTING: Initial majority opinion is drafted
7. CIRCULATION: Draft opinions circulate for feedback
8. REVISION: Opinions are revised based on input
9. FINAL VOTE: Final positions are locked in
10. PUBLICATION: Final decision is published with reasoning

KEY CONSENSUS PRINCIPLES:
- Everyone's voice is heard
- Decisions are based on reasoning, not politics
- Process is structured and fair
- Minority opinions are respected and recorded
- Final decisions have clear justification

THE ALGORITHM:
=============
1. FRAME: Clearly define the decision to be made
2. GATHER: Collect input from all stakeholders
3. EXPLORE: Examine all viewpoints and concerns
4. CONVERGE: Find areas of agreement and disagreement
5. NEGOTIATE: Work through differences constructively
6. VALIDATE: Test proposed solutions with the group
7. DECIDE: Reach final consensus or acceptable compromise
8. COMMIT: Ensure everyone commits to supporting the decision

PSEUDO CODE:
===========
class ConsensusProcess:
    def __init__(self, participants):
        self.participants = participants
        self.positions = {}
        self.concerns = {}
        self.proposals = []
        self.consensus_threshold = 0.8  # 80% agreement needed
    
    def build_consensus(self, decision_topic):
        # Phase 1: Gather individual positions
        for participant in self.participants:
            position = participant.get_position(decision_topic)
            concerns = participant.get_concerns(decision_topic)
            self.positions[participant] = position
            self.concerns[participant] = concerns
        
        # Phase 2: Find common ground
        common_elements = self.find_common_ground()
        
        # Phase 3: Address differences
        remaining_differences = self.identify_differences()
        
        while remaining_differences and not self.has_consensus():
            # Generate compromise proposals
            proposals = self.generate_proposals(remaining_differences)
            
            # Get feedback on proposals
            for proposal in proposals:
                feedback = self.gather_feedback(proposal)
                refined_proposal = self.refine_proposal(proposal, feedback)
                
                if self.evaluate_consensus(refined_proposal):
                    return refined_proposal
            
            # If no proposal works, facilitate negotiation
            remaining_differences = self.facilitate_negotiation()
        
        return self.final_consensus_decision()

WHY IS THIS ESSENTIAL?
====================
- Ensures all perspectives are considered
- Builds buy-in and commitment from all participants
- Produces higher quality decisions through diverse input
- Reduces conflict and resistance to implementation
- Creates shared ownership of outcomes
- Develops group decision-making capability over time
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

class ConsensusStage(Enum):
    FRAMING = "framing"
    INFORMATION_GATHERING = "information_gathering" 
    POSITION_SHARING = "position_sharing"
    EXPLORATION = "exploration"
    NEGOTIATION = "negotiation"
    VALIDATION = "validation"
    DECISION = "decision"
    COMMITMENT = "commitment"

class AgreementLevel(Enum):
    STRONG_SUPPORT = 5
    SUPPORT = 4
    NEUTRAL = 3
    CONCERNS = 2
    STRONG_OPPOSITION = 1

class ParticipantRole(Enum):
    FACILITATOR = "facilitator"
    STAKEHOLDER = "stakeholder"
    EXPERT = "expert"
    DECISION_MAKER = "decision_maker"
    OBSERVER = "observer"

@dataclass
class Position:
    """Participant's position on a decision topic"""
    participant_id: str
    topic: str
    stance: str
    reasoning: List[str]
    concerns: List[str]
    requirements: List[str]
    agreement_level: AgreementLevel
    flexibility: float  # 0.0 to 1.0 - how willing to compromise

@dataclass
class Proposal:
    """Proposed solution or decision"""
    id: str
    title: str
    description: str
    key_elements: List[str]
    addresses_concerns: List[str]
    trade_offs: List[str]
    proposed_by: str
    support_scores: Dict[str, AgreementLevel] = field(default_factory=dict)
    refinements: List[str] = field(default_factory=list)

@dataclass
class ConsensusSession:
    """Session for building consensus on a specific topic"""
    id: str
    topic: str
    facilitator_id: str
    participants: List[str]
    stage: ConsensusStage
    positions: Dict[str, Position]
    proposals: List[Proposal]
    discussion_history: List[Dict[str, Any]] = field(default_factory=list)
    consensus_threshold: float = 0.8
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    final_decision: Optional[Proposal] = None

class ConsensusAgent:
    """
    Agent that participates in consensus building processes
    """
    
    def __init__(self, agent_id: str, role: ParticipantRole, expertise_areas: List[str]):
        self.agent_id = agent_id
        self.role = role
        self.expertise_areas = expertise_areas
        
        # Consensus participation
        self.active_sessions: Dict[str, ConsensusSession] = {}
        self.position_history: List[Position] = []
        self.compromise_willingness = 0.7  # How willing to compromise (0.0 to 1.0)
        
        # Communication and collaboration
        self.communication_style = "collaborative"  # collaborative, assertive, analytical
        self.influence_factors = ["expertise", "reasoning", "relationship"]
        
        # Performance tracking
        self.consensus_participation_count = 0
        self.successful_consensus_count = 0
        self.satisfaction_with_outcomes = []
    
    async def participate_in_consensus(self, session: ConsensusSession, 
                                     consensus_system: 'ConsensusSystem') -> Dict[str, Any]:
        """
        Participate in a consensus building session
        
        Args:
            session: The consensus session to participate in
            consensus_system: The system managing the consensus process
            
        Returns:
            Participation results and final position
        """
        
        print(f"\n{self.agent_id} PARTICIPATING IN CONSENSUS")
        print(f"Topic: {session.topic}")
        print(f"Role: {self.role.value}")
        print("-" * 50)
        
        self.active_sessions[session.id] = session
        
        # Participate through all stages
        participation_results = {}
        
        while session.stage != ConsensusStage.COMMITMENT:
            if session.stage == ConsensusStage.FRAMING:
                result = await self.participate_in_framing(session, consensus_system)
            elif session.stage == ConsensusStage.INFORMATION_GATHERING:
                result = await self.gather_information(session, consensus_system)
            elif session.stage == ConsensusStage.POSITION_SHARING:
                result = await self.share_position(session, consensus_system)
            elif session.stage == ConsensusStage.EXPLORATION:
                result = await self.explore_alternatives(session, consensus_system)
            elif session.stage == ConsensusStage.NEGOTIATION:
                result = await self.participate_in_negotiation(session, consensus_system)
            elif session.stage == ConsensusStage.VALIDATION:
                result = await self.validate_proposals(session, consensus_system)
            elif session.stage == ConsensusStage.DECISION:
                result = await self.make_final_decision(session, consensus_system)
            
            participation_results[session.stage.value] = result
            
            # Wait for stage to complete
            await asyncio.sleep(0.2)
        
        # Final commitment
        commitment_result = await self.commit_to_decision(session, consensus_system)
        participation_results["commitment"] = commitment_result
        
        # Update tracking
        self.consensus_participation_count += 1
        if session.final_decision:
            self.successful_consensus_count += 1
        
        del self.active_sessions[session.id]
        
        return {
            "participant": self.agent_id,
            "session_topic": session.topic,
            "participation_results": participation_results,
            "final_satisfaction": commitment_result.get("satisfaction", 0.5),
            "consensus_achieved": session.final_decision is not None
        }
    
    async def participate_in_framing(self, session: ConsensusSession, 
                                   consensus_system: 'ConsensusSystem') -> Dict[str, Any]:
        """Participate in framing the decision topic"""
        
        print(f"  {self.agent_id} participating in framing stage")
        
        # Contribute to framing based on role
        if self.role == ParticipantRole.FACILITATOR:
            framing_contribution = await self.facilitate_framing(session)
        elif self.role == ParticipantRole.EXPERT:
            framing_contribution = await self.provide_expert_framing(session)
        else:
            framing_contribution = await self.provide_stakeholder_framing(session)
        
        return framing_contribution
    
    async def facilitate_framing(self, session: ConsensusSession) -> Dict[str, Any]:
        """Facilitate the framing process"""
        await asyncio.sleep(0.1)
        
        return {
            "framing_questions": [
                "What specific decision do we need to make?",
                "What are the key criteria for a good decision?",
                "Who are the key stakeholders affected?",
                "What are the constraints and requirements?"
            ],
            "process_structure": "Structured discussion with equal participation",
            "success_criteria": "Clear problem definition and decision criteria"
        }
    
    async def provide_expert_framing(self, session: ConsensusSession) -> Dict[str, Any]:
        """Provide expert perspective on framing"""
        await asyncio.sleep(0.1)
        
        relevant_expertise = [area for area in self.expertise_areas 
                            if area.lower() in session.topic.lower()]
        
        return {
            "expert_perspective": f"From {relevant_expertise} standpoint, key considerations are...",
            "technical_constraints": ["constraint_1", "constraint_2"],
            "critical_factors": ["factor_1", "factor_2", "factor_3"],
            "recommended_approach": "Evidence-based decision making"
        }
    
    async def provide_stakeholder_framing(self, session: ConsensusSession) -> Dict[str, Any]:
        """Provide stakeholder perspective on framing"""
        await asyncio.sleep(0.1)
        
        return {
            "stakeholder_interests": ["interest_1", "interest_2"],
            "impact_concerns": ["concern_1", "concern_2"],
            "desired_outcomes": ["outcome_1", "outcome_2"],
            "implementation_considerations": ["consideration_1", "consideration_2"]
        }
    
    async def gather_information(self, session: ConsensusSession, 
                               consensus_system: 'ConsensusSystem') -> Dict[str, Any]:
        """Gather and share relevant information"""
        
        print(f"  {self.agent_id} gathering information")
        
        # Simulate information gathering based on expertise
        information_gathered = {}
        
        for expertise in self.expertise_areas:
            if expertise.lower() in session.topic.lower():
                info = await self.gather_expert_information(session.topic, expertise)
                information_gathered[expertise] = info
        
        return {
            "information_contributed": information_gathered,
            "sources_consulted": ["source_1", "source_2", "source_3"],
            "confidence_level": 0.8,
            "information_gaps_identified": ["gap_1", "gap_2"]
        }
    
    async def gather_expert_information(self, topic: str, expertise: str) -> Dict[str, Any]:
        """Gather information in area of expertise"""
        await asyncio.sleep(0.1)
        
        return {
            "key_facts": [f"Fact 1 about {expertise}", f"Fact 2 about {expertise}"],
            "best_practices": [f"Best practice 1", f"Best practice 2"],
            "potential_issues": [f"Issue 1", f"Issue 2"],
            "recommendations": [f"Recommendation 1", f"Recommendation 2"]
        }
    
    async def share_position(self, session: ConsensusSession, 
                           consensus_system: 'ConsensusSystem') -> Dict[str, Any]:
        """Share initial position on the topic"""
        
        print(f"  {self.agent_id} sharing position")
        
        # Develop position based on role and expertise
        position = await self.develop_position(session.topic)
        
        # Add to session
        session.positions[self.agent_id] = position
        self.position_history.append(position)
        
        print(f"    Position: {position.stance}")
        print(f"    Agreement level: {position.agreement_level.name}")
        
        return {
            "position_shared": position.stance,
            "key_reasoning": position.reasoning,
            "main_concerns": position.concerns,
            "flexibility": position.flexibility
        }
    
    async def develop_position(self, topic: str) -> Position:
        """Develop position on the topic"""
        
        # Simulate position development
        await asyncio.sleep(0.1)
        
        # Position varies by role and topic
        if "implement" in topic.lower():
            if self.role == ParticipantRole.EXPERT:
                stance = "Support with technical modifications"
                agreement_level = AgreementLevel.SUPPORT
                flexibility = 0.6
            else:
                stance = "Support with stakeholder considerations"
                agreement_level = AgreementLevel.SUPPORT
                flexibility = 0.8
        else:
            stance = "Need more information before deciding"
            agreement_level = AgreementLevel.NEUTRAL
            flexibility = 0.7
        
        return Position(
            participant_id=self.agent_id,
            topic=topic,
            stance=stance,
            reasoning=[
                f"Based on {self.role.value} perspective",
                f"Considering {self.expertise_areas} expertise",
                "Weighing costs and benefits"
            ],
            concerns=[
                "Implementation complexity",
                "Resource requirements",
                "Stakeholder impact"
            ],
            requirements=[
                "Clear success metrics",
                "Adequate resources",
                "Stakeholder buy-in"
            ],
            agreement_level=agreement_level,
            flexibility=flexibility
        )
    
    async def explore_alternatives(self, session: ConsensusSession, 
                                 consensus_system: 'ConsensusSystem') -> Dict[str, Any]:
        """Explore alternative solutions and approaches"""
        
        print(f"  {self.agent_id} exploring alternatives")
        
        # Generate alternative proposals
        alternatives = await self.generate_alternatives(session)
        
        # Evaluate existing proposals
        evaluations = {}
        for proposal in session.proposals:
            evaluation = await self.evaluate_proposal(proposal, session)
            evaluations[proposal.id] = evaluation
        
        return {
            "alternatives_proposed": len(alternatives),
            "proposal_evaluations": evaluations,
            "creative_suggestions": ["suggestion_1", "suggestion_2"],
            "synthesis_opportunities": ["opportunity_1", "opportunity_2"]
        }
    
    async def generate_alternatives(self, session: ConsensusSession) -> List[Proposal]:
        """Generate alternative proposals"""
        
        await asyncio.sleep(0.1)
        
        alternatives = []
        
        # Generate alternatives based on expertise
        for i, expertise in enumerate(self.expertise_areas):
            if expertise.lower() in session.topic.lower():
                alternative = Proposal(
                    id=f"alt_{self.agent_id}_{i}",
                    title=f"Alternative based on {expertise}",
                    description=f"Approach emphasizing {expertise} considerations",
                    key_elements=[f"Element 1", f"Element 2"],
                    addresses_concerns=["concern_1", "concern_2"],
                    trade_offs=["trade_off_1", "trade_off_2"],
                    proposed_by=self.agent_id
                )
                alternatives.append(alternative)
        
        return alternatives
    
    async def evaluate_proposal(self, proposal: Proposal, session: ConsensusSession) -> Dict[str, Any]:
        """Evaluate a specific proposal"""
        
        await asyncio.sleep(0.05)
        
        # Evaluate based on own position and concerns
        my_position = session.positions.get(self.agent_id)
        
        if my_position:
            # Check alignment with position
            alignment_score = 0.7  # Simplified scoring
            
            # Check if concerns are addressed
            concerns_addressed = len([c for c in my_position.concerns 
                                    if c in proposal.addresses_concerns])
            concern_score = concerns_addressed / len(my_position.concerns) if my_position.concerns else 1.0
            
            overall_score = (alignment_score + concern_score) / 2
            
            if overall_score >= 0.8:
                agreement = AgreementLevel.STRONG_SUPPORT
            elif overall_score >= 0.6:
                agreement = AgreementLevel.SUPPORT
            elif overall_score >= 0.4:
                agreement = AgreementLevel.NEUTRAL
            elif overall_score >= 0.2:
                agreement = AgreementLevel.CONCERNS
            else:
                agreement = AgreementLevel.STRONG_OPPOSITION
        else:
            agreement = AgreementLevel.NEUTRAL
        
        return {
            "agreement_level": agreement,
            "strengths": ["strength_1", "strength_2"],
            "weaknesses": ["weakness_1", "weakness_2"],
            "suggested_improvements": ["improvement_1", "improvement_2"]
        }
    
    async def participate_in_negotiation(self, session: ConsensusSession, 
                                       consensus_system: 'ConsensusSystem') -> Dict[str, Any]:
        """Participate in negotiation to resolve differences"""
        
        print(f"  {self.agent_id} participating in negotiation")
        
        # Identify areas where compromise is possible
        compromise_areas = await self.identify_compromise_opportunities(session)
        
        # Make concessions where appropriate
        concessions = await self.consider_concessions(session, compromise_areas)
        
        return {
            "compromise_areas_identified": compromise_areas,
            "concessions_offered": concessions,
            "negotiation_style": self.communication_style,
            "remaining_requirements": ["requirement_1", "requirement_2"]
        }
    
    async def identify_compromise_opportunities(self, session: ConsensusSession) -> List[str]:
        """Identify where compromise might be possible"""
        
        await asyncio.sleep(0.1)
        
        my_position = session.positions.get(self.agent_id)
        if not my_position:
            return []
        
        # Areas where flexibility exists
        flexible_areas = []
        
        if my_position.flexibility > 0.7:
            flexible_areas.extend(["implementation_timeline", "resource_allocation"])
        
        if my_position.flexibility > 0.5:
            flexible_areas.extend(["scope_adjustments", "approach_modifications"])
        
        return flexible_areas
    
    async def consider_concessions(self, session: ConsensusSession, 
                                 compromise_areas: List[str]) -> List[str]:
        """Consider what concessions to offer"""
        
        await asyncio.sleep(0.1)
        
        concessions = []
        
        # Willing to make concessions based on compromise willingness
        for area in compromise_areas:
            if self.compromise_willingness > 0.6:
                concessions.append(f"Flexible on {area}")
        
        return concessions
    
    async def validate_proposals(self, session: ConsensusSession, 
                               consensus_system: 'ConsensusSystem') -> Dict[str, Any]:
        """Validate final proposals before decision"""
        
        print(f"  {self.agent_id} validating proposals")
        
        validations = {}
        
        for proposal in session.proposals:
            validation = await self.validate_proposal_feasibility(proposal, session)
            validations[proposal.id] = validation
        
        return {
            "proposal_validations": validations,
            "feasibility_assessment": "detailed_review_completed",
            "implementation_concerns": ["concern_1", "concern_2"]
        }
    
    async def validate_proposal_feasibility(self, proposal: Proposal, 
                                          session: ConsensusSession) -> Dict[str, Any]:
        """Validate feasibility of a specific proposal"""
        
        await asyncio.sleep(0.05)
        
        # Check against expertise
        feasibility_score = 0.8  # Simplified
        
        return {
            "feasibility_score": feasibility_score,
            "implementation_risks": ["risk_1", "risk_2"],
            "resource_requirements": ["resource_1", "resource_2"],
            "success_probability": 0.75
        }
    
    async def make_final_decision(self, session: ConsensusSession, 
                                consensus_system: 'ConsensusSystem') -> Dict[str, Any]:
        """Make final decision on proposals"""
        
        print(f"  {self.agent_id} making final decision")
        
        # Rank proposals
        proposal_rankings = await self.rank_proposals(session.proposals, session)
        
        return {
            "proposal_rankings": proposal_rankings,
            "final_preference": proposal_rankings[0] if proposal_rankings else None,
            "decision_confidence": 0.8
        }
    
    async def rank_proposals(self, proposals: List[Proposal], 
                           session: ConsensusSession) -> List[str]:
        """Rank proposals in order of preference"""
        
        await asyncio.sleep(0.1)
        
        # Simple ranking based on previous evaluations
        rankings = []
        
        for proposal in proposals:
            # Add to rankings (simplified logic)
            rankings.append(proposal.id)
        
        return rankings
    
    async def commit_to_decision(self, session: ConsensusSession, 
                               consensus_system: 'ConsensusSystem') -> Dict[str, Any]:
        """Commit to supporting the final decision"""
        
        print(f"  {self.agent_id} committing to decision")
        
        if session.final_decision:
            # Assess satisfaction with decision
            satisfaction = await self.assess_decision_satisfaction(session.final_decision, session)
            
            # Commit to implementation
            commitment_level = min(1.0, satisfaction + 0.2)  # Slight boost for group decision
            
            self.satisfaction_with_outcomes.append(satisfaction)
            
            return {
                "decision_accepted": True,
                "satisfaction_level": satisfaction,
                "commitment_level": commitment_level,
                "implementation_support": "Will actively support implementation"
            }
        else:
            return {
                "decision_accepted": False,
                "satisfaction_level": 0.3,
                "commitment_level": 0.5,
                "implementation_support": "Limited support due to lack of consensus"
            }
    
    async def assess_decision_satisfaction(self, decision: Proposal, 
                                         session: ConsensusSession) -> float:
        """Assess satisfaction with the final decision"""
        
        my_position = session.positions.get(self.agent_id)
        if not my_position:
            return 0.5
        
        # Simplified satisfaction calculation
        if my_position.agreement_level == AgreementLevel.STRONG_SUPPORT:
            return 0.9
        elif my_position.agreement_level == AgreementLevel.SUPPORT:
            return 0.8
        elif my_position.agreement_level == AgreementLevel.NEUTRAL:
            return 0.6
        elif my_position.agreement_level == AgreementLevel.CONCERNS:
            return 0.4
        else:
            return 0.2
    
    def get_consensus_summary(self) -> Dict[str, Any]:
        """Get summary of consensus participation"""
        
        avg_satisfaction = (sum(self.satisfaction_with_outcomes) / len(self.satisfaction_with_outcomes) 
                          if self.satisfaction_with_outcomes else 0.5)
        
        success_rate = (self.successful_consensus_count / self.consensus_participation_count 
                       if self.consensus_participation_count > 0 else 0.0)
        
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "expertise_areas": self.expertise_areas,
            "consensus_participations": self.consensus_participation_count,
            "successful_consensuses": self.successful_consensus_count,
            "success_rate": success_rate,
            "average_satisfaction": avg_satisfaction,
            "compromise_willingness": self.compromise_willingness,
            "communication_style": self.communication_style
        }

class ConsensusSystem:
    """
    System for managing consensus building processes
    
    EXAMPLE USAGE:
    =============
    # Create consensus system
    system = ConsensusSystem("decision_making")
    
    # Add participants with different roles
    facilitator = ConsensusAgent("facilitator", ParticipantRole.FACILITATOR, ["process"])
    expert = ConsensusAgent("expert", ParticipantRole.EXPERT, ["technology"])
    stakeholder = ConsensusAgent("stakeholder", ParticipantRole.STAKEHOLDER, ["business"])
    
    system.add_participant(facilitator)
    system.add_participant(expert) 
    system.add_participant(stakeholder)
    
    # Build consensus on a decision
    result = await system.build_consensus("Should we implement new AI system?")
    """
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.participants: Dict[str, ConsensusAgent] = {}
        self.active_sessions: Dict[str, ConsensusSession] = {}
        self.completed_sessions: List[ConsensusSession] = []
        
        # Consensus parameters
        self.default_consensus_threshold = 0.8
        self.max_session_duration = 3600  # 1 hour
        self.stage_timeouts = {
            ConsensusStage.FRAMING: 300,
            ConsensusStage.INFORMATION_GATHERING: 600,
            ConsensusStage.POSITION_SHARING: 400,
            ConsensusStage.EXPLORATION: 800,
            ConsensusStage.NEGOTIATION: 1200,
            ConsensusStage.VALIDATION: 300,
            ConsensusStage.DECISION: 400
        }
    
    def add_participant(self, agent: ConsensusAgent) -> None:
        """Add participant to the consensus system"""
        self.participants[agent.agent_id] = agent
        print(f"Added consensus participant: {agent.agent_id} ({agent.role.value})")
    
    async def build_consensus(self, topic: str, 
                            participants: List[str] = None,
                            facilitator_id: str = None) -> Dict[str, Any]:
        """
        Build consensus on a specific topic
        
        Args:
            topic: The decision topic
            participants: List of participant IDs (default: all participants)
            facilitator_id: ID of facilitator (default: find facilitator role)
            
        Returns:
            Consensus building results
        """
        
        print(f"\nBUILDING CONSENSUS: {topic}")
        print("=" * 60)
        
        # Set up session participants
        if participants is None:
            participants = list(self.participants.keys())
        
        # Find facilitator
        if facilitator_id is None:
            facilitator_id = self.find_facilitator(participants)
        
        if not facilitator_id:
            print("Warning: No facilitator found, using first participant")
            facilitator_id = participants[0]
        
        # Create consensus session
        session = ConsensusSession(
            id=f"consensus_{uuid.uuid4().hex[:8]}",
            topic=topic,
            facilitator_id=facilitator_id,
            participants=participants,
            stage=ConsensusStage.FRAMING,
            positions={},
            proposals=[]
        )
        
        self.active_sessions[session.id] = session
        
        # Execute consensus process
        consensus_result = await self.execute_consensus_process(session)
        
        # Move to completed sessions
        session.end_time = time.time()
        self.completed_sessions.append(session)
        del self.active_sessions[session.id]
        
        return consensus_result
    
    def find_facilitator(self, participants: List[str]) -> Optional[str]:
        """Find participant with facilitator role"""
        for participant_id in participants:
            participant = self.participants.get(participant_id)
            if participant and participant.role == ParticipantRole.FACILITATOR:
                return participant_id
        return None
    
    async def execute_consensus_process(self, session: ConsensusSession) -> Dict[str, Any]:
        """Execute the full consensus building process"""
        
        stage_results = {}
        
        # Execute each stage
        consensus_stages = [
            ConsensusStage.FRAMING,
            ConsensusStage.INFORMATION_GATHERING,
            ConsensusStage.POSITION_SHARING,
            ConsensusStage.EXPLORATION,
            ConsensusStage.NEGOTIATION,
            ConsensusStage.VALIDATION,
            ConsensusStage.DECISION
        ]
        
        for stage in consensus_stages:
            session.stage = stage
            print(f"\nStage: {stage.value.upper()}")
            print("-" * 30)
            
            stage_result = await self.execute_stage(session)
            stage_results[stage.value] = stage_result
            
            # Check if consensus achieved early
            if stage == ConsensusStage.EXPLORATION:
                early_consensus = self.check_early_consensus(session)
                if early_consensus:
                    print("Early consensus achieved!")
                    break
        
        # Final consensus evaluation
        final_consensus = await self.finalize_consensus(session)
        
        total_time = time.time() - session.start_time
        
        return {
            "session_id": session.id,
            "topic": session.topic,
            "total_time": total_time,
            "stages_completed": list(stage_results.keys()),
            "stage_results": stage_results,
            "final_consensus": final_consensus,
            "consensus_achieved": session.final_decision is not None,
            "participant_satisfaction": self.calculate_overall_satisfaction(session)
        }
    
    async def execute_stage(self, session: ConsensusSession) -> Dict[str, Any]:
        """Execute a specific consensus stage"""
        
        stage_start = time.time()
        participant_results = {}
        
        # Have all participants engage in the current stage
        tasks = []
        for participant_id in session.participants:
            participant = self.participants.get(participant_id)
            if participant:
                if session.stage == ConsensusStage.FRAMING:
                    task = participant.participate_in_framing(session, self)
                elif session.stage == ConsensusStage.INFORMATION_GATHERING:
                    task = participant.gather_information(session, self)
                elif session.stage == ConsensusStage.POSITION_SHARING:
                    task = participant.share_position(session, self)
                elif session.stage == ConsensusStage.EXPLORATION:
                    task = participant.explore_alternatives(session, self)
                elif session.stage == ConsensusStage.NEGOTIATION:
                    task = participant.participate_in_negotiation(session, self)
                elif session.stage == ConsensusStage.VALIDATION:
                    task = participant.validate_proposals(session, self)
                elif session.stage == ConsensusStage.DECISION:
                    task = participant.make_final_decision(session, self)
                
                tasks.append((participant_id, task))
        
        # Execute all participant tasks
        for participant_id, task in tasks:
            try:
                result = await task
                participant_results[participant_id] = result
            except Exception as e:
                print(f"Error with participant {participant_id}: {e}")
                participant_results[participant_id] = {"error": str(e)}
        
        stage_time = time.time() - stage_start
        
        # Stage-specific processing
        if session.stage == ConsensusStage.EXPLORATION:
            await self.process_alternatives(session, participant_results)
        elif session.stage == ConsensusStage.DECISION:
            await self.select_final_decision(session, participant_results)
        
        return {
            "stage": session.stage.value,
            "execution_time": stage_time,
            "participant_results": participant_results,
            "stage_summary": self.summarize_stage(session, participant_results)
        }
    
    async def process_alternatives(self, session: ConsensusSession, 
                                 participant_results: Dict[str, Any]) -> None:
        """Process alternative proposals from exploration stage"""
        
        # Collect all alternatives from participants
        for participant_id, result in participant_results.items():
            if "alternatives_proposed" in result:
                # Add new proposals to session (simplified)
                for i in range(result["alternatives_proposed"]):
                    proposal = Proposal(
                        id=f"proposal_{participant_id}_{i}",
                        title=f"Proposal from {participant_id}",
                        description=f"Alternative proposal based on {participant_id}'s perspective",
                        key_elements=["element_1", "element_2"],
                        addresses_concerns=["concern_1"],
                        trade_offs=["trade_off_1"],
                        proposed_by=participant_id
                    )
                    session.proposals.append(proposal)
        
        print(f"  Total proposals generated: {len(session.proposals)}")
    
    async def select_final_decision(self, session: ConsensusSession, 
                                  participant_results: Dict[str, Any]) -> None:
        """Select final decision based on participant rankings"""
        
        if not session.proposals:
            print("  No proposals to decide on")
            return
        
        # Collect rankings from all participants
        all_rankings = {}
        for participant_id, result in participant_results.items():
            if "proposal_rankings" in result:
                all_rankings[participant_id] = result["proposal_rankings"]
        
        # Simple voting: count preferences
        proposal_scores = {}
        for proposal in session.proposals:
            proposal_scores[proposal.id] = 0
        
        for participant_id, rankings in all_rankings.items():
            for i, proposal_id in enumerate(rankings):
                if proposal_id in proposal_scores:
                    # Higher score for higher ranking (reverse order)
                    proposal_scores[proposal_id] += len(rankings) - i
        
        # Select highest scoring proposal
        if proposal_scores:
            best_proposal_id = max(proposal_scores.items(), key=lambda x: x[1])[0]
            session.final_decision = next(p for p in session.proposals if p.id == best_proposal_id)
            print(f"  Selected proposal: {best_proposal_id}")
        else:
            print("  Unable to select final decision")
    
    def check_early_consensus(self, session: ConsensusSession) -> bool:
        """Check if consensus has been achieved early"""
        
        if not session.positions:
            return False
        
        # Check agreement levels
        support_count = 0
        total_count = len(session.positions)
        
        for position in session.positions.values():
            if position.agreement_level.value >= AgreementLevel.SUPPORT.value:
                support_count += 1
        
        consensus_ratio = support_count / total_count
        return consensus_ratio >= session.consensus_threshold
    
    async def finalize_consensus(self, session: ConsensusSession) -> Dict[str, Any]:
        """Finalize the consensus process"""
        
        if session.final_decision:
            # Have all participants commit
            session.stage = ConsensusStage.COMMITMENT
            
            commitment_results = {}
            for participant_id in session.participants:
                participant = self.participants.get(participant_id)
                if participant:
                    commitment = await participant.commit_to_decision(session, self)
                    commitment_results[participant_id] = commitment
            
            return {
                "decision_reached": True,
                "final_decision": session.final_decision.title,
                "participant_commitments": commitment_results,
                "consensus_quality": self.assess_consensus_quality(session)
            }
        else:
            return {
                "decision_reached": False,
                "reason": "Unable to reach consensus within process",
                "partial_agreements": self.identify_partial_agreements(session)
            }
    
    def assess_consensus_quality(self, session: ConsensusSession) -> Dict[str, Any]:
        """Assess the quality of the consensus reached"""
        
        if not session.positions:
            return {"quality": "unknown"}
        
        # Calculate agreement distribution
        agreement_levels = [pos.agreement_level.value for pos in session.positions.values()]
        avg_agreement = statistics.mean(agreement_levels)
        agreement_std = statistics.stdev(agreement_levels) if len(agreement_levels) > 1 else 0
        
        return {
            "average_agreement": avg_agreement,
            "agreement_consistency": 1.0 - (agreement_std / 2.0),  # Normalized
            "process_completeness": 1.0,  # Completed all stages
            "participation_rate": len(session.positions) / len(session.participants)
        }
    
    def identify_partial_agreements(self, session: ConsensusSession) -> List[str]:
        """Identify areas of partial agreement"""
        
        partial_agreements = []
        
        # Look for common elements in positions
        if session.positions:
            common_concerns = []
            all_concerns = [concern for pos in session.positions.values() for concern in pos.concerns]
            
            for concern in set(all_concerns):
                if all_concerns.count(concern) >= len(session.positions) * 0.5:
                    common_concerns.append(concern)
            
            partial_agreements.extend(common_concerns)
        
        return partial_agreements
    
    def summarize_stage(self, session: ConsensusSession, 
                       participant_results: Dict[str, Any]) -> str:
        """Summarize what happened in a stage"""
        
        if session.stage == ConsensusStage.POSITION_SHARING:
            return f"{len(session.positions)} positions shared"
        elif session.stage == ConsensusStage.EXPLORATION:
            return f"{len(session.proposals)} proposals generated"
        elif session.stage == ConsensusStage.DECISION:
            return f"Final decision: {'reached' if session.final_decision else 'not reached'}"
        else:
            return f"Stage {session.stage.value} completed"
    
    def calculate_overall_satisfaction(self, session: ConsensusSession) -> float:
        """Calculate overall participant satisfaction"""
        
        satisfactions = []
        for participant_id in session.participants:
            participant = self.participants.get(participant_id)
            if participant and participant.satisfaction_with_outcomes:
                satisfactions.append(participant.satisfaction_with_outcomes[-1])
        
        return statistics.mean(satisfactions) if satisfactions else 0.5
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        
        total_sessions = len(self.completed_sessions)
        successful_sessions = len([s for s in self.completed_sessions if s.final_decision])
        
        return {
            "system_id": self.system_id,
            "total_participants": len(self.participants),
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "success_rate": successful_sessions / total_sessions if total_sessions > 0 else 0,
            "participant_summaries": {p.agent_id: p.get_consensus_summary() 
                                    for p in self.participants.values()}
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_corporate_decision_making():
    """Demo: Corporate team making strategic decision through consensus"""
    print("\nDEMO 1: CORPORATE STRATEGIC DECISION MAKING")
    print("=" * 60)
    
    # Create consensus system
    system = ConsensusSystem("corporate_decisions")
    
    # Add corporate decision makers
    facilitator = ConsensusAgent("hr_director", ParticipantRole.FACILITATOR, ["process", "human_resources"])
    cto = ConsensusAgent("cto", ParticipantRole.EXPERT, ["technology", "architecture"])
    cfo = ConsensusAgent("cfo", ParticipantRole.STAKEHOLDER, ["finance", "budget"])
    product_manager = ConsensusAgent("product_manager", ParticipantRole.STAKEHOLDER, ["product", "customers"])
    
    system.add_participant(facilitator)
    system.add_participant(cto)
    system.add_participant(cfo)
    system.add_participant(product_manager)
    
    # Build consensus on strategic decision
    result = await system.build_consensus("Should we invest $2M in AI infrastructure upgrade?")
    
    print(f"\nCorporate Decision Results:")
    print(f"- Consensus achieved: {result['consensus_achieved']}")
    print(f"- Decision time: {result['total_time']:.1f} seconds")
    print(f"- Participant satisfaction: {result['participant_satisfaction']:.2f}")
    
    if result['final_consensus']['decision_reached']:
        print(f"- Final decision: {result['final_consensus']['final_decision']}")

async def demo_technical_committee():
    """Demo: Technical committee consensus on standards"""
    print("\nDEMO 2: TECHNICAL COMMITTEE STANDARDS DECISION")
    print("=" * 60)
    
    system = ConsensusSystem("technical_standards")
    
    # Create technical committee
    chair = ConsensusAgent("committee_chair", ParticipantRole.FACILITATOR, ["standards", "governance"])
    security_expert = ConsensusAgent("security_expert", ParticipantRole.EXPERT, ["security", "compliance"])
    architect = ConsensusAgent("architect", ParticipantRole.EXPERT, ["architecture", "design"])
    developer_rep = ConsensusAgent("developer_rep", ParticipantRole.STAKEHOLDER, ["development", "implementation"])
    ops_rep = ConsensusAgent("ops_rep", ParticipantRole.STAKEHOLDER, ["operations", "maintenance"])
    
    # Add different compromise willingness
    security_expert.compromise_willingness = 0.4  # Less willing to compromise on security
    developer_rep.compromise_willingness = 0.8    # More flexible
    
    for agent in [chair, security_expert, architect, developer_rep, ops_rep]:
        system.add_participant(agent)
    
    # Consensus on technical standard
    result = await system.build_consensus("Should we adopt new microservices security framework?")
    
    print(f"\nTechnical Committee Results:")
    print(f"- Stages completed: {len(result['stages_completed'])}")
    print(f"- Final agreement: {result['consensus_achieved']}")
    
    # Show individual participant results
    for participant_id, participant in system.participants.items():
        summary = participant.get_consensus_summary()
        print(f"- {participant_id}: {summary['compromise_willingness']:.1f} compromise willingness")

async def demo_community_planning():
    """Demo: Community planning consensus with diverse stakeholders"""
    print("\nDEMO 3: COMMUNITY PLANNING CONSENSUS")
    print("=" * 60)
    
    system = ConsensusSystem("community_planning")
    
    # Create diverse community stakeholders
    moderator = ConsensusAgent("city_planner", ParticipantRole.FACILITATOR, ["urban_planning", "process"])
    environmental = ConsensusAgent("environmental_group", ParticipantRole.STAKEHOLDER, ["environment", "sustainability"])
    business = ConsensusAgent("business_association", ParticipantRole.STAKEHOLDER, ["business", "economic"])
    residents = ConsensusAgent("residents_council", ParticipantRole.STAKEHOLDER, ["community", "quality_of_life"])
    transport = ConsensusAgent("transport_expert", ParticipantRole.EXPERT, ["transportation", "infrastructure"])
    
    # Different communication styles
    environmental.communication_style = "analytical"
    business.communication_style = "assertive"
    residents.communication_style = "collaborative"
    
    for agent in [moderator, environmental, business, residents, transport]:
        system.add_participant(agent)
    
    # Community decision
    result = await system.build_consensus("Should we build new shopping complex in downtown area?")
    
    print(f"\nCommunity Planning Results:")
    print(f"- Process duration: {result['total_time']:.1f} seconds")
    print(f"- Consensus quality: {result['final_consensus'].get('consensus_quality', {})}")
    
    if not result['consensus_achieved']:
        partial = result['final_consensus'].get('partial_agreements', [])
        print(f"- Partial agreements: {len(partial)} areas")

async def main():
    """
    Demonstrate Consensus Building Agents for collaborative decision making
    
    WHAT YOU'LL LEARN:
    ================
    1. How to structure consensus processes for effective group decisions
    2. How to balance different roles and perspectives in decision making
    3. How to handle disagreements and build compromise solutions
    4. How to validate decisions and ensure commitment from participants
    5. How consensus building improves decision quality and buy-in
    
    REAL WORLD APPLICATIONS:
    =======================
    - Corporate strategic planning and board decisions
    - Technical standards committees and architecture reviews
    - Government policy making and legislative processes
    - Community planning and public participation
    - International negotiations and treaty development
    - Organizational change management and transformation
    """
    
    print("CONSENSUS BUILDING AGENTS DEMONSTRATION")
    print("This shows how to facilitate collaborative decision making with diverse stakeholders!")
    
    await demo_corporate_decision_making()
    await demo_technical_committee()
    await demo_community_planning()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Structured consensus processes ensure all voices are heard")
    print("✓ Different roles contribute different perspectives and expertise")
    print("✓ Compromise and negotiation help resolve conflicts constructively")
    print("✓ Validation stages improve decision quality and feasibility")
    print("✓ Consensus decisions have higher implementation success rates")
    print("\nTRY IT YOURSELF:")
    print("- Add domain-specific consensus protocols and decision criteria")
    print("- Implement weighted voting and expertise-based influence")
    print("- Create consensus analytics and decision quality metrics")
    print("- Build real-time consensus tracking and progress visualization")

if __name__ == "__main__":
    asyncio.run(main())
