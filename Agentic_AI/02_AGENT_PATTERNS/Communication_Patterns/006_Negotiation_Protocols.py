#!/usr/bin/env python3
"""
Negotiation Protocols: Strategic Agent Bargaining
=================================================

WHAT IS THE PROBLEM?
==================
Agents with conflicting interests need to reach agreements:
- Resource allocation conflicts between competing agents
- Price negotiations in multi-agent marketplaces
- Task assignment when agents have preferences
- Coalition formation with different utility functions
- Deadline conflicts and scheduling disputes

Example: Meeting Room Booking Chaos
FIRST-COME-FIRST-SERVED (Inefficient):
- Marketing team books conference room for 2-4 PM
- Sales team urgently needs same room for client meeting
- Engineering team also requested room for sprint planning
- No mechanism to resolve conflicts fairly
- Resources allocated inefficiently based on timing not value

REAL WORLD EXAMPLE:
=================
How do airlines handle overbooking negotiations?

AIRLINE SEAT AUCTION SYSTEM:
When flight is overbooked:
1. INITIAL OFFER: Airline offers $200 voucher for volunteers
2. EVALUATION: Passengers assess their flexibility and needs
3. BIDDING: If insufficient volunteers, airline increases offer
4. COUNTER-OFFERS: Passengers can request specific compensation
5. NEGOTIATION: Airline and passengers reach mutually acceptable deal
6. AGREEMENT: Final compensation agreed, volunteers identified

BENEFITS:
- Fair compensation based on actual passenger value
- Efficient allocation of scarce airline seats
- Voluntary participation - no forced bumping
- Market-driven pricing that reflects true costs
- Win-win outcomes for both airline and passengers

THE ALGORITHM:
=============
1. INITIALIZATION: Define negotiation parameters and constraints
2. PROPOSALS: Agents make initial offers based on their preferences
3. EVALUATION: Each agent evaluates proposals against their utility function
4. COUNTER-OFFERS: Agents respond with counter-proposals
5. CONCESSIONS: Agents gradually adjust their demands over time
6. AGREEMENT: Check if proposals are mutually acceptable
7. COMMITMENT: Finalize agreement and enforce terms

NEGOTIATION STRATEGIES:
- Competitive: Maximize own utility, minimal concessions
- Cooperative: Seek mutually beneficial outcomes
- Tit-for-Tat: Mirror opponent's cooperation/competition
- Time-dependent: Increase concessions as deadline approaches
- Behavior-adaptive: Learn from opponent's patterns

WHY IS THIS POWERFUL?
===================
- Enables fair resource allocation in multi-agent systems
- Supports automated contract negotiation and deal-making
- Allows agents to resolve conflicts without human intervention
- Creates market-efficient outcomes through competitive bidding
- Enables complex coalition formation and partnership agreements
- Provides foundation for decentralized autonomous organizations
"""

import asyncio
import time
import json
import uuid
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Callable, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class NegotiationType(Enum):
    """Types of negotiation protocols"""
    BILATERAL = "bilateral"                 # Two-party negotiation
    MULTILATERAL = "multilateral"           # Multiple parties
    AUCTION = "auction"                     # Auction-based allocation
    COALITION = "coalition"                 # Coalition formation
    BARGAINING = "bargaining"               # Sequential bargaining
    TOURNAMENT = "tournament"               # Competitive elimination

class NegotiationStrategy(Enum):
    """Negotiation strategies"""
    COMPETITIVE = "competitive"             # Maximize own utility
    COOPERATIVE = "cooperative"             # Seek mutual benefit
    TIT_FOR_TAT = "tit_for_tat"            # Mirror opponent behavior
    TIME_DEPENDENT = "time_dependent"       # Deadline-driven concessions
    BEHAVIOR_ADAPTIVE = "behavior_adaptive" # Learn from opponent patterns
    RANDOM = "random"                       # Random concessions

class ProposalType(Enum):
    """Types of proposals"""
    INITIAL_OFFER = "initial_offer"
    COUNTER_OFFER = "counter_offer"
    FINAL_OFFER = "final_offer"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"
    WITHDRAWAL = "withdrawal"

class NegotiationStatus(Enum):
    """Status of negotiations"""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    AGREEMENT_REACHED = "agreement_reached"
    FAILED = "failed"
    TIMEOUT = "timeout"
    WITHDRAWN = "withdrawn"

@dataclass
class NegotiationIssue:
    """Single issue being negotiated"""
    name: str
    value_type: str                         # "numeric", "categorical", "boolean"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    possible_values: Optional[List[Any]] = None
    weight: float = 1.0                     # Importance to agent
    
    def is_valid_value(self, value: Any) -> bool:
        """Check if value is valid for this issue"""
        if self.value_type == "numeric":
            return (self.min_value is None or value >= self.min_value) and \
                   (self.max_value is None or value <= self.max_value)
        elif self.value_type == "categorical":
            return self.possible_values is None or value in self.possible_values
        elif self.value_type == "boolean":
            return isinstance(value, bool)
        return True

@dataclass
class Proposal:
    """Negotiation proposal"""
    id: str
    negotiation_id: str
    proposer_id: str
    proposal_type: ProposalType
    
    # Proposal content
    issues: Dict[str, Any] = field(default_factory=dict)  # issue_name -> proposed_value
    conditions: List[str] = field(default_factory=list)   # Additional conditions
    
    # Proposal metadata
    timestamp: float = field(default_factory=time.time)
    round_number: int = 0
    expires_at: Optional[float] = None
    
    # Evaluation
    utility_estimate: Optional[float] = None
    confidence: float = 1.0
    
    def __post_init__(self):
        """Initialize proposal"""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def is_expired(self) -> bool:
        """Check if proposal has expired"""
        return self.expires_at is not None and time.time() > self.expires_at
    
    def serialize(self) -> str:
        """Serialize proposal to JSON"""
        data = asdict(self)
        data['proposal_type'] = self.proposal_type.value
        return json.dumps(data)
    
    @classmethod
    def deserialize(cls, data: str) -> 'Proposal':
        """Deserialize proposal from JSON"""
        obj = json.loads(data)
        obj['proposal_type'] = ProposalType(obj['proposal_type'])
        return cls(**obj)

class UtilityFunction:
    """Utility function for evaluating proposals"""
    
    def __init__(self, issues: List[NegotiationIssue]):
        self.issues = {issue.name: issue for issue in issues}
        self.reservation_value = 0.0  # BATNA (Best Alternative to Negotiated Agreement)
    
    def evaluate(self, proposal: Dict[str, Any]) -> float:
        """Evaluate utility of proposal"""
        total_utility = 0.0
        total_weight = 0.0
        
        for issue_name, issue in self.issues.items():
            if issue_name in proposal:
                value = proposal[issue_name]
                issue_utility = self._evaluate_issue_utility(issue, value)
                weighted_utility = issue_utility * issue.weight
                total_utility += weighted_utility
                total_weight += issue.weight
        
        # Normalize by total weight
        if total_weight > 0:
            return total_utility / total_weight
        return 0.0
    
    def _evaluate_issue_utility(self, issue: NegotiationIssue, value: Any) -> float:
        """Evaluate utility for single issue"""
        if issue.value_type == "numeric":
            if issue.min_value is not None and issue.max_value is not None:
                # Normalize to 0-1 range
                normalized = (value - issue.min_value) / (issue.max_value - issue.min_value)
                return max(0.0, min(1.0, normalized))
            else:
                return float(value)
        
        elif issue.value_type == "categorical":
            if issue.possible_values:
                # Simple binary utility: 1.0 if preferred value, 0.0 otherwise
                # In practice, you'd have more sophisticated categorical utilities
                return 1.0 if value == issue.possible_values[0] else 0.0
            return 0.5
        
        elif issue.value_type == "boolean":
            return 1.0 if value else 0.0
        
        return 0.0
    
    def is_acceptable(self, proposal: Dict[str, Any]) -> bool:
        """Check if proposal is acceptable (above reservation value)"""
        return self.evaluate(proposal) >= self.reservation_value

class NegotiationAgent(ABC):
    """Abstract base class for negotiating agents"""
    
    def __init__(self, agent_id: str, strategy: NegotiationStrategy = NegotiationStrategy.COOPERATIVE):
        self.agent_id = agent_id
        self.strategy = strategy
        self.utility_function: Optional[UtilityFunction] = None
        self.negotiation_history: List[Dict[str, Any]] = []
        
        # Strategy parameters
        self.concession_rate = 0.1      # How quickly to make concessions
        self.patience = 0.8             # Willingness to wait (0-1)
        self.risk_tolerance = 0.5       # Risk appetite (0-1)
        
        # Learning parameters
        self.opponent_model: Dict[str, Any] = {}
        self.past_negotiations: List[Dict[str, Any]] = []
    
    @abstractmethod
    async def generate_initial_proposal(self, negotiation: 'Negotiation') -> Proposal:
        """Generate initial proposal for negotiation"""
        pass
    
    @abstractmethod
    async def respond_to_proposal(self, proposal: Proposal, negotiation: 'Negotiation') -> Proposal:
        """Respond to opponent's proposal"""
        pass
    
    @abstractmethod
    async def evaluate_final_offer(self, proposal: Proposal) -> bool:
        """Decide whether to accept final offer"""
        pass
    
    def set_utility_function(self, utility_function: UtilityFunction) -> None:
        """Set utility function for this agent"""
        self.utility_function = utility_function
    
    def update_opponent_model(self, proposal: Proposal) -> None:
        """Update model of opponent's preferences"""
        if proposal.proposer_id != self.agent_id:
            # Simple opponent modeling: track their proposals
            if proposal.proposer_id not in self.opponent_model:
                self.opponent_model[proposal.proposer_id] = {
                    'proposals': [],
                    'concession_pattern': [],
                    'estimated_reservation_value': 0.0
                }
            
            self.opponent_model[proposal.proposer_id]['proposals'].append(proposal)
    
    def calculate_concession(self, round_number: int, max_rounds: int) -> float:
        """Calculate concession factor based on strategy and round"""
        if self.strategy == NegotiationStrategy.TIME_DEPENDENT:
            # Increase concessions as deadline approaches
            time_pressure = round_number / max_rounds
            return self.concession_rate * (1 + time_pressure)
        
        elif self.strategy == NegotiationStrategy.COMPETITIVE:
            # Minimal concessions
            return self.concession_rate * 0.5
        
        elif self.strategy == NegotiationStrategy.COOPERATIVE:
            # Moderate concessions to reach agreement
            return self.concession_rate
        
        elif self.strategy == NegotiationStrategy.TIT_FOR_TAT:
            # Mirror opponent's concession pattern
            # Simplified: use base concession rate
            return self.concession_rate
        
        else:
            return self.concession_rate

class SimpleNegotiationAgent(NegotiationAgent):
    """Simple negotiation agent implementation"""
    
    def __init__(self, agent_id: str, strategy: NegotiationStrategy = NegotiationStrategy.COOPERATIVE):
        super().__init__(agent_id, strategy)
        self.target_utility = 0.8       # Desired utility level
        self.minimum_utility = 0.3      # Minimum acceptable utility
    
    async def generate_initial_proposal(self, negotiation: 'Negotiation') -> Proposal:
        """Generate initial proposal"""
        if not self.utility_function:
            raise ValueError("Utility function not set")
        
        # Start with proposal that gives us high utility
        proposal_values = {}
        
        for issue_name, issue in self.utility_function.issues.items():
            if issue.value_type == "numeric":
                if issue.min_value is not None and issue.max_value is not None:
                    # Start with value that maximizes our utility
                    if self.strategy == NegotiationStrategy.COMPETITIVE:
                        # Ask for maximum value
                        proposal_values[issue_name] = issue.max_value
                    else:
                        # Start at 80% of range
                        range_size = issue.max_value - issue.min_value
                        proposal_values[issue_name] = issue.min_value + (range_size * 0.8)
                else:
                    proposal_values[issue_name] = 100.0  # Default high value
            
            elif issue.value_type == "categorical":
                if issue.possible_values:
                    # Choose first (preferred) value
                    proposal_values[issue_name] = issue.possible_values[0]
                else:
                    proposal_values[issue_name] = "preferred_option"
            
            elif issue.value_type == "boolean":
                proposal_values[issue_name] = True  # Assume True is preferred
        
        proposal = Proposal(
            id="",
            negotiation_id=negotiation.negotiation_id,
            proposer_id=self.agent_id,
            proposal_type=ProposalType.INITIAL_OFFER,
            issues=proposal_values,
            round_number=1,
            expires_at=time.time() + 300  # 5 minutes expiry
        )
        
        # Evaluate our own proposal
        proposal.utility_estimate = self.utility_function.evaluate(proposal_values)
        
        return proposal
    
    async def respond_to_proposal(self, proposal: Proposal, negotiation: 'Negotiation') -> Proposal:
        """Respond to opponent's proposal"""
        if not self.utility_function:
            raise ValueError("Utility function not set")
        
        # Update opponent model
        self.update_opponent_model(proposal)
        
        # Evaluate opponent's proposal
        opponent_utility = self.utility_function.evaluate(proposal.issues)
        
        # Check if we should accept
        if opponent_utility >= self.minimum_utility:
            if (self.strategy == NegotiationStrategy.COOPERATIVE and 
                opponent_utility >= self.minimum_utility * 1.2):
                # Accept good cooperative offers
                return Proposal(
                    id="",
                    negotiation_id=negotiation.negotiation_id,
                    proposer_id=self.agent_id,
                    proposal_type=ProposalType.ACCEPTANCE,
                    issues=proposal.issues,
                    round_number=proposal.round_number + 1
                )
        
        # Generate counter-offer
        counter_offer_values = {}
        concession_factor = self.calculate_concession(
            proposal.round_number, 
            negotiation.max_rounds
        )
        
        for issue_name, issue in self.utility_function.issues.items():
            if issue_name in proposal.issues:
                opponent_value = proposal.issues[issue_name]
                
                if issue.value_type == "numeric" and issue.min_value is not None and issue.max_value is not None:
                    # Calculate compromise position
                    our_ideal = issue.max_value if self.strategy == NegotiationStrategy.COMPETITIVE else issue.min_value + (issue.max_value - issue.min_value) * 0.8
                    
                    # Move toward opponent's position by concession factor
                    compromise = our_ideal + (opponent_value - our_ideal) * concession_factor
                    counter_offer_values[issue_name] = max(issue.min_value, min(issue.max_value, compromise))
                
                elif issue.value_type == "categorical":
                    # For categorical, either accept or propose alternative
                    if random.random() < concession_factor:
                        counter_offer_values[issue_name] = opponent_value
                    else:
                        if issue.possible_values:
                            # Choose different value
                            alternatives = [v for v in issue.possible_values if v != opponent_value]
                            counter_offer_values[issue_name] = random.choice(alternatives) if alternatives else opponent_value
                        else:
                            counter_offer_values[issue_name] = "alternative_option"
                
                elif issue.value_type == "boolean":
                    # For boolean, gradually concede
                    if random.random() < concession_factor:
                        counter_offer_values[issue_name] = opponent_value
                    else:
                        counter_offer_values[issue_name] = not opponent_value
            else:
                # Issue not in opponent's proposal - use our preference
                if issue.value_type == "numeric":
                    counter_offer_values[issue_name] = issue.max_value
                elif issue.value_type == "categorical":
                    counter_offer_values[issue_name] = issue.possible_values[0] if issue.possible_values else "preferred"
                elif issue.value_type == "boolean":
                    counter_offer_values[issue_name] = True
        
        counter_proposal = Proposal(
            id="",
            negotiation_id=negotiation.negotiation_id,
            proposer_id=self.agent_id,
            proposal_type=ProposalType.COUNTER_OFFER,
            issues=counter_offer_values,
            round_number=proposal.round_number + 1,
            expires_at=time.time() + 300
        )
        
        counter_proposal.utility_estimate = self.utility_function.evaluate(counter_offer_values)
        
        return counter_proposal
    
    async def evaluate_final_offer(self, proposal: Proposal) -> bool:
        """Decide whether to accept final offer"""
        if not self.utility_function:
            return False
        
        utility = self.utility_function.evaluate(proposal.issues)
        return utility >= self.minimum_utility

class Negotiation:
    """Represents a negotiation session"""
    
    def __init__(self, negotiation_id: str, negotiation_type: NegotiationType, 
                 issues: List[NegotiationIssue], participants: List[str]):
        self.negotiation_id = negotiation_id
        self.negotiation_type = negotiation_type
        self.issues = {issue.name: issue for issue in issues}
        self.participants = participants
        
        # Negotiation state
        self.status = NegotiationStatus.INITIATED
        self.current_round = 0
        self.max_rounds = 10
        self.started_at = time.time()
        self.timeout = 600.0  # 10 minutes
        
        # Proposal history
        self.proposals: List[Proposal] = []
        self.agreements: List[Proposal] = []
        
        # Final outcome
        self.final_agreement: Optional[Dict[str, Any]] = None
        self.winning_agent: Optional[str] = None
        
        self.logger = logging.getLogger(__name__)
    
    def add_proposal(self, proposal: Proposal) -> None:
        """Add proposal to negotiation history"""
        self.proposals.append(proposal)
        
        if proposal.proposal_type == ProposalType.ACCEPTANCE:
            self.agreements.append(proposal)
    
    def is_expired(self) -> bool:
        """Check if negotiation has expired"""
        return time.time() - self.started_at > self.timeout
    
    def is_complete(self) -> bool:
        """Check if negotiation is complete"""
        return (self.status in [NegotiationStatus.AGREEMENT_REACHED, 
                               NegotiationStatus.FAILED, 
                               NegotiationStatus.TIMEOUT] or
                self.current_round >= self.max_rounds or
                self.is_expired())
    
    def get_latest_proposal(self, agent_id: str = None) -> Optional[Proposal]:
        """Get latest proposal (optionally from specific agent)"""
        filtered_proposals = self.proposals
        if agent_id:
            filtered_proposals = [p for p in self.proposals if p.proposer_id == agent_id]
        
        return filtered_proposals[-1] if filtered_proposals else None
    
    def finalize_agreement(self, agreed_proposal: Proposal) -> None:
        """Finalize negotiation with agreement"""
        self.status = NegotiationStatus.AGREEMENT_REACHED
        self.final_agreement = agreed_proposal.issues
        self.winning_agent = agreed_proposal.proposer_id

class NegotiationProtocol:
    """
    Complete negotiation protocol system
    
    EXAMPLE USAGE:
    =============
    # Create negotiation protocol
    protocol = NegotiationProtocol()
    
    # Define negotiation issues
    issues = [
        NegotiationIssue("price", "numeric", 100, 1000, weight=0.6),
        NegotiationIssue("delivery_time", "numeric", 1, 30, weight=0.4)
    ]
    
    # Create agents
    buyer = SimpleNegotiationAgent("buyer", NegotiationStrategy.COMPETITIVE)
    seller = SimpleNegotiationAgent("seller", NegotiationStrategy.COOPERATIVE)
    
    # Set up utility functions
    buyer_utility = UtilityFunction(issues)  # Buyer wants low price, fast delivery
    seller_utility = UtilityFunction(issues)  # Seller wants high price, flexible delivery
    
    buyer.set_utility_function(buyer_utility)
    seller.set_utility_function(seller_utility)
    
    # Start negotiation
    negotiation = await protocol.start_bilateral_negotiation(
        "price_negotiation", issues, buyer, seller
    )
    """
    
    def __init__(self):
        self.active_negotiations: Dict[str, Negotiation] = {}
        self.completed_negotiations: List[Negotiation] = []
        self.registered_agents: Dict[str, NegotiationAgent] = {}
        
        # Protocol statistics
        self.stats = {
            'negotiations_started': 0,
            'successful_agreements': 0,
            'failed_negotiations': 0,
            'total_rounds': 0,
            'average_negotiation_time': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def register_agent(self, agent: NegotiationAgent) -> None:
        """Register negotiation agent"""
        self.registered_agents[agent.agent_id] = agent
        self.logger.info(f"Agent registered: {agent.agent_id}")
    
    async def start_bilateral_negotiation(self, negotiation_id: str, 
                                        issues: List[NegotiationIssue],
                                        agent1: NegotiationAgent, 
                                        agent2: NegotiationAgent,
                                        max_rounds: int = 10) -> Negotiation:
        """Start bilateral negotiation between two agents"""
        
        # Create negotiation
        negotiation = Negotiation(
            negotiation_id=negotiation_id,
            negotiation_type=NegotiationType.BILATERAL,
            issues=issues,
            participants=[agent1.agent_id, agent2.agent_id]
        )
        negotiation.max_rounds = max_rounds
        
        self.active_negotiations[negotiation_id] = negotiation
        self.stats['negotiations_started'] += 1
        
        # Run negotiation
        result = await self._run_bilateral_negotiation(negotiation, agent1, agent2)
        
        # Move to completed negotiations
        self.completed_negotiations.append(negotiation)
        del self.active_negotiations[negotiation_id]
        
        return result
    
    async def start_auction(self, auction_id: str, item_description: Dict[str, Any],
                           auctioneer: str, bidders: List[NegotiationAgent],
                           auction_type: str = "english") -> Negotiation:
        """Start auction-based negotiation"""
        
        # Create auction issues
        issues = [
            NegotiationIssue("bid_amount", "numeric", 0, 10000, weight=1.0)
        ]
        
        negotiation = Negotiation(
            negotiation_id=auction_id,
            negotiation_type=NegotiationType.AUCTION,
            issues=issues,
            participants=[auctioneer] + [bidder.agent_id for bidder in bidders]
        )
        
        self.active_negotiations[auction_id] = negotiation
        self.stats['negotiations_started'] += 1
        
        # Run auction
        result = await self._run_auction(negotiation, auctioneer, bidders, auction_type)
        
        # Move to completed
        self.completed_negotiations.append(negotiation)
        del self.active_negotiations[auction_id]
        
        return result
    
    async def _run_bilateral_negotiation(self, negotiation: Negotiation,
                                       agent1: NegotiationAgent, 
                                       agent2: NegotiationAgent) -> Negotiation:
        """Run bilateral negotiation process"""
        
        self.logger.info(f"Starting bilateral negotiation: {negotiation.negotiation_id}")
        negotiation.status = NegotiationStatus.IN_PROGRESS
        
        # Agent1 makes initial proposal
        current_proposer = agent1
        other_agent = agent2
        
        try:
            initial_proposal = await current_proposer.generate_initial_proposal(negotiation)
            negotiation.add_proposal(initial_proposal)
            negotiation.current_round = 1
            
            self.logger.info(f"Initial proposal from {current_proposer.agent_id}: {initial_proposal.issues}")
            
            while not negotiation.is_complete():
                # Other agent responds
                response = await other_agent.respond_to_proposal(initial_proposal, negotiation)
                negotiation.add_proposal(response)
                negotiation.current_round += 1
                
                self.logger.info(f"Round {negotiation.current_round}: {other_agent.agent_id} responds with {response.proposal_type.value}")
                
                # Check if agreement reached
                if response.proposal_type == ProposalType.ACCEPTANCE:
                    negotiation.finalize_agreement(response)
                    self.stats['successful_agreements'] += 1
                    self.logger.info(f"Agreement reached in negotiation {negotiation.negotiation_id}")
                    break
                
                elif response.proposal_type == ProposalType.REJECTION:
                    negotiation.status = NegotiationStatus.FAILED
                    self.stats['failed_negotiations'] += 1
                    self.logger.info(f"Negotiation {negotiation.negotiation_id} failed - proposal rejected")
                    break
                
                # Continue with counter-offers
                initial_proposal = response
                current_proposer, other_agent = other_agent, current_proposer
                
                # Add small delay to simulate thinking time
                await asyncio.sleep(0.1)
            
            if negotiation.current_round >= negotiation.max_rounds:
                negotiation.status = NegotiationStatus.TIMEOUT
                self.stats['failed_negotiations'] += 1
                self.logger.info(f"Negotiation {negotiation.negotiation_id} timed out")
        
        except Exception as e:
            self.logger.error(f"Error in negotiation {negotiation.negotiation_id}: {e}")
            negotiation.status = NegotiationStatus.FAILED
            self.stats['failed_negotiations'] += 1
        
        # Update statistics
        self.stats['total_rounds'] += negotiation.current_round
        negotiation_time = time.time() - negotiation.started_at
        
        current_avg = self.stats['average_negotiation_time']
        total_negotiations = self.stats['successful_agreements'] + self.stats['failed_negotiations']
        
        if total_negotiations > 0:
            self.stats['average_negotiation_time'] = (
                (current_avg * (total_negotiations - 1) + negotiation_time) / total_negotiations
            )
        
        return negotiation
    
    async def _run_auction(self, negotiation: Negotiation, auctioneer: str,
                          bidders: List[NegotiationAgent], auction_type: str) -> Negotiation:
        """Run auction-based negotiation"""
        
        self.logger.info(f"Starting {auction_type} auction: {negotiation.negotiation_id}")
        negotiation.status = NegotiationStatus.IN_PROGRESS
        
        if auction_type == "english":
            # English auction: ascending price
            current_bid = 0.0
            current_winner = None
            
            for round_num in range(1, negotiation.max_rounds + 1):
                negotiation.current_round = round_num
                round_bids = []
                
                # Collect bids from all bidders
                for bidder in bidders:
                    # Simple bidding: increase by random amount
                    bid_increase = random.uniform(10, 100)
                    bid_amount = current_bid + bid_increase
                    
                    # Check if bidder wants to bid (simplified)
                    if random.random() > 0.3:  # 70% chance to bid
                        bid_proposal = Proposal(
                            id="",
                            negotiation_id=negotiation.negotiation_id,
                            proposer_id=bidder.agent_id,
                            proposal_type=ProposalType.INITIAL_OFFER,
                            issues={"bid_amount": bid_amount},
                            round_number=round_num
                        )
                        
                        round_bids.append((bidder.agent_id, bid_amount, bid_proposal))
                        negotiation.add_proposal(bid_proposal)
                
                if round_bids:
                    # Find highest bid
                    winner_id, winning_bid, winning_proposal = max(round_bids, key=lambda x: x[1])
                    current_bid = winning_bid
                    current_winner = winner_id
                    
                    self.logger.info(f"Round {round_num}: Highest bid ${winning_bid:.2f} from {winner_id}")
                else:
                    # No bids - auction ends
                    break
                
                await asyncio.sleep(0.1)  # Brief pause between rounds
            
            if current_winner:
                # Create winning proposal
                winning_proposal = Proposal(
                    id="",
                    negotiation_id=negotiation.negotiation_id,
                    proposer_id=current_winner,
                    proposal_type=ProposalType.ACCEPTANCE,
                    issues={"bid_amount": current_bid, "winner": current_winner},
                    round_number=negotiation.current_round
                )
                
                negotiation.finalize_agreement(winning_proposal)
                negotiation.winning_agent = current_winner
                self.stats['successful_agreements'] += 1
                
                self.logger.info(f"Auction won by {current_winner} with bid ${current_bid:.2f}")
            else:
                negotiation.status = NegotiationStatus.FAILED
                self.stats['failed_negotiations'] += 1
                self.logger.info("Auction failed - no bids received")
        
        return negotiation
    
    def get_negotiation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive negotiation statistics"""
        success_rate = 0.0
        total_negotiations = self.stats['successful_agreements'] + self.stats['failed_negotiations']
        
        if total_negotiations > 0:
            success_rate = self.stats['successful_agreements'] / total_negotiations
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'active_negotiations': len(self.active_negotiations),
            'completed_negotiations': len(self.completed_negotiations),
            'registered_agents': len(self.registered_agents)
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_simple_bilateral_negotiation():
    """Demo: Simple bilateral price negotiation"""
    print("\nDEMO 1: BILATERAL PRICE NEGOTIATION")
    print("=" * 50)
    
    protocol = NegotiationProtocol()
    
    # Define negotiation issues
    issues = [
        NegotiationIssue("price", "numeric", 100, 1000, weight=0.7),
        NegotiationIssue("delivery_days", "numeric", 1, 30, weight=0.3)
    ]
    
    # Create buyer (wants low price, fast delivery)
    buyer = SimpleNegotiationAgent("buyer", NegotiationStrategy.COMPETITIVE)
    buyer_utility = UtilityFunction(issues)
    buyer_utility.reservation_value = 0.3
    buyer.minimum_utility = 0.3
    buyer.target_utility = 0.8
    buyer.set_utility_function(buyer_utility)
    
    # Create seller (wants high price, flexible delivery)
    seller = SimpleNegotiationAgent("seller", NegotiationStrategy.COOPERATIVE)
    seller_utility = UtilityFunction(issues)
    seller_utility.reservation_value = 0.4
    seller.minimum_utility = 0.4
    seller.target_utility = 0.7
    seller.set_utility_function(seller_utility)
    
    print(f"Buyer strategy: {buyer.strategy.value} (wants low price, fast delivery)")
    print(f"Seller strategy: {seller.strategy.value} (wants high price, flexible delivery)")
    
    # Register agents
    protocol.register_agent(buyer)
    protocol.register_agent(seller)
    
    # Start negotiation
    negotiation = await protocol.start_bilateral_negotiation(
        "laptop_purchase", issues, buyer, seller, max_rounds=6
    )
    
    # Show results
    print(f"\nNegotiation Result:")
    print(f"  Status: {negotiation.status.value}")
    print(f"  Rounds: {negotiation.current_round}")
    print(f"  Duration: {time.time() - negotiation.started_at:.2f}s")
    
    if negotiation.final_agreement:
        print(f"  Agreement: {negotiation.final_agreement}")
        
        # Calculate utilities for both parties
        buyer_final_utility = buyer.utility_function.evaluate(negotiation.final_agreement)
        seller_final_utility = seller.utility_function.evaluate(negotiation.final_agreement)
        
        print(f"  Buyer utility: {buyer_final_utility:.2f}")
        print(f"  Seller utility: {seller_final_utility:.2f}")
    else:
        print("  No agreement reached")

async def demo_multi_issue_negotiation():
    """Demo: Multi-issue negotiation with complex preferences"""
    print("\nDEMO 2: MULTI-ISSUE CONTRACT NEGOTIATION")
    print("=" * 50)
    
    protocol = NegotiationProtocol()
    
    # Define complex contract issues
    issues = [
        NegotiationIssue("price", "numeric", 50000, 200000, weight=0.4),
        NegotiationIssue("duration_months", "numeric", 6, 24, weight=0.3),
        NegotiationIssue("payment_terms", "categorical", possible_values=["upfront", "monthly", "milestone"], weight=0.2),
        NegotiationIssue("support_included", "boolean", weight=0.1)
    ]
    
    # Create client (wants lower price, shorter duration, milestone payments, with support)
    client = SimpleNegotiationAgent("client", NegotiationStrategy.TIME_DEPENDENT)
    client_utility = UtilityFunction(issues)
    client.set_utility_function(client_utility)
    
    # Create vendor (wants higher price, longer duration, upfront payment, no support)
    vendor = SimpleNegotiationAgent("vendor", NegotiationStrategy.COOPERATIVE)
    vendor_utility = UtilityFunction(issues)
    vendor.set_utility_function(vendor_utility)
    
    print("Negotiating software development contract...")
    print("Issues: price, duration, payment terms, support inclusion")
    
    protocol.register_agent(client)
    protocol.register_agent(vendor)
    
    # Start negotiation
    negotiation = await protocol.start_bilateral_negotiation(
        "software_contract", issues, client, vendor, max_rounds=8
    )
    
    print(f"\nContract Negotiation Result:")
    print(f"  Status: {negotiation.status.value}")
    print(f"  Negotiation rounds: {negotiation.current_round}")
    
    if negotiation.final_agreement:
        agreement = negotiation.final_agreement
        print(f"  Final contract terms:")
        print(f"    - Price: ${agreement.get('price', 0):,.2f}")
        print(f"    - Duration: {agreement.get('duration_months', 0)} months")
        print(f"    - Payment: {agreement.get('payment_terms', 'N/A')}")
        print(f"    - Support: {'Yes' if agreement.get('support_included', False) else 'No'}")
    
    # Show proposal history
    print(f"\n  Proposal history ({len(negotiation.proposals)} proposals):")
    for i, proposal in enumerate(negotiation.proposals[-4:], 1):  # Show last 4 proposals
        print(f"    {i}. {proposal.proposer_id}: {proposal.proposal_type.value}")

async def demo_auction_negotiation():
    """Demo: Auction-based resource allocation"""
    print("\nDEMO 3: AUCTION-BASED RESOURCE ALLOCATION")
    print("=" * 50)
    
    protocol = NegotiationProtocol()
    
    # Create multiple bidders for cloud computing resources
    bidders = []
    for i in range(5):
        bidder_strategy = random.choice([
            NegotiationStrategy.COMPETITIVE,
            NegotiationStrategy.COOPERATIVE,
            NegotiationStrategy.TIME_DEPENDENT
        ])
        
        bidder = SimpleNegotiationAgent(f"company_{i+1}", bidder_strategy)
        bidders.append(bidder)
        protocol.register_agent(bidder)
    
    print(f"Auctioning cloud compute instances to {len(bidders)} companies")
    print("Bidder strategies:", [b.strategy.value for b in bidders])
    
    # Item being auctioned
    cloud_instance = {
        "type": "high_performance_compute",
        "cpu_cores": 64,
        "memory_gb": 256,
        "storage_tb": 10,
        "duration_hours": 24
    }
    
    print(f"Item: {cloud_instance['type']} - {cloud_instance['cpu_cores']} cores, {cloud_instance['memory_gb']}GB RAM")
    
    # Start English auction
    auction = await protocol.start_auction(
        "cloud_instance_auction",
        cloud_instance,
        "cloud_provider",
        bidders,
        "english"
    )
    
    print(f"\nAuction Result:")
    print(f"  Status: {auction.status.value}")
    print(f"  Bidding rounds: {auction.current_round}")
    
    if auction.winning_agent and auction.final_agreement:
        winning_bid = auction.final_agreement.get('bid_amount', 0)
        print(f"  Winner: {auction.winning_agent}")
        print(f"  Winning bid: ${winning_bid:.2f}")
        print(f"  Resource utilization efficiency achieved")
    else:
        print("  Auction failed - no winner")

async def demo_negotiation_strategies():
    """Demo: Different negotiation strategies comparison"""
    print("\nDEMO 4: NEGOTIATION STRATEGIES COMPARISON")
    print("=" * 50)
    
    protocol = NegotiationProtocol()
    
    # Test different strategy combinations
    strategy_pairs = [
        (NegotiationStrategy.COMPETITIVE, NegotiationStrategy.COOPERATIVE),
        (NegotiationStrategy.COOPERATIVE, NegotiationStrategy.COOPERATIVE),
        (NegotiationStrategy.TIME_DEPENDENT, NegotiationStrategy.COMPETITIVE),
        (NegotiationStrategy.TIT_FOR_TAT, NegotiationStrategy.TIME_DEPENDENT)
    ]
    
    # Simple negotiation issue
    issues = [
        NegotiationIssue("offer_amount", "numeric", 0, 1000, weight=1.0)
    ]
    
    results = []
    
    for i, (strategy1, strategy2) in enumerate(strategy_pairs):
        print(f"\nTest {i+1}: {strategy1.value} vs {strategy2.value}")
        
        # Create agents with different strategies
        agent1 = SimpleNegotiationAgent(f"agent1_{i}", strategy1)
        agent2 = SimpleNegotiationAgent(f"agent2_{i}", strategy2)
        
        # Set utility functions
        utility1 = UtilityFunction(issues)
        utility2 = UtilityFunction(issues)
        
        agent1.set_utility_function(utility1)
        agent2.set_utility_function(utility2)
        
        protocol.register_agent(agent1)
        protocol.register_agent(agent2)
        
        # Run negotiation
        negotiation = await protocol.start_bilateral_negotiation(
            f"strategy_test_{i}", issues, agent1, agent2, max_rounds=5
        )
        
        result = {
            'strategies': (strategy1.value, strategy2.value),
            'status': negotiation.status.value,
            'rounds': negotiation.current_round,
            'agreement': negotiation.final_agreement,
            'duration': time.time() - negotiation.started_at
        }
        
        results.append(result)
        
        print(f"  Result: {result['status']} in {result['rounds']} rounds")
        if result['agreement']:
            print(f"  Agreement: {result['agreement']}")
    
    # Summary
    print(f"\nStrategy Comparison Summary:")
    successful = [r for r in results if r['status'] == 'agreement_reached']
    print(f"  Successful negotiations: {len(successful)}/{len(results)}")
    
    if successful:
        avg_rounds = sum(r['rounds'] for r in successful) / len(successful)
        print(f"  Average rounds to agreement: {avg_rounds:.1f}")

async def demo_coalition_formation():
    """Demo: Simple coalition formation negotiation"""
    print("\nDEMO 5: COALITION FORMATION")
    print("=" * 50)
    
    # Simplified coalition formation demo
    print("Simulating coalition formation for joint project bid...")
    
    # Define project requirements and agent capabilities
    project_requirements = {
        "software_dev": 100,
        "marketing": 50,
        "design": 30
    }
    
    companies = {
        "TechCorp": {"software_dev": 80, "marketing": 20, "design": 10},
        "MarketPro": {"software_dev": 20, "marketing": 90, "design": 15},
        "DesignHub": {"software_dev": 30, "marketing": 10, "design": 80},
        "FullStack": {"software_dev": 60, "marketing": 60, "design": 40}
    }
    
    print(f"Project needs: {project_requirements}")
    print("Company capabilities:")
    for company, capabilities in companies.items():
        print(f"  {company}: {capabilities}")
    
    # Simple coalition formation logic
    print(f"\nEvaluating possible coalitions...")
    
    # Check all possible 2-company coalitions
    company_names = list(companies.keys())
    successful_coalitions = []
    
    for i in range(len(company_names)):
        for j in range(i+1, len(company_names)):
            company1, company2 = company_names[i], company_names[j]
            
            # Calculate combined capabilities
            combined = {}
            for capability in project_requirements:
                cap1 = companies[company1].get(capability, 0)
                cap2 = companies[company2].get(capability, 0)
                combined[capability] = cap1 + cap2
            
            # Check if coalition meets requirements
            can_complete = all(
                combined[cap] >= project_requirements[cap] 
                for cap in project_requirements
            )
            
            if can_complete:
                efficiency = sum(
                    combined[cap] / project_requirements[cap] 
                    for cap in project_requirements
                ) / len(project_requirements)
                
                successful_coalitions.append({
                    'coalition': [company1, company2],
                    'capabilities': combined,
                    'efficiency': efficiency
                })
                
                print(f"  ✅ {company1} + {company2}: efficiency {efficiency:.2f}")
            else:
                print(f"  ❌ {company1} + {company2}: insufficient capabilities")
    
    if successful_coalitions:
        # Find most efficient coalition
        best_coalition = max(successful_coalitions, key=lambda x: x['efficiency'])
        print(f"\nOptimal coalition: {' + '.join(best_coalition['coalition'])}")
        print(f"Efficiency score: {best_coalition['efficiency']:.2f}")
        print(f"Combined capabilities: {best_coalition['capabilities']}")
    else:
        print("No viable 2-company coalitions found")

async def main():
    """
    Demonstrate Negotiation Protocols for strategic agent bargaining
    
    WHAT YOU'LL LEARN:
    ================
    1. How to design multi-agent negotiation systems
    2. How to implement different negotiation strategies and protocols
    3. How to handle multi-issue negotiations with complex preferences
    4. How to run auction-based resource allocation mechanisms
    5. How to evaluate negotiation outcomes and strategy effectiveness
    
    REAL WORLD APPLICATIONS:
    =======================
    - Automated contract negotiation and deal-making
    - Resource allocation in cloud computing and distributed systems
    - Supply chain coordination and vendor negotiations
    - Real estate and financial market transactions
    - Coalition formation for collaborative projects
    - Conflict resolution in multi-agent systems
    """
    
    print("NEGOTIATION PROTOCOLS DEMONSTRATION")
    print("Showing how agents negotiate and reach strategic agreements!")
    
    await demo_simple_bilateral_negotiation()
    await demo_multi_issue_negotiation()
    await demo_auction_negotiation()
    await demo_negotiation_strategies()
    await demo_coalition_formation()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Negotiation protocols enable strategic agent interactions")
    print("✓ Different strategies produce different outcomes")
    print("✓ Multi-issue negotiations handle complex preferences")
    print("✓ Auctions provide efficient resource allocation")
    print("✓ Utility functions quantify agent preferences")
    print("✓ Coalition formation solves collaborative challenges")
    print("\nTHE POWER OF NEGOTIATION PROTOCOLS:")
    print("- Enables automated deal-making and contract negotiation")
    print("- Provides fair and efficient resource allocation")
    print("- Supports complex multi-party strategic interactions")
    print("- Creates win-win outcomes through cooperative strategies")

if __name__ == "__main__":
    asyncio.run(main())
