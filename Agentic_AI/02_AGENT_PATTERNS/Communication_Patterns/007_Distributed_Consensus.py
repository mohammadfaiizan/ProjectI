#!/usr/bin/env python3
"""
Distributed Consensus: Agreement in Unreliable Networks
=======================================================

WHAT IS THE PROBLEM?
==================
Distributed agents need to agree on shared state despite:
- Network partitions that split agent groups
- Message delays and packet loss
- Agent failures and byzantine behavior
- Concurrent updates from multiple sources
- No global clock for ordering events

Example: Bank Account Chaos
NAIVE APPROACH (Dangerous):
- Multiple bank branches update same account
- Branch A sees balance $1000, approves $800 withdrawal
- Branch B sees balance $1000, approves $600 withdrawal  
- Customer now has $1400 but account shows -$400
- No way to detect or resolve conflicts

REAL WORLD EXAMPLE:
=================
How does your bank maintain consistent balances across branches?

BLOCKCHAIN CONSENSUS EXAMPLE:
When you send Bitcoin:
1. PROPOSE: Transaction broadcast to network nodes
2. VALIDATE: Nodes verify transaction legitimacy
3. COMPETE: Miners compete to add block to chain
4. CONSENSUS: Network agrees on winning block
5. COMMIT: Transaction permanently recorded
6. FINALITY: All nodes converge on same ledger state

BENEFITS:
- Prevents double-spending without central authority
- Maintains consistency across thousands of nodes
- Tolerates Byzantine failures and network partitions
- Provides irreversible transaction finality
- Scales globally with cryptographic security

THE ALGORITHM:
=============
1. PROPOSAL: Leader proposes new state/transaction
2. PREPARE: Followers acknowledge proposal readiness
3. PROMISE: Followers promise not to accept older proposals
4. ACCEPT: Leader sends accept message with chosen value
5. COMMIT: Followers commit the agreed value
6. FINALITY: All correct nodes converge on same state

CONSENSUS TYPES:
- Crash Fault Tolerant (CFT): Handles node crashes
- Byzantine Fault Tolerant (BFT): Handles malicious nodes
- Probabilistic: Eventually consistent with high probability
- Deterministic: Guaranteed consistency under assumptions

WHY IS THIS CRITICAL?
===================
- Enables distributed databases and blockchain systems
- Provides foundation for distributed computing
- Ensures data consistency without single points of failure
- Powers modern cloud infrastructure and microservices
- Enables decentralized autonomous organizations
- Critical for financial systems and mission-critical applications
"""

import asyncio
import time
import json
import uuid
import random
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ConsensusType(Enum):
    """Types of consensus algorithms"""
    RAFT = "raft"                           # Leader-based consensus
    PBFT = "pbft"                          # Practical Byzantine Fault Tolerance
    PAXOS = "paxos"                        # Classic consensus algorithm
    POW = "proof_of_work"                  # Blockchain proof of work
    POS = "proof_of_stake"                 # Blockchain proof of stake
    GOSSIP = "gossip"                      # Epidemic consensus

class NodeRole(Enum):
    """Roles in consensus protocol"""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    VALIDATOR = "validator"
    OBSERVER = "observer"

class MessageType(Enum):
    """Types of consensus messages"""
    PROPOSAL = "proposal"
    PREPARE = "prepare"
    PROMISE = "promise"
    ACCEPT = "accept"
    COMMIT = "commit"
    HEARTBEAT = "heartbeat"
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    BLOCK_PROPOSAL = "block_proposal"
    BLOCK_VOTE = "block_vote"

class ConsensusState(Enum):
    """State of consensus process"""
    PREPARING = "preparing"
    PROPOSING = "proposing"
    VOTING = "voting"
    COMMITTING = "committing"
    COMMITTED = "committed"
    FAILED = "failed"

@dataclass
class ConsensusMessage:
    """Message for consensus protocol"""
    id: str
    message_type: MessageType
    sender_id: str
    term: int                               # Logical time/epoch
    
    # Message content
    proposal_id: Optional[str] = None
    value: Any = None
    sequence_number: int = 0
    
    # Consensus metadata
    view_number: int = 0                    # View in PBFT
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None
    
    # Raft-specific
    leader_id: Optional[str] = None
    last_log_index: int = 0
    last_log_term: int = 0
    commit_index: int = 0
    
    # PBFT-specific
    digest: Optional[str] = None
    client_id: Optional[str] = None
    request_timestamp: Optional[float] = None
    
    def __post_init__(self):
        """Initialize message"""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def calculate_digest(self) -> str:
        """Calculate message digest for integrity"""
        content = f"{self.message_type.value}:{self.sender_id}:{self.term}:{self.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def serialize(self) -> str:
        """Serialize message to JSON"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        return json.dumps(data)
    
    @classmethod
    def deserialize(cls, data: str) -> 'ConsensusMessage':
        """Deserialize message from JSON"""
        obj = json.loads(data)
        obj['message_type'] = MessageType(obj['message_type'])
        return cls(**obj)

@dataclass
class LogEntry:
    """Entry in the consensus log"""
    index: int
    term: int
    value: Any
    timestamp: float = field(default_factory=time.time)
    committed: bool = False
    client_id: Optional[str] = None
    
    def serialize(self) -> str:
        """Serialize log entry"""
        return json.dumps(asdict(self))

class ConsensusNode(ABC):
    """Abstract base class for consensus nodes"""
    
    def __init__(self, node_id: str, cluster_size: int):
        self.node_id = node_id
        self.cluster_size = cluster_size
        self.role = NodeRole.FOLLOWER
        
        # Consensus state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Network and timing
        self.peers: Set[str] = set()
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.election_timeout = random.uniform(150, 300)  # milliseconds
        self.heartbeat_interval = 50  # milliseconds
        
        # State machine
        self.state_machine: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'elections_started': 0,
            'proposals_made': 0,
            'votes_cast': 0
        }
        
        self.logger = logging.getLogger(f"Node-{node_id}")
        self.running = False
        self.last_heartbeat = time.time()
    
    @abstractmethod
    async def start(self) -> None:
        """Start the consensus node"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the consensus node"""
        pass
    
    @abstractmethod
    async def propose_value(self, value: Any) -> bool:
        """Propose a value for consensus"""
        pass
    
    @abstractmethod
    async def handle_message(self, message: ConsensusMessage) -> Optional[ConsensusMessage]:
        """Handle incoming consensus message"""
        pass
    
    def add_peer(self, peer_id: str) -> None:
        """Add peer to cluster"""
        self.peers.add(peer_id)
    
    def remove_peer(self, peer_id: str) -> None:
        """Remove peer from cluster"""
        self.peers.discard(peer_id)
    
    def majority_size(self) -> int:
        """Calculate majority size for consensus"""
        return (self.cluster_size // 2) + 1
    
    def apply_to_state_machine(self, entry: LogEntry) -> None:
        """Apply committed entry to state machine"""
        if entry.value and isinstance(entry.value, dict):
            operation = entry.value.get('operation')
            key = entry.value.get('key')
            value = entry.value.get('value')
            
            if operation == 'set' and key:
                self.state_machine[key] = value
                self.logger.info(f"Applied: SET {key} = {value}")
            elif operation == 'delete' and key:
                self.state_machine.pop(key, None)
                self.logger.info(f"Applied: DELETE {key}")

class RaftNode(ConsensusNode):
    """Raft consensus algorithm implementation"""
    
    def __init__(self, node_id: str, cluster_size: int):
        super().__init__(node_id, cluster_size)
        
        # Raft-specific state
        self.leader_id: Optional[str] = None
        self.next_index: Dict[str, int] = {}  # For each peer
        self.match_index: Dict[str, int] = {}  # For each peer
        
        # Timing
        self.election_timer: Optional[asyncio.Task] = None
        self.heartbeat_timer: Optional[asyncio.Task] = None
        
        # Message handlers
        self.message_handlers = {
            MessageType.VOTE_REQUEST: self.handle_vote_request,
            MessageType.VOTE_RESPONSE: self.handle_vote_response,
            MessageType.PROPOSAL: self.handle_append_entries,
            MessageType.HEARTBEAT: self.handle_append_entries
        }
    
    async def start(self) -> None:
        """Start Raft node"""
        self.running = True
        self.logger.info(f"Starting Raft node {self.node_id}")
        
        # Initialize peer indices
        for peer_id in self.peers:
            self.next_index[peer_id] = len(self.log) + 1
            self.match_index[peer_id] = 0
        
        # Start election timer
        await self.reset_election_timer()
    
    async def stop(self) -> None:
        """Stop Raft node"""
        self.running = False
        
        if self.election_timer:
            self.election_timer.cancel()
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
        
        self.logger.info(f"Stopped Raft node {self.node_id}")
    
    async def propose_value(self, value: Any) -> bool:
        """Propose value (only leader can propose)"""
        if self.role != NodeRole.LEADER:
            self.logger.warning(f"Cannot propose - not leader (role: {self.role.value})")
            return False
        
        # Create log entry
        entry = LogEntry(
            index=len(self.log) + 1,
            term=self.current_term,
            value=value
        )
        
        self.log.append(entry)
        self.stats['proposals_made'] += 1
        
        self.logger.info(f"Leader proposing value: {value}")
        
        # Replicate to followers
        await self.replicate_log_entries()
        
        return True
    
    async def handle_message(self, message: ConsensusMessage) -> Optional[ConsensusMessage]:
        """Handle incoming Raft message"""
        self.stats['messages_received'] += 1
        
        # Update term if message has higher term
        if message.term > self.current_term:
            self.current_term = message.term
            self.voted_for = None
            if self.role != NodeRole.FOLLOWER:
                await self.become_follower()
        
        # Reject messages from lower terms
        if message.term < self.current_term:
            return None
        
        handler = self.message_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        
        return None
    
    async def handle_vote_request(self, message: ConsensusMessage) -> ConsensusMessage:
        """Handle vote request from candidate"""
        vote_granted = False
        
        # Check if we can vote for this candidate
        if (self.voted_for is None or self.voted_for == message.sender_id):
            # Check if candidate's log is at least as up-to-date as ours
            candidate_log_ok = True
            if self.log:
                last_log_term = self.log[-1].term
                last_log_index = len(self.log)
                
                if (message.last_log_term < last_log_term or 
                    (message.last_log_term == last_log_term and 
                     message.last_log_index < last_log_index)):
                    candidate_log_ok = False
            
            if candidate_log_ok:
                vote_granted = True
                self.voted_for = message.sender_id
                await self.reset_election_timer()
                self.logger.info(f"Voted for {message.sender_id} in term {message.term}")
        
        self.stats['votes_cast'] += 1
        
        return ConsensusMessage(
            id="",
            message_type=MessageType.VOTE_RESPONSE,
            sender_id=self.node_id,
            term=self.current_term,
            value=vote_granted
        )
    
    async def handle_vote_response(self, message: ConsensusMessage) -> None:
        """Handle vote response (only relevant for candidates)"""
        if self.role != NodeRole.CANDIDATE:
            return
        
        if message.value:  # Vote granted
            self.votes_received += 1
            self.logger.info(f"Received vote from {message.sender_id} ({self.votes_received}/{self.majority_size()})")
            
            if self.votes_received >= self.majority_size():
                await self.become_leader()
    
    async def handle_append_entries(self, message: ConsensusMessage) -> ConsensusMessage:
        """Handle append entries (log replication and heartbeat)"""
        success = False
        
        # Reset election timer - we heard from leader
        await self.reset_election_timer()
        self.leader_id = message.leader_id
        
        if self.role == NodeRole.CANDIDATE:
            await self.become_follower()
        
        # Check log consistency
        if message.last_log_index == 0 or (
            len(self.log) >= message.last_log_index and
            self.log[message.last_log_index - 1].term == message.last_log_term
        ):
            success = True
            
            # Append new entries if any
            if message.value and isinstance(message.value, list):
                # Remove conflicting entries
                if len(self.log) > message.last_log_index:
                    self.log = self.log[:message.last_log_index]
                
                # Append new entries
                for entry_data in message.value:
                    entry = LogEntry(**entry_data)
                    self.log.append(entry)
            
            # Update commit index
            if message.commit_index > self.commit_index:
                old_commit_index = self.commit_index
                self.commit_index = min(message.commit_index, len(self.log))
                
                # Apply newly committed entries
                for i in range(old_commit_index, self.commit_index):
                    self.apply_to_state_machine(self.log[i])
                    self.log[i].committed = True
        
        return ConsensusMessage(
            id="",
            message_type=MessageType.PROMISE,
            sender_id=self.node_id,
            term=self.current_term,
            value=success,
            last_log_index=len(self.log)
        )
    
    async def start_election(self) -> None:
        """Start leader election"""
        self.current_term += 1
        self.role = NodeRole.CANDIDATE
        self.voted_for = self.node_id
        self.votes_received = 1  # Vote for self
        
        self.stats['elections_started'] += 1
        self.logger.info(f"Starting election for term {self.current_term}")
        
        # Send vote requests to all peers
        last_log_index = len(self.log)
        last_log_term = self.log[-1].term if self.log else 0
        
        vote_request = ConsensusMessage(
            id="",
            message_type=MessageType.VOTE_REQUEST,
            sender_id=self.node_id,
            term=self.current_term,
            last_log_index=last_log_index,
            last_log_term=last_log_term
        )
        
        # In a real implementation, you would send this to all peers
        # Here we simulate the process
        await self.reset_election_timer()
    
    async def become_leader(self) -> None:
        """Become cluster leader"""
        self.role = NodeRole.LEADER
        self.leader_id = self.node_id
        
        # Initialize leader state
        for peer_id in self.peers:
            self.next_index[peer_id] = len(self.log) + 1
            self.match_index[peer_id] = 0
        
        self.logger.info(f"Became leader for term {self.current_term}")
        
        # Start sending heartbeats
        await self.start_heartbeat_timer()
        
        # Send initial heartbeat
        await self.send_heartbeats()
    
    async def become_follower(self) -> None:
        """Become follower"""
        self.role = NodeRole.FOLLOWER
        
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
            self.heartbeat_timer = None
        
        await self.reset_election_timer()
    
    async def reset_election_timer(self) -> None:
        """Reset election timeout timer"""
        if self.election_timer:
            self.election_timer.cancel()
        
        timeout = random.uniform(150, 300) / 1000  # Convert to seconds
        self.election_timer = asyncio.create_task(self.election_timeout())
    
    async def election_timeout(self) -> None:
        """Handle election timeout"""
        await asyncio.sleep(self.election_timeout / 1000)
        
        if self.running and self.role != NodeRole.LEADER:
            await self.start_election()
    
    async def start_heartbeat_timer(self) -> None:
        """Start heartbeat timer for leader"""
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
        
        self.heartbeat_timer = asyncio.create_task(self.heartbeat_loop())
    
    async def heartbeat_loop(self) -> None:
        """Send periodic heartbeats as leader"""
        while self.running and self.role == NodeRole.LEADER:
            await self.send_heartbeats()
            await asyncio.sleep(self.heartbeat_interval / 1000)
    
    async def send_heartbeats(self) -> None:
        """Send heartbeat to all followers"""
        heartbeat = ConsensusMessage(
            id="",
            message_type=MessageType.HEARTBEAT,
            sender_id=self.node_id,
            term=self.current_term,
            leader_id=self.node_id,
            commit_index=self.commit_index
        )
        
        self.stats['messages_sent'] += len(self.peers)
        # In real implementation, send to all peers
    
    async def replicate_log_entries(self) -> None:
        """Replicate log entries to followers"""
        for peer_id in self.peers:
            await self.send_append_entries(peer_id)
    
    async def send_append_entries(self, peer_id: str) -> None:
        """Send append entries to specific peer"""
        next_index = self.next_index[peer_id]
        prev_log_index = next_index - 1
        prev_log_term = 0
        
        if prev_log_index > 0 and prev_log_index <= len(self.log):
            prev_log_term = self.log[prev_log_index - 1].term
        
        # Get entries to send
        entries = []
        if next_index <= len(self.log):
            entries = [asdict(entry) for entry in self.log[next_index - 1:]]
        
        append_entries = ConsensusMessage(
            id="",
            message_type=MessageType.PROPOSAL,
            sender_id=self.node_id,
            term=self.current_term,
            leader_id=self.node_id,
            last_log_index=prev_log_index,
            last_log_term=prev_log_term,
            value=entries,
            commit_index=self.commit_index
        )
        
        self.stats['messages_sent'] += 1

class PBFTNode(ConsensusNode):
    """Practical Byzantine Fault Tolerance implementation"""
    
    def __init__(self, node_id: str, cluster_size: int):
        super().__init__(node_id, cluster_size)
        
        # PBFT state
        self.view_number = 0
        self.sequence_number = 0
        self.is_primary = False
        self.request_queue: deque = deque()
        
        # Message logs
        self.prepare_messages: Dict[str, List[ConsensusMessage]] = defaultdict(list)
        self.commit_messages: Dict[str, List[ConsensusMessage]] = defaultdict(list)
        
        # Message handlers
        self.message_handlers = {
            MessageType.PROPOSAL: self.handle_pre_prepare,
            MessageType.PREPARE: self.handle_prepare,
            MessageType.COMMIT: self.handle_commit
        }
    
    async def start(self) -> None:
        """Start PBFT node"""
        self.running = True
        self.logger.info(f"Starting PBFT node {self.node_id}")
        
        # Determine if this node is primary for view 0
        self.is_primary = (self.view_number % self.cluster_size) == int(self.node_id.split('_')[-1])
    
    async def stop(self) -> None:
        """Stop PBFT node"""
        self.running = False
        self.logger.info(f"Stopped PBFT node {self.node_id}")
    
    async def propose_value(self, value: Any) -> bool:
        """Propose value (only primary can propose)"""
        if not self.is_primary:
            self.logger.warning("Cannot propose - not primary")
            return False
        
        self.sequence_number += 1
        
        # Create pre-prepare message
        pre_prepare = ConsensusMessage(
            id="",
            message_type=MessageType.PROPOSAL,
            sender_id=self.node_id,
            term=self.current_term,
            view_number=self.view_number,
            sequence_number=self.sequence_number,
            value=value,
            digest=self.calculate_digest(value)
        )
        
        self.stats['proposals_made'] += 1
        self.logger.info(f"Primary proposing value: {value}")
        
        # In real implementation, broadcast to all replicas
        return True
    
    async def handle_message(self, message: ConsensusMessage) -> Optional[ConsensusMessage]:
        """Handle incoming PBFT message"""
        self.stats['messages_received'] += 1
        
        handler = self.message_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        
        return None
    
    async def handle_pre_prepare(self, message: ConsensusMessage) -> Optional[ConsensusMessage]:
        """Handle pre-prepare message from primary"""
        # Validate message
        if (message.view_number == self.view_number and
            message.sequence_number > self.sequence_number and
            self.validate_digest(message.value, message.digest)):
            
            self.sequence_number = message.sequence_number
            
            # Send prepare message
            prepare = ConsensusMessage(
                id="",
                message_type=MessageType.PREPARE,
                sender_id=self.node_id,
                term=self.current_term,
                view_number=message.view_number,
                sequence_number=message.sequence_number,
                digest=message.digest
            )
            
            self.stats['messages_sent'] += 1
            self.logger.info(f"Sending prepare for sequence {message.sequence_number}")
            
            return prepare
        
        return None
    
    async def handle_prepare(self, message: ConsensusMessage) -> Optional[ConsensusMessage]:
        """Handle prepare message from replica"""
        key = f"{message.view_number}:{message.sequence_number}:{message.digest}"
        self.prepare_messages[key].append(message)
        
        # Check if we have enough prepare messages (2f+1 including pre-prepare)
        if len(self.prepare_messages[key]) >= self.byzantine_threshold():
            # Send commit message
            commit = ConsensusMessage(
                id="",
                message_type=MessageType.COMMIT,
                sender_id=self.node_id,
                term=self.current_term,
                view_number=message.view_number,
                sequence_number=message.sequence_number,
                digest=message.digest
            )
            
            self.stats['messages_sent'] += 1
            self.logger.info(f"Sending commit for sequence {message.sequence_number}")
            
            return commit
        
        return None
    
    async def handle_commit(self, message: ConsensusMessage) -> None:
        """Handle commit message from replica"""
        key = f"{message.view_number}:{message.sequence_number}:{message.digest}"
        self.commit_messages[key].append(message)
        
        # Check if we have enough commit messages (2f+1)
        if len(self.commit_messages[key]) >= self.byzantine_threshold():
            # Execute the operation
            await self.execute_operation(message)
    
    async def execute_operation(self, message: ConsensusMessage) -> None:
        """Execute committed operation"""
        # Find the original value from prepare messages
        key = f"{message.view_number}:{message.sequence_number}:{message.digest}"
        
        # In real implementation, extract value from pre-prepare
        # For demo, we'll simulate
        self.logger.info(f"Executing operation for sequence {message.sequence_number}")
        
        # Apply to state machine
        if hasattr(message, 'value') and message.value:
            entry = LogEntry(
                index=message.sequence_number,
                term=self.current_term,
                value=message.value,
                committed=True
            )
            self.apply_to_state_machine(entry)
    
    def byzantine_threshold(self) -> int:
        """Calculate Byzantine fault tolerance threshold (2f+1)"""
        f = (self.cluster_size - 1) // 3  # Maximum Byzantine faults
        return 2 * f + 1
    
    def calculate_digest(self, value: Any) -> str:
        """Calculate digest for value"""
        content = json.dumps(value, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def validate_digest(self, value: Any, digest: str) -> bool:
        """Validate digest matches value"""
        return self.calculate_digest(value) == digest

class ConsensusCluster:
    """
    Manages a cluster of consensus nodes
    
    EXAMPLE USAGE:
    =============
    # Create Raft cluster
    cluster = ConsensusCluster("raft_cluster", ConsensusType.RAFT, 5)
    await cluster.start()
    
    # Propose values
    success = await cluster.propose_value({"operation": "set", "key": "balance", "value": 1000})
    
    # Check consensus
    state = cluster.get_cluster_state()
    """
    
    def __init__(self, cluster_id: str, consensus_type: ConsensusType, size: int):
        self.cluster_id = cluster_id
        self.consensus_type = consensus_type
        self.size = size
        self.nodes: Dict[str, ConsensusNode] = {}
        
        # Network simulation
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.network_partition: Set[str] = set()
        self.message_delay = 0.01  # 10ms default delay
        
        # Cluster state
        self.running = False
        self.stats = {
            'total_proposals': 0,
            'successful_consensus': 0,
            'failed_consensus': 0,
            'leader_elections': 0
        }
        
        self.logger = logging.getLogger(f"Cluster-{cluster_id}")
        
        self._create_nodes()
    
    def _create_nodes(self) -> None:
        """Create consensus nodes"""
        for i in range(self.size):
            node_id = f"node_{i}"
            
            if self.consensus_type == ConsensusType.RAFT:
                node = RaftNode(node_id, self.size)
            elif self.consensus_type == ConsensusType.PBFT:
                node = PBFTNode(node_id, self.size)
            else:
                raise ValueError(f"Unsupported consensus type: {self.consensus_type}")
            
            # Add all other nodes as peers
            for j in range(self.size):
                if j != i:
                    node.add_peer(f"node_{j}")
            
            self.nodes[node_id] = node
    
    async def start(self) -> None:
        """Start consensus cluster"""
        self.running = True
        self.logger.info(f"Starting {self.consensus_type.value} cluster with {self.size} nodes")
        
        # Start all nodes
        for node in self.nodes.values():
            await node.start()
        
        # Start message processing
        asyncio.create_task(self.process_messages())
    
    async def stop(self) -> None:
        """Stop consensus cluster"""
        self.running = False
        
        # Stop all nodes
        for node in self.nodes.values():
            await node.stop()
        
        self.logger.info("Consensus cluster stopped")
    
    async def propose_value(self, value: Any, proposer_id: str = None) -> bool:
        """Propose value for consensus"""
        self.stats['total_proposals'] += 1
        
        # Find leader or primary
        leader_node = None
        
        if self.consensus_type == ConsensusType.RAFT:
            for node in self.nodes.values():
                if isinstance(node, RaftNode) and node.role == NodeRole.LEADER:
                    leader_node = node
                    break
        elif self.consensus_type == ConsensusType.PBFT:
            for node in self.nodes.values():
                if isinstance(node, PBFTNode) and node.is_primary:
                    leader_node = node
                    break
        
        if not leader_node:
            # If no leader, trigger election (for Raft)
            if self.consensus_type == ConsensusType.RAFT:
                for node in self.nodes.values():
                    if isinstance(node, RaftNode) and node.role == NodeRole.FOLLOWER:
                        await node.start_election()
                        break
            return False
        
        # Propose value
        success = await leader_node.propose_value(value)
        
        if success:
            self.stats['successful_consensus'] += 1
        else:
            self.stats['failed_consensus'] += 1
        
        return success
    
    async def process_messages(self) -> None:
        """Process messages between nodes"""
        while self.running:
            try:
                # Simulate message processing
                await asyncio.sleep(0.01)  # Small delay to prevent busy loop
                
                # In a real implementation, this would handle message routing
                # between nodes, including network delays and partitions
                
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
    
    def partition_network(self, partition1: List[str], partition2: List[str]) -> None:
        """Simulate network partition"""
        self.network_partition = set(partition1)
        self.logger.warning(f"Network partitioned: {partition1} isolated from {partition2}")
    
    def heal_network(self) -> None:
        """Heal network partition"""
        self.network_partition.clear()
        self.logger.info("Network partition healed")
    
    def get_cluster_state(self) -> Dict[str, Any]:
        """Get current cluster state"""
        nodes_state = {}
        leader_count = 0
        
        for node_id, node in self.nodes.items():
            state = {
                'role': node.role.value,
                'term': node.current_term,
                'log_length': len(node.log),
                'commit_index': node.commit_index,
                'state_machine': node.state_machine.copy()
            }
            
            if hasattr(node, 'leader_id'):
                state['leader_id'] = node.leader_id
            
            if node.role == NodeRole.LEADER:
                leader_count += 1
            
            nodes_state[node_id] = state
        
        return {
            'cluster_id': self.cluster_id,
            'consensus_type': self.consensus_type.value,
            'cluster_size': self.size,
            'nodes': nodes_state,
            'leader_count': leader_count,
            'network_partitioned': bool(self.network_partition),
            'stats': self.stats
        }
    
    def get_consensus_state(self) -> Dict[str, Any]:
        """Check if cluster has reached consensus"""
        # Check if all nodes have same state machine
        state_machines = [node.state_machine for node in self.nodes.values()]
        
        if not state_machines:
            return {'consensus': False, 'reason': 'No nodes'}
        
        first_state = state_machines[0]
        all_same = all(state == first_state for state in state_machines)
        
        return {
            'consensus': all_same,
            'consistent_state': first_state if all_same else None,
            'node_states': {
                node_id: node.state_machine 
                for node_id, node in self.nodes.items()
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_raft_consensus():
    """Demo: Raft consensus with leader election"""
    print("\nDEMO 1: RAFT CONSENSUS ALGORITHM")
    print("=" * 50)
    
    # Create Raft cluster
    cluster = ConsensusCluster("raft_demo", ConsensusType.RAFT, 5)
    await cluster.start()
    
    print(f"Started Raft cluster with 5 nodes")
    
    # Wait for leader election
    await asyncio.sleep(1.0)
    
    # Check initial state
    state = cluster.get_cluster_state()
    print(f"Leader count: {state['leader_count']}")
    
    # Propose some values
    operations = [
        {"operation": "set", "key": "account_123", "value": 1000},
        {"operation": "set", "key": "account_456", "value": 2500},
        {"operation": "set", "key": "account_789", "value": 500}
    ]
    
    print(f"\nProposing {len(operations)} operations:")
    for i, op in enumerate(operations):
        print(f"  {i+1}. {op}")
        success = await cluster.propose_value(op)
        print(f"     Result: {'SUCCESS' if success else 'FAILED'}")
        await asyncio.sleep(0.5)  # Small delay between proposals
    
    # Wait for consensus
    await asyncio.sleep(2.0)
    
    # Check final consensus state
    consensus_state = cluster.get_consensus_state()
    print(f"\nConsensus reached: {consensus_state['consensus']}")
    
    if consensus_state['consensus']:
        print(f"Final state machine: {consensus_state['consistent_state']}")
    else:
        print("Node states differ:")
        for node_id, state in consensus_state['node_states'].items():
            print(f"  {node_id}: {state}")
    
    cluster_stats = cluster.get_cluster_state()
    print(f"\nCluster statistics: {cluster_stats['stats']}")
    
    await cluster.stop()

async def demo_pbft_consensus():
    """Demo: PBFT Byzantine fault tolerance"""
    print("\nDEMO 2: PBFT BYZANTINE FAULT TOLERANCE")
    print("=" * 50)
    
    # Create PBFT cluster (needs 3f+1 nodes for f Byzantine faults)
    cluster = ConsensusCluster("pbft_demo", ConsensusType.PBFT, 4)  # Tolerates 1 Byzantine fault
    await cluster.start()
    
    print(f"Started PBFT cluster with 4 nodes (tolerates 1 Byzantine fault)")
    
    # Wait for initialization
    await asyncio.sleep(0.5)
    
    # Propose values
    transactions = [
        {"type": "transfer", "from": "alice", "to": "bob", "amount": 100},
        {"type": "transfer", "from": "bob", "to": "charlie", "amount": 50},
        {"type": "balance_update", "account": "alice", "balance": 900}
    ]
    
    print(f"\nProposing {len(transactions)} transactions:")
    for i, tx in enumerate(transactions):
        print(f"  {i+1}. {tx}")
        success = await cluster.propose_value(tx)
        print(f"     Result: {'SUCCESS' if success else 'FAILED'}")
        await asyncio.sleep(0.3)
    
    # Wait for Byzantine consensus
    await asyncio.sleep(2.0)
    
    # Check consensus
    consensus_state = cluster.get_consensus_state()
    print(f"\nByzantine consensus reached: {consensus_state['consensus']}")
    
    if consensus_state['consensus']:
        print(f"Agreed state: {consensus_state['consistent_state']}")
    
    await cluster.stop()

async def demo_network_partition():
    """Demo: Consensus behavior during network partition"""
    print("\nDEMO 3: NETWORK PARTITION TOLERANCE")
    print("=" * 50)
    
    cluster = ConsensusCluster("partition_demo", ConsensusType.RAFT, 5)
    await cluster.start()
    
    print("Started 5-node Raft cluster")
    
    # Wait for normal operation
    await asyncio.sleep(1.0)
    
    # Propose initial value
    print("\n1. Normal operation - proposing initial value:")
    await cluster.propose_value({"operation": "set", "key": "counter", "value": 0})
    await asyncio.sleep(1.0)
    
    state_before = cluster.get_consensus_state()
    print(f"   State before partition: {state_before['consistent_state']}")
    
    # Create network partition: 3 nodes vs 2 nodes
    print("\n2. Creating network partition (3 vs 2 nodes):")
    cluster.partition_network(["node_0", "node_1"], ["node_2", "node_3", "node_4"])
    
    # Try to propose value (should succeed in majority partition)
    print("   Proposing value in majority partition:")
    await cluster.propose_value({"operation": "set", "key": "counter", "value": 1})
    await asyncio.sleep(1.0)
    
    # Check state during partition
    state_during = cluster.get_cluster_state()
    print(f"   Cluster state during partition:")
    print(f"     Leader count: {state_during['leader_count']}")
    
    # Heal partition
    print("\n3. Healing network partition:")
    cluster.heal_network()
    await asyncio.sleep(2.0)  # Wait for reconciliation
    
    # Check final state
    state_after = cluster.get_consensus_state()
    print(f"   Final consensus: {state_after['consensus']}")
    print(f"   Final state: {state_after['consistent_state']}")
    
    await cluster.stop()

async def demo_concurrent_proposals():
    """Demo: Handling concurrent proposals"""
    print("\nDEMO 4: CONCURRENT PROPOSAL HANDLING")
    print("=" * 50)
    
    cluster = ConsensusCluster("concurrent_demo", ConsensusType.RAFT, 3)
    await cluster.start()
    
    print("Started 3-node Raft cluster for concurrent proposal testing")
    
    # Wait for leader election
    await asyncio.sleep(1.0)
    
    # Submit multiple concurrent proposals
    print("\nSubmitting 10 concurrent proposals:")
    
    async def submit_proposal(proposal_id: int):
        operation = {
            "operation": "increment", 
            "key": "concurrent_counter", 
            "proposal_id": proposal_id,
            "timestamp": time.time()
        }
        
        success = await cluster.propose_value(operation)
        print(f"  Proposal {proposal_id}: {'SUCCESS' if success else 'FAILED'}")
        return success
    
    # Create concurrent tasks
    tasks = [submit_proposal(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    # Wait for all proposals to be processed
    await asyncio.sleep(3.0)
    
    # Check final state
    consensus_state = cluster.get_consensus_state()
    print(f"\nFinal consensus: {consensus_state['consensus']}")
    
    if consensus_state['consensus']:
        state = consensus_state['consistent_state']
        print(f"Final state machine (showing last 5 entries):")
        items = list(state.items())[-5:]
        for key, value in items:
            print(f"  {key}: {value}")
    
    successful_proposals = sum(results)
    print(f"\nSuccessful concurrent proposals: {successful_proposals}/{len(results)}")
    
    await cluster.stop()

async def demo_consensus_comparison():
    """Demo: Compare different consensus algorithms"""
    print("\nDEMO 5: CONSENSUS ALGORITHM COMPARISON")
    print("=" * 50)
    
    # Test both Raft and PBFT
    algorithms = [
        (ConsensusType.RAFT, "Raft (CFT)", 5),
        (ConsensusType.PBFT, "PBFT (BFT)", 4)
    ]
    
    results = {}
    
    for algo_type, algo_name, cluster_size in algorithms:
        print(f"\nTesting {algo_name}:")
        
        cluster = ConsensusCluster(f"compare_{algo_type.value}", algo_type, cluster_size)
        await cluster.start()
        
        # Wait for initialization
        await asyncio.sleep(0.5)
        
        # Test proposal throughput
        start_time = time.time()
        successful_proposals = 0
        
        for i in range(5):
            operation = {"operation": "test", "sequence": i, "algorithm": algo_type.value}
            success = await cluster.propose_value(operation)
            if success:
                successful_proposals += 1
            await asyncio.sleep(0.1)  # Small delay between proposals
        
        # Wait for completion
        await asyncio.sleep(1.0)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Check consensus
        consensus_state = cluster.get_consensus_state()
        cluster_state = cluster.get_cluster_state()
        
        results[algo_name] = {
            'successful_proposals': successful_proposals,
            'total_time': duration,
            'throughput': successful_proposals / duration,
            'consensus_achieved': consensus_state['consensus'],
            'total_messages': sum(node.stats['messages_sent'] for node in cluster.nodes.values()),
            'cluster_stats': cluster_state['stats']
        }
        
        print(f"  Successful proposals: {successful_proposals}/5")
        print(f"  Time taken: {duration:.2f}s")
        print(f"  Consensus achieved: {consensus_state['consensus']}")
        
        await cluster.stop()
    
    # Comparison summary
    print(f"\nCOMPARISON SUMMARY:")
    print("=" * 30)
    
    for algo_name, metrics in results.items():
        print(f"\n{algo_name}:")
        print(f"  Throughput: {metrics['throughput']:.2f} proposals/sec")
        print(f"  Total messages: {metrics['total_messages']}")
        print(f"  Consensus: {'✅' if metrics['consensus_achieved'] else '❌'}")

async def main():
    """
    Demonstrate Distributed Consensus for agreement in unreliable networks
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement distributed consensus algorithms (Raft, PBFT)
    2. How to handle leader election and log replication
    3. How to achieve Byzantine fault tolerance
    4. How to deal with network partitions and node failures
    5. How to compare different consensus approaches
    
    REAL WORLD APPLICATIONS:
    =======================
    - Distributed databases and data consistency
    - Blockchain and cryptocurrency systems
    - Cloud infrastructure coordination
    - Microservices state management
    - Distributed file systems and storage
    - Mission-critical system coordination
    """
    
    print("DISTRIBUTED CONSENSUS DEMONSTRATION")
    print("Showing how agents agree on shared state in unreliable networks!")
    
    await demo_raft_consensus()
    await demo_pbft_consensus()
    await demo_network_partition()
    await demo_concurrent_proposals()
    await demo_consensus_comparison()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Consensus enables agreement despite failures")
    print("✓ Leader election provides ordering and coordination")
    print("✓ Log replication ensures consistent state across nodes")
    print("✓ Byzantine fault tolerance handles malicious behavior")
    print("✓ Network partitions require majority consensus")
    print("✓ Different algorithms trade-off performance vs fault tolerance")
    print("\nTHE POWER OF DISTRIBUTED CONSENSUS:")
    print("- Enables reliable distributed systems without single points of failure")
    print("- Provides foundation for blockchain and cryptocurrency systems")
    print("- Powers modern cloud infrastructure and microservices")
    print("- Ensures data consistency across global distributed databases")

if __name__ == "__main__":
    asyncio.run(main())
