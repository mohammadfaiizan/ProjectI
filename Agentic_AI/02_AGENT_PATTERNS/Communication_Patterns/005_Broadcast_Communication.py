#!/usr/bin/env python3
"""
Broadcast Communication: Efficient Multi-Agent Coordination
===========================================================

WHAT IS THE PROBLEM?
==================
Coordinating many agents simultaneously is challenging:
- Point-to-point messaging doesn't scale to hundreds of agents
- No efficient way to send same message to multiple recipients
- Difficult to maintain consistent state across all agents
- Manual recipient management becomes unwieldy
- Network traffic explodes with individual messages

Example: Emergency Alert Chaos
INDIVIDUAL MESSAGING (Inefficient):
- Emergency system sends 10,000 individual messages
- Each message requires separate network connection
- Takes 30 seconds to reach all devices
- High bandwidth usage and server load
- Some messages fail, creating inconsistent alerts

REAL WORLD EXAMPLE:
=================
How does your smartphone receive emergency alerts?

EMERGENCY BROADCAST SYSTEM:
When severe weather threatens:
1. COMPOSE: Emergency center creates single alert message
2. BROADCAST: Message sent to all cell towers simultaneously
3. AMPLIFY: Each tower broadcasts to all phones in area
4. RECEIVE: Millions of phones get alert within seconds
5. ACKNOWLEDGE: System tracks delivery statistics
6. REDUNDANCY: Multiple broadcast channels ensure reliability

BENEFITS:
- Reaches millions of devices in seconds
- Single message scales to unlimited recipients
- Efficient use of network infrastructure
- Guaranteed message consistency across recipients
- Works even during network congestion

THE ALGORITHM:
=============
1. COMPOSE: Create message for broadcast distribution
2. TARGET: Define recipient groups or broadcast zones
3. REPLICATE: Create copies for all delivery channels
4. TRANSMIT: Send simultaneously across all channels
5. AMPLIFY: Use hierarchical distribution for scale
6. CONFIRM: Track delivery success across recipients
7. RETRY: Handle failed deliveries with fallback methods

PATTERNS:
- Unreliable Broadcast: Best-effort delivery
- Reliable Broadcast: Guaranteed delivery confirmation
- Atomic Broadcast: All-or-nothing delivery guarantee
- Causal Broadcast: Maintains message ordering
- Hierarchical Broadcast: Tree-structured distribution

WHY IS THIS POWERFUL?
===================
- Scales to unlimited number of recipients
- Provides efficient network utilization
- Enables real-time coordination of agent swarms
- Supports emergency response and critical updates
- Powers distributed consensus and synchronization
- Enables massive multiplayer coordination
"""

import asyncio
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
import random
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BroadcastType(Enum):
    """Types of broadcast communication"""
    UNRELIABLE = "unreliable"           # Best-effort delivery
    RELIABLE = "reliable"               # Guaranteed delivery with ACK
    ATOMIC = "atomic"                   # All-or-nothing delivery
    CAUSAL = "causal"                   # Maintains causal ordering
    TOTAL_ORDER = "total_order"         # Global message ordering

class BroadcastScope(Enum):
    """Scope of broadcast distribution"""
    LOCAL = "local"                     # Local group/cluster
    REGIONAL = "regional"               # Regional distribution
    GLOBAL = "global"                   # Global broadcast
    ZONE_BASED = "zone_based"          # Geographic zones
    ROLE_BASED = "role_based"          # Role-specific groups

class DeliveryMode(Enum):
    """Message delivery modes"""
    IMMEDIATE = "immediate"             # Send immediately
    BATCH = "batch"                     # Batch multiple messages
    SCHEDULED = "scheduled"             # Send at specific time
    TRIGGERED = "triggered"             # Send on specific condition

@dataclass
class BroadcastMessage:
    """Message for broadcast communication"""
    id: str
    content: Dict[str, Any]
    sender_id: str
    
    # Broadcast properties
    broadcast_type: BroadcastType = BroadcastType.UNRELIABLE
    scope: BroadcastScope = BroadcastScope.GLOBAL
    delivery_mode: DeliveryMode = DeliveryMode.IMMEDIATE
    
    # Message metadata
    timestamp: float = field(default_factory=time.time)
    sequence_number: int = 0
    ttl: Optional[float] = None         # Time to live
    priority: int = 1                   # Higher = more important
    
    # Delivery tracking
    target_groups: List[str] = field(default_factory=list)
    target_zones: List[str] = field(default_factory=list)
    excluded_agents: Set[str] = field(default_factory=set)
    
    # Reliability tracking
    delivery_confirmations: Set[str] = field(default_factory=set)
    failed_deliveries: Set[str] = field(default_factory=set)
    retry_count: int = 0
    max_retries: int = 3
    
    # Causal ordering
    vector_clock: Dict[str, int] = field(default_factory=dict)
    causally_depends_on: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize message with defaults"""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry counter"""
        self.retry_count += 1
    
    def add_delivery_confirmation(self, agent_id: str) -> None:
        """Add delivery confirmation"""
        self.delivery_confirmations.add(agent_id)
        self.failed_deliveries.discard(agent_id)
    
    def add_delivery_failure(self, agent_id: str) -> None:
        """Add delivery failure"""
        self.failed_deliveries.add(agent_id)
    
    def get_delivery_ratio(self, total_recipients: int) -> float:
        """Get delivery success ratio"""
        if total_recipients == 0:
            return 1.0
        return len(self.delivery_confirmations) / total_recipients
    
    def serialize(self) -> str:
        """Serialize message to JSON"""
        data = asdict(self)
        data['broadcast_type'] = self.broadcast_type.value
        data['scope'] = self.scope.value
        data['delivery_mode'] = self.delivery_mode.value
        data['excluded_agents'] = list(self.excluded_agents)
        data['delivery_confirmations'] = list(self.delivery_confirmations)
        data['failed_deliveries'] = list(self.failed_deliveries)
        return json.dumps(data)
    
    @classmethod
    def deserialize(cls, data: str) -> 'BroadcastMessage':
        """Deserialize message from JSON"""
        obj = json.loads(data)
        obj['broadcast_type'] = BroadcastType(obj['broadcast_type'])
        obj['scope'] = BroadcastScope(obj['scope'])
        obj['delivery_mode'] = DeliveryMode(obj['delivery_mode'])
        obj['excluded_agents'] = set(obj['excluded_agents'])
        obj['delivery_confirmations'] = set(obj['delivery_confirmations'])
        obj['failed_deliveries'] = set(obj['failed_deliveries'])
        return cls(**obj)

class BroadcastReceiver(ABC):
    """Abstract base class for broadcast receivers"""
    
    @abstractmethod
    async def receive_broadcast(self, message: BroadcastMessage) -> bool:
        """Receive broadcast message. Return True if successfully processed."""
        pass
    
    @abstractmethod
    def get_agent_id(self) -> str:
        """Get unique agent identifier"""
        pass
    
    @abstractmethod
    def get_groups(self) -> List[str]:
        """Get groups this agent belongs to"""
        pass
    
    @abstractmethod
    def get_zone(self) -> str:
        """Get zone this agent belongs to"""
        pass

class AgentGroup:
    """Group of agents for targeted broadcasting"""
    
    def __init__(self, group_id: str, description: str = ""):
        self.group_id = group_id
        self.description = description
        self.members: Set[str] = set()
        self.created_at = time.time()
        self.message_count = 0
        self.active = True
    
    def add_member(self, agent_id: str) -> None:
        """Add agent to group"""
        self.members.add(agent_id)
    
    def remove_member(self, agent_id: str) -> None:
        """Remove agent from group"""
        self.members.discard(agent_id)
    
    def get_members(self) -> Set[str]:
        """Get all group members"""
        return self.members.copy()
    
    def get_size(self) -> int:
        """Get group size"""
        return len(self.members)
    
    def increment_message_count(self) -> None:
        """Increment message counter"""
        self.message_count += 1

class BroadcastZone:
    """Geographic or logical zone for broadcast distribution"""
    
    def __init__(self, zone_id: str, capacity: int = 1000):
        self.zone_id = zone_id
        self.capacity = capacity
        self.agents: Set[str] = set()
        self.sub_zones: List['BroadcastZone'] = []
        self.parent_zone: Optional['BroadcastZone'] = None
        self.created_at = time.time()
    
    def add_agent(self, agent_id: str) -> bool:
        """Add agent to zone"""
        if len(self.agents) < self.capacity:
            self.agents.add(agent_id)
            return True
        return False
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove agent from zone"""
        self.agents.discard(agent_id)
    
    def add_sub_zone(self, sub_zone: 'BroadcastZone') -> None:
        """Add sub-zone"""
        sub_zone.parent_zone = self
        self.sub_zones.append(sub_zone)
    
    def get_all_agents(self) -> Set[str]:
        """Get all agents in zone and sub-zones"""
        all_agents = self.agents.copy()
        for sub_zone in self.sub_zones:
            all_agents.update(sub_zone.get_all_agents())
        return all_agents
    
    def get_zone_path(self) -> str:
        """Get hierarchical zone path"""
        if self.parent_zone:
            return f"{self.parent_zone.get_zone_path()}.{self.zone_id}"
        return self.zone_id

class BroadcastDeliveryEngine:
    """Engine for delivering broadcast messages"""
    
    def __init__(self, max_concurrent_deliveries: int = 100):
        self.max_concurrent_deliveries = max_concurrent_deliveries
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_deliveries)
        
        # Delivery statistics
        self.delivery_stats = {
            'messages_delivered': 0,
            'delivery_failures': 0,
            'average_delivery_time': 0.0,
            'total_recipients': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def deliver_to_agents(self, message: BroadcastMessage, 
                               recipients: List[BroadcastReceiver]) -> Dict[str, bool]:
        """Deliver message to multiple agents concurrently"""
        
        if not recipients:
            return {}
        
        delivery_results = {}
        start_time = time.time()
        
        # Create delivery tasks
        tasks = []
        for recipient in recipients:
            if recipient.get_agent_id() not in message.excluded_agents:
                task = asyncio.create_task(
                    self._deliver_to_single_agent(message, recipient)
                )
                tasks.append((task, recipient.get_agent_id()))
        
        # Wait for all deliveries to complete
        for task, agent_id in tasks:
            try:
                success = await task
                delivery_results[agent_id] = success
                
                if success:
                    message.add_delivery_confirmation(agent_id)
                    self.delivery_stats['messages_delivered'] += 1
                else:
                    message.add_delivery_failure(agent_id)
                    self.delivery_stats['delivery_failures'] += 1
                
            except Exception as e:
                self.logger.error(f"Delivery error to {agent_id}: {e}")
                delivery_results[agent_id] = False
                message.add_delivery_failure(agent_id)
                self.delivery_stats['delivery_failures'] += 1
        
        # Update statistics
        delivery_time = time.time() - start_time
        recipient_count = len(recipients)
        
        self.delivery_stats['total_recipients'] += recipient_count
        
        if recipient_count > 0:
            avg_time_per_recipient = delivery_time / recipient_count
            current_avg = self.delivery_stats['average_delivery_time']
            total_msgs = self.delivery_stats['messages_delivered'] + self.delivery_stats['delivery_failures']
            
            # Running average
            if total_msgs > 0:
                self.delivery_stats['average_delivery_time'] = (
                    (current_avg * (total_msgs - recipient_count) + delivery_time) / total_msgs
                )
        
        return delivery_results
    
    async def _deliver_to_single_agent(self, message: BroadcastMessage, 
                                      recipient: BroadcastReceiver) -> bool:
        """Deliver message to single agent"""
        try:
            success = await recipient.receive_broadcast(message)
            return success
        except Exception as e:
            self.logger.error(f"Delivery error to {recipient.get_agent_id()}: {e}")
            return False
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """Get delivery statistics"""
        total_attempts = self.delivery_stats['messages_delivered'] + self.delivery_stats['delivery_failures']
        success_rate = (self.delivery_stats['messages_delivered'] / total_attempts 
                       if total_attempts > 0 else 0.0)
        
        return {
            **self.delivery_stats,
            'success_rate': success_rate,
            'total_delivery_attempts': total_attempts
        }

class BroadcastSystem:
    """
    Complete broadcast communication system
    
    EXAMPLE USAGE:
    =============
    # Create broadcast system
    broadcast = BroadcastSystem()
    await broadcast.start()
    
    # Register agents
    agent1 = MyBroadcastReceiver("agent1")
    broadcast.register_agent(agent1)
    
    # Create groups
    broadcast.create_group("emergency_responders")
    broadcast.add_agent_to_group("agent1", "emergency_responders")
    
    # Broadcast message
    message = BroadcastMessage(
        id="",
        content={"alert": "Emergency evacuation required"},
        sender_id="emergency_center",
        broadcast_type=BroadcastType.RELIABLE,
        target_groups=["emergency_responders"]
    )
    
    result = await broadcast.send_broadcast(message)
    """
    
    def __init__(self):
        self.agents: Dict[str, BroadcastReceiver] = {}
        self.groups: Dict[str, AgentGroup] = {}
        self.zones: Dict[str, BroadcastZone] = {}
        self.delivery_engine = BroadcastDeliveryEngine()
        
        # Message tracking
        self.message_history: List[BroadcastMessage] = []
        self.pending_reliable_messages: Dict[str, BroadcastMessage] = {}
        self.causal_order_buffer: Dict[str, List[BroadcastMessage]] = defaultdict(list)
        
        # System state
        self.running = False
        self.sequence_counter = 0
        self.vector_clock: Dict[str, int] = defaultdict(int)
        
        # Background workers
        self.retry_worker_active = False
        self.cleanup_worker_active = False
        
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
    
    async def start(self) -> None:
        """Start broadcast system"""
        self.running = True
        self.retry_worker_active = True
        self.cleanup_worker_active = True
        
        # Start background workers
        asyncio.create_task(self._retry_worker())
        asyncio.create_task(self._cleanup_worker())
        
        self.logger.info("Broadcast system started")
    
    async def stop(self) -> None:
        """Stop broadcast system"""
        self.running = False
        self.retry_worker_active = False
        self.cleanup_worker_active = False
        
        self.logger.info("Broadcast system stopped")
    
    def register_agent(self, agent: BroadcastReceiver) -> None:
        """Register agent for broadcast reception"""
        agent_id = agent.get_agent_id()
        self.agents[agent_id] = agent
        
        # Add to zone if specified
        zone_id = agent.get_zone()
        if zone_id and zone_id not in self.zones:
            self.create_zone(zone_id)
        
        if zone_id in self.zones:
            self.zones[zone_id].add_agent(agent_id)
        
        self.logger.info(f"Agent registered: {agent_id}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Remove from all groups
            for group in self.groups.values():
                group.remove_member(agent_id)
            
            # Remove from zone
            zone_id = agent.get_zone()
            if zone_id in self.zones:
                self.zones[zone_id].remove_agent(agent_id)
            
            del self.agents[agent_id]
            self.logger.info(f"Agent unregistered: {agent_id}")
    
    def create_group(self, group_id: str, description: str = "") -> AgentGroup:
        """Create agent group"""
        group = AgentGroup(group_id, description)
        self.groups[group_id] = group
        self.logger.info(f"Group created: {group_id}")
        return group
    
    def delete_group(self, group_id: str) -> bool:
        """Delete agent group"""
        if group_id in self.groups:
            del self.groups[group_id]
            self.logger.info(f"Group deleted: {group_id}")
            return True
        return False
    
    def add_agent_to_group(self, agent_id: str, group_id: str) -> bool:
        """Add agent to group"""
        if group_id in self.groups and agent_id in self.agents:
            self.groups[group_id].add_member(agent_id)
            return True
        return False
    
    def remove_agent_from_group(self, agent_id: str, group_id: str) -> bool:
        """Remove agent from group"""
        if group_id in self.groups:
            self.groups[group_id].remove_member(agent_id)
            return True
        return False
    
    def create_zone(self, zone_id: str, capacity: int = 1000) -> BroadcastZone:
        """Create broadcast zone"""
        zone = BroadcastZone(zone_id, capacity)
        self.zones[zone_id] = zone
        self.logger.info(f"Zone created: {zone_id}")
        return zone
    
    def create_hierarchical_zones(self, parent_zone_id: str, 
                                 sub_zone_ids: List[str]) -> None:
        """Create hierarchical zone structure"""
        if parent_zone_id not in self.zones:
            self.create_zone(parent_zone_id)
        
        parent_zone = self.zones[parent_zone_id]
        
        for sub_zone_id in sub_zone_ids:
            if sub_zone_id not in self.zones:
                self.create_zone(sub_zone_id)
            
            sub_zone = self.zones[sub_zone_id]
            parent_zone.add_sub_zone(sub_zone)
    
    async def send_broadcast(self, message: BroadcastMessage) -> Dict[str, Any]:
        """Send broadcast message to recipients"""
        
        # Assign sequence number
        with self.lock:
            self.sequence_counter += 1
            message.sequence_number = self.sequence_counter
        
        # Update vector clock for causal ordering
        if message.broadcast_type == BroadcastType.CAUSAL:
            self.vector_clock[message.sender_id] += 1
            message.vector_clock = self.vector_clock.copy()
        
        # Find recipients
        recipients = self._find_recipients(message)
        
        if not recipients:
            return {
                'message_id': message.id,
                'recipients_found': 0,
                'delivery_results': {},
                'success': False,
                'error': 'No recipients found'
            }
        
        # Handle different broadcast types
        if message.broadcast_type == BroadcastType.CAUSAL:
            return await self._send_causal_broadcast(message, recipients)
        elif message.broadcast_type == BroadcastType.ATOMIC:
            return await self._send_atomic_broadcast(message, recipients)
        else:
            return await self._send_standard_broadcast(message, recipients)
    
    async def _send_standard_broadcast(self, message: BroadcastMessage, 
                                     recipients: List[BroadcastReceiver]) -> Dict[str, Any]:
        """Send standard broadcast (unreliable or reliable)"""
        
        # Store message in history
        self.message_history.append(message)
        
        # Deliver to recipients
        delivery_results = await self.delivery_engine.deliver_to_agents(message, recipients)
        
        # Handle reliable broadcast
        if message.broadcast_type == BroadcastType.RELIABLE:
            # Track for retries
            self.pending_reliable_messages[message.id] = message
        
        success_count = sum(1 for success in delivery_results.values() if success)
        
        return {
            'message_id': message.id,
            'recipients_found': len(recipients),
            'successful_deliveries': success_count,
            'failed_deliveries': len(recipients) - success_count,
            'delivery_results': delivery_results,
            'success': success_count > 0,
            'delivery_ratio': success_count / len(recipients) if recipients else 0
        }
    
    async def _send_causal_broadcast(self, message: BroadcastMessage, 
                                   recipients: List[BroadcastReceiver]) -> Dict[str, Any]:
        """Send causal broadcast with ordering constraints"""
        
        # Check if all causally dependent messages have been delivered
        for dep_msg_id in message.causally_depends_on:
            # In a real implementation, you'd check if all recipients have received dependent messages
            pass
        
        # For simplicity, treat as standard broadcast
        return await self._send_standard_broadcast(message, recipients)
    
    async def _send_atomic_broadcast(self, message: BroadcastMessage, 
                                   recipients: List[BroadcastReceiver]) -> Dict[str, Any]:
        """Send atomic broadcast (all-or-nothing delivery)"""
        
        # Phase 1: Prepare phase (check if all agents can receive)
        prepare_results = await self._atomic_prepare_phase(message, recipients)
        
        if not all(prepare_results.values()):
            # Some agents can't receive - abort
            await self._atomic_abort_phase(message, recipients)
            return {
                'message_id': message.id,
                'recipients_found': len(recipients),
                'successful_deliveries': 0,
                'failed_deliveries': len(recipients),
                'delivery_results': prepare_results,
                'success': False,
                'error': 'Atomic broadcast failed in prepare phase'
            }
        
        # Phase 2: Commit phase (deliver to all agents)
        delivery_results = await self.delivery_engine.deliver_to_agents(message, recipients)
        
        all_successful = all(delivery_results.values())
        
        if not all_successful:
            # Some deliveries failed - this shouldn't happen after successful prepare
            self.logger.error(f"Atomic broadcast inconsistency for message {message.id}")
        
        return {
            'message_id': message.id,
            'recipients_found': len(recipients),
            'successful_deliveries': sum(1 for success in delivery_results.values() if success),
            'failed_deliveries': sum(1 for success in delivery_results.values() if not success),
            'delivery_results': delivery_results,
            'success': all_successful,
            'atomic_guarantee': all_successful
        }
    
    async def _atomic_prepare_phase(self, message: BroadcastMessage, 
                                  recipients: List[BroadcastReceiver]) -> Dict[str, bool]:
        """Atomic broadcast prepare phase"""
        # Simplified prepare phase - in reality, this would involve
        # checking if agents have resources and can commit to receiving
        prepare_results = {}
        
        for recipient in recipients:
            # Simulate prepare check (could fail with small probability)
            can_receive = random.random() > 0.05  # 95% success rate
            prepare_results[recipient.get_agent_id()] = can_receive
        
        return prepare_results
    
    async def _atomic_abort_phase(self, message: BroadcastMessage, 
                                recipients: List[BroadcastReceiver]) -> None:
        """Atomic broadcast abort phase"""
        # Notify all agents that the broadcast is aborted
        # In a real implementation, this would clean up any prepared state
        self.logger.info(f"Atomic broadcast aborted for message {message.id}")
    
    def _find_recipients(self, message: BroadcastMessage) -> List[BroadcastReceiver]:
        """Find recipients based on message targeting"""
        recipients = []
        
        if message.scope == BroadcastScope.GLOBAL and not message.target_groups and not message.target_zones:
            # Broadcast to all agents
            recipients = list(self.agents.values())
        
        elif message.target_groups:
            # Broadcast to specific groups
            for group_id in message.target_groups:
                if group_id in self.groups:
                    group = self.groups[group_id]
                    for agent_id in group.get_members():
                        if agent_id in self.agents:
                            recipients.append(self.agents[agent_id])
        
        elif message.target_zones:
            # Broadcast to specific zones
            for zone_id in message.target_zones:
                if zone_id in self.zones:
                    zone = self.zones[zone_id]
                    for agent_id in zone.get_all_agents():
                        if agent_id in self.agents:
                            recipients.append(self.agents[agent_id])
        
        elif message.scope == BroadcastScope.ROLE_BASED:
            # Find agents by role (group membership)
            for agent in self.agents.values():
                agent_groups = agent.get_groups()
                if any(group in message.target_groups for group in agent_groups):
                    recipients.append(agent)
        
        # Remove duplicates and excluded agents
        unique_recipients = []
        seen_agents = set()
        
        for recipient in recipients:
            agent_id = recipient.get_agent_id()
            if (agent_id not in seen_agents and 
                agent_id not in message.excluded_agents):
                unique_recipients.append(recipient)
                seen_agents.add(agent_id)
        
        return unique_recipients
    
    async def _retry_worker(self) -> None:
        """Background worker for retrying failed reliable broadcasts"""
        while self.retry_worker_active:
            try:
                retry_messages = []
                
                # Find messages that need retry
                for message_id, message in list(self.pending_reliable_messages.items()):
                    if message.is_expired():
                        # Message expired - remove from pending
                        del self.pending_reliable_messages[message_id]
                        continue
                    
                    if message.failed_deliveries and message.can_retry():
                        retry_messages.append(message)
                
                # Retry failed deliveries
                for message in retry_messages:
                    message.increment_retry()
                    
                    # Find failed recipients
                    failed_recipients = []
                    for agent_id in message.failed_deliveries:
                        if agent_id in self.agents:
                            failed_recipients.append(self.agents[agent_id])
                    
                    if failed_recipients:
                        self.logger.info(f"Retrying broadcast {message.id} to {len(failed_recipients)} agents")
                        
                        # Clear failed deliveries for retry
                        message.failed_deliveries.clear()
                        
                        # Retry delivery
                        delivery_results = await self.delivery_engine.deliver_to_agents(
                            message, failed_recipients
                        )
                        
                        # Check if all retries succeeded
                        if all(delivery_results.values()):
                            # All successful - remove from pending
                            self.pending_reliable_messages.pop(message.id, None)
                
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Retry worker error: {e}")
                await asyncio.sleep(5.0)
    
    async def _cleanup_worker(self) -> None:
        """Background worker for cleaning up old messages"""
        while self.cleanup_worker_active:
            try:
                current_time = time.time()
                max_history_age = 3600  # 1 hour
                
                # Clean up old message history
                self.message_history = [
                    msg for msg in self.message_history
                    if current_time - msg.timestamp < max_history_age
                ]
                
                # Clean up expired pending messages
                expired_messages = []
                for message_id, message in self.pending_reliable_messages.items():
                    if message.is_expired():
                        expired_messages.append(message_id)
                
                for message_id in expired_messages:
                    del self.pending_reliable_messages[message_id]
                    self.logger.warning(f"Reliable broadcast {message_id} expired")
                
                await asyncio.sleep(300.0)  # Cleanup every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cleanup worker error: {e}")
                await asyncio.sleep(300.0)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'agents_registered': len(self.agents),
            'groups_created': len(self.groups),
            'zones_created': len(self.zones),
            'messages_in_history': len(self.message_history),
            'pending_reliable_messages': len(self.pending_reliable_messages),
            'delivery_engine_stats': self.delivery_engine.get_delivery_stats(),
            'sequence_counter': self.sequence_counter,
            'group_details': {
                group_id: {
                    'members': len(group.members),
                    'message_count': group.message_count,
                    'active': group.active
                }
                for group_id, group in self.groups.items()
            },
            'zone_details': {
                zone_id: {
                    'agents': len(zone.agents),
                    'capacity': zone.capacity,
                    'sub_zones': len(zone.sub_zones)
                }
                for zone_id, zone in self.zones.items()
            }
        }

# Example broadcast receivers
class AlertReceiver(BroadcastReceiver):
    """Receiver for emergency alerts"""
    
    def __init__(self, agent_id: str, zone: str = "default", groups: List[str] = None):
        self.agent_id = agent_id
        self.zone = zone
        self.groups = groups or []
        self.alerts_received: List[Dict[str, Any]] = []
        self.active = True
    
    async def receive_broadcast(self, message: BroadcastMessage) -> bool:
        """Receive and process alert broadcast"""
        try:
            if not self.active:
                return False
            
            alert_data = {
                'message_id': message.id,
                'content': message.content,
                'sender': message.sender_id,
                'timestamp': message.timestamp,
                'priority': message.priority,
                'received_at': time.time()
            }
            
            self.alerts_received.append(alert_data)
            
            # Simulate processing delay
            await asyncio.sleep(0.01)
            
            alert_type = message.content.get('type', 'info')
            alert_msg = message.content.get('message', 'No message')
            
            print(f"🚨 {self.agent_id} received {alert_type.upper()}: {alert_msg}")
            
            return True
            
        except Exception as e:
            print(f"Error processing alert in {self.agent_id}: {e}")
            return False
    
    def get_agent_id(self) -> str:
        return self.agent_id
    
    def get_groups(self) -> List[str]:
        return self.groups.copy()
    
    def get_zone(self) -> str:
        return self.zone

class NewsReceiver(BroadcastReceiver):
    """Receiver for news broadcasts"""
    
    def __init__(self, agent_id: str, interests: List[str] = None):
        self.agent_id = agent_id
        self.interests = interests or []
        self.news_received: List[Dict[str, Any]] = []
        self.zone = "global"
        self.groups = ["news_subscribers"]
    
    async def receive_broadcast(self, message: BroadcastMessage) -> bool:
        """Receive and process news broadcast"""
        try:
            news_data = {
                'message_id': message.id,
                'headline': message.content.get('headline', 'No headline'),
                'category': message.content.get('category', 'general'),
                'content': message.content.get('content', ''),
                'sender': message.sender_id,
                'timestamp': message.timestamp
            }
            
            # Filter by interests
            if self.interests:
                category = news_data['category']
                if category not in self.interests:
                    return True  # Successfully ignored
            
            self.news_received.append(news_data)
            
            print(f"📰 {self.agent_id} received news: {news_data['headline']} [{news_data['category']}]")
            
            return True
            
        except Exception as e:
            print(f"Error processing news in {self.agent_id}: {e}")
            return False
    
    def get_agent_id(self) -> str:
        return self.agent_id
    
    def get_groups(self) -> List[str]:
        return self.groups
    
    def get_zone(self) -> str:
        return self.zone

class SystemMonitor(BroadcastReceiver):
    """Receiver for system monitoring broadcasts"""
    
    def __init__(self, agent_id: str, zone: str = "datacenter"):
        self.agent_id = agent_id
        self.zone = zone
        self.groups = ["system_monitors", "ops_team"]
        self.alerts_processed = 0
        self.critical_alerts = 0
    
    async def receive_broadcast(self, message: BroadcastMessage) -> bool:
        """Receive and process monitoring broadcast"""
        try:
            alert_level = message.content.get('level', 'info')
            alert_message = message.content.get('alert', 'System event')
            source = message.content.get('source', 'unknown')
            
            self.alerts_processed += 1
            
            if alert_level == 'critical':
                self.critical_alerts += 1
                print(f"🔥 {self.agent_id} - CRITICAL ALERT from {source}: {alert_message}")
            elif alert_level == 'warning':
                print(f"⚠️  {self.agent_id} - WARNING from {source}: {alert_message}")
            else:
                print(f"ℹ️  {self.agent_id} - INFO from {source}: {alert_message}")
            
            return True
            
        except Exception as e:
            print(f"Error processing monitor alert in {self.agent_id}: {e}")
            return False
    
    def get_agent_id(self) -> str:
        return self.agent_id
    
    def get_groups(self) -> List[str]:
        return self.groups
    
    def get_zone(self) -> str:
        return self.zone

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_broadcasting():
    """Demo: Basic broadcast to all agents"""
    print("\nDEMO 1: BASIC BROADCAST COMMUNICATION")
    print("=" * 50)
    
    # Create broadcast system
    broadcast = BroadcastSystem()
    await broadcast.start()
    
    # Register multiple agents
    agents = []
    for i in range(5):
        agent = AlertReceiver(f"agent_{i}", zone="office", groups=["employees"])
        agents.append(agent)
        broadcast.register_agent(agent)
    
    # Send global broadcast
    message = BroadcastMessage(
        id="",
        content={
            "type": "announcement", 
            "message": "All-hands meeting at 3 PM",
            "priority": "normal"
        },
        sender_id="hr_department",
        broadcast_type=BroadcastType.UNRELIABLE,
        scope=BroadcastScope.GLOBAL
    )
    
    print(f"Broadcasting to {len(agents)} agents...")
    result = await broadcast.send_broadcast(message)
    
    print(f"Broadcast result:")
    print(f"  - Recipients found: {result['recipients_found']}")
    print(f"  - Successful deliveries: {result['successful_deliveries']}")
    print(f"  - Failed deliveries: {result['failed_deliveries']}")
    print(f"  - Delivery ratio: {result['delivery_ratio']:.1%}")
    
    await broadcast.stop()

async def demo_group_broadcasting():
    """Demo: Group-based targeted broadcasting"""
    print("\nDEMO 2: GROUP-BASED BROADCASTING")
    print("=" * 50)
    
    broadcast = BroadcastSystem()
    await broadcast.start()
    
    # Create groups
    broadcast.create_group("emergency_responders", "First responders and medical team")
    broadcast.create_group("security_team", "Security and safety personnel")
    broadcast.create_group("executives", "Management and executive team")
    
    # Register agents with different groups
    agents_data = [
        ("firefighter_1", "datacenter_a", ["emergency_responders"]),
        ("firefighter_2", "datacenter_a", ["emergency_responders"]),
        ("medic_1", "datacenter_a", ["emergency_responders"]),
        ("security_1", "datacenter_b", ["security_team"]),
        ("security_2", "datacenter_b", ["security_team"]),
        ("ceo", "headquarters", ["executives"]),
        ("cto", "headquarters", ["executives"]),
        ("regular_employee", "office", [])
    ]
    
    for agent_id, zone, groups in agents_data:
        agent = AlertReceiver(agent_id, zone, groups)
        broadcast.register_agent(agent)
        
        # Add to groups
        for group in groups:
            broadcast.add_agent_to_group(agent_id, group)
    
    # Send targeted broadcasts to different groups
    broadcasts = [
        {
            "target_groups": ["emergency_responders"],
            "content": {"type": "emergency", "message": "Fire detected in server room A", "action": "evacuate"},
            "sender": "fire_detection_system"
        },
        {
            "target_groups": ["security_team"],
            "content": {"type": "security", "message": "Unauthorized access attempt detected", "action": "investigate"},
            "sender": "security_system"
        },
        {
            "target_groups": ["executives"],
            "content": {"type": "business", "message": "Quarterly board meeting moved to Friday", "action": "reschedule"},
            "sender": "board_secretary"
        }
    ]
    
    for broadcast_data in broadcasts:
        message = BroadcastMessage(
            id="",
            content=broadcast_data["content"],
            sender_id=broadcast_data["sender"],
            broadcast_type=BroadcastType.RELIABLE,
            target_groups=broadcast_data["target_groups"]
        )
        
        print(f"\nBroadcasting to groups: {broadcast_data['target_groups']}")
        print(f"Message: {broadcast_data['content']['message']}")
        
        result = await broadcast.send_broadcast(message)
        print(f"  - Delivered to {result['successful_deliveries']} agents")
    
    await broadcast.stop()

async def demo_zone_broadcasting():
    """Demo: Zone-based geographic broadcasting"""
    print("\nDEMO 3: ZONE-BASED BROADCASTING")
    print("=" * 50)
    
    broadcast = BroadcastSystem()
    await broadcast.start()
    
    # Create hierarchical zone structure
    broadcast.create_zone("global", capacity=10000)
    broadcast.create_zone("north_america", capacity=5000)
    broadcast.create_zone("europe", capacity=3000)
    broadcast.create_zone("usa", capacity=2000)
    broadcast.create_zone("canada", capacity=1000)
    broadcast.create_zone("uk", capacity=1500)
    broadcast.create_zone("germany", capacity=1000)
    
    # Create hierarchy
    broadcast.create_hierarchical_zones("global", ["north_america", "europe"])
    broadcast.create_hierarchical_zones("north_america", ["usa", "canada"])
    broadcast.create_hierarchical_zones("europe", ["uk", "germany"])
    
    # Register agents in different zones
    zone_agents = [
        ("weather_usa_1", "usa", ["weather_services"]),
        ("weather_usa_2", "usa", ["weather_services"]),
        ("weather_canada_1", "canada", ["weather_services"]),
        ("weather_uk_1", "uk", ["weather_services"]),
        ("weather_uk_2", "uk", ["weather_services"]),
        ("weather_germany_1", "germany", ["weather_services"])
    ]
    
    for agent_id, zone, groups in zone_agents:
        agent = AlertReceiver(agent_id, zone, groups)
        broadcast.register_agent(agent)
    
    # Send zone-specific broadcasts
    zone_broadcasts = [
        {
            "zones": ["usa"],
            "content": {"type": "weather", "message": "Hurricane warning for East Coast", "severity": "high"},
            "sender": "noaa_weather"
        },
        {
            "zones": ["uk"],
            "content": {"type": "weather", "message": "Heavy rain and flooding expected", "severity": "medium"},
            "sender": "met_office"
        },
        {
            "zones": ["north_america"],
            "content": {"type": "system", "message": "Scheduled maintenance on regional servers", "severity": "low"},
            "sender": "ops_team"
        }
    ]
    
    for broadcast_data in zone_broadcasts:
        message = BroadcastMessage(
            id="",
            content=broadcast_data["content"],
            sender_id=broadcast_data["sender"],
            broadcast_type=BroadcastType.RELIABLE,
            target_zones=broadcast_data["zones"]
        )
        
        print(f"\nBroadcasting to zones: {broadcast_data['zones']}")
        print(f"Message: {broadcast_data['content']['message']}")
        
        result = await broadcast.send_broadcast(message)
        print(f"  - Delivered to {result['successful_deliveries']} agents in zone(s)")
    
    await broadcast.stop()

async def demo_reliable_broadcasting():
    """Demo: Reliable broadcast with retry mechanisms"""
    print("\nDEMO 4: RELIABLE BROADCASTING")
    print("=" * 50)
    
    broadcast = BroadcastSystem()
    await broadcast.start()
    
    # Create group
    broadcast.create_group("critical_systems")
    
    # Register agents (some will simulate failures)
    class UnreliableReceiver(AlertReceiver):
        def __init__(self, agent_id: str, failure_rate: float = 0.3):
            super().__init__(agent_id, "datacenter", ["critical_systems"])
            self.failure_rate = failure_rate
            self.attempt_count = 0
        
        async def receive_broadcast(self, message: BroadcastMessage) -> bool:
            self.attempt_count += 1
            
            # Simulate intermittent failures
            if random.random() < self.failure_rate:
                print(f"❌ {self.agent_id} failed to receive message (attempt {self.attempt_count})")
                return False
            else:
                print(f"✅ {self.agent_id} successfully received message (attempt {self.attempt_count})")
                return await super().receive_broadcast(message)
    
    # Register mix of reliable and unreliable agents
    agents = []
    for i in range(5):
        if i < 2:
            agent = AlertReceiver(f"reliable_agent_{i}", "datacenter", ["critical_systems"])
        else:
            agent = UnreliableReceiver(f"unreliable_agent_{i}", failure_rate=0.4)
        
        agents.append(agent)
        broadcast.register_agent(agent)
        broadcast.add_agent_to_group(agent.get_agent_id(), "critical_systems")
    
    # Send critical reliable broadcast
    message = BroadcastMessage(
        id="",
        content={
            "type": "critical", 
            "message": "Database backup completed successfully",
            "system": "backup_service"
        },
        sender_id="backup_system",
        broadcast_type=BroadcastType.RELIABLE,
        target_groups=["critical_systems"],
        max_retries=3,
        ttl=30.0  # 30 second timeout
    )
    
    print("Sending reliable broadcast to critical systems...")
    print("Some agents may fail initially but will be retried automatically")
    
    result = await broadcast.send_broadcast(message)
    
    print(f"\nInitial broadcast result:")
    print(f"  - Recipients: {result['recipients_found']}")
    print(f"  - Initial successful deliveries: {result['successful_deliveries']}")
    print(f"  - Initial failed deliveries: {result['failed_deliveries']}")
    
    # Wait for retry mechanism to work
    print("\nWaiting for automatic retries...")
    await asyncio.sleep(8.0)
    
    stats = broadcast.get_system_statistics()
    print(f"Final delivery stats: {stats['delivery_engine_stats']}")
    
    await broadcast.stop()

async def demo_atomic_broadcasting():
    """Demo: Atomic broadcast (all-or-nothing delivery)"""
    print("\nDEMO 5: ATOMIC BROADCASTING")
    print("=" * 50)
    
    broadcast = BroadcastSystem()
    await broadcast.start()
    
    # Create group for distributed transaction
    broadcast.create_group("database_nodes")
    
    # Register database nodes
    nodes = []
    for i in range(4):
        node = AlertReceiver(f"db_node_{i}", "cluster", ["database_nodes"])
        nodes.append(node)
        broadcast.register_agent(node)
        broadcast.add_agent_to_group(node.get_agent_id(), "database_nodes")
    
    # Test atomic broadcasts
    atomic_tests = [
        {
            "description": "Successful atomic transaction",
            "content": {"transaction": "commit", "tx_id": "TX_001", "operation": "transfer_funds"}
        },
        {
            "description": "Failed atomic transaction", 
            "content": {"transaction": "rollback", "tx_id": "TX_002", "operation": "update_balance"}
        }
    ]
    
    for test in atomic_tests:
        print(f"\nTesting: {test['description']}")
        
        message = BroadcastMessage(
            id="",
            content=test["content"],
            sender_id="transaction_coordinator",
            broadcast_type=BroadcastType.ATOMIC,
            target_groups=["database_nodes"]
        )
        
        result = await broadcast.send_broadcast(message)
        
        print(f"  - Atomic guarantee: {result.get('atomic_guarantee', False)}")
        print(f"  - Successful deliveries: {result['successful_deliveries']}")
        print(f"  - Failed deliveries: {result['failed_deliveries']}")
        
        if result.get('atomic_guarantee'):
            print("  ✅ All nodes received transaction - consistency maintained")
        else:
            print("  ❌ Atomic broadcast failed - transaction aborted")
    
    await broadcast.stop()

async def main():
    """
    Demonstrate Broadcast Communication for efficient multi-agent coordination
    
    WHAT YOU'LL LEARN:
    ================
    1. How to design scalable broadcast communication systems
    2. How to implement different broadcast reliability guarantees
    3. How to organize agents into groups and zones for targeting
    4. How to handle large-scale message distribution efficiently
    5. How to ensure consistency with atomic and causal broadcasts
    
    REAL WORLD APPLICATIONS:
    =======================
    - Emergency alert and notification systems
    - Distributed system coordination and consensus
    - News and content distribution networks
    - IoT device management and control systems
    - Financial market data distribution
    - Gaming and virtual world synchronization
    """
    
    print("BROADCAST COMMUNICATION DEMONSTRATION")
    print("Showing how agents coordinate through efficient broadcasting!")
    
    await demo_basic_broadcasting()
    await demo_group_broadcasting()
    await demo_zone_broadcasting()
    await demo_reliable_broadcasting()
    await demo_atomic_broadcasting()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Broadcast enables efficient one-to-many communication")
    print("✓ Group and zone targeting allows selective message delivery")
    print("✓ Reliable broadcasts ensure critical messages reach all recipients")
    print("✓ Atomic broadcasts maintain consistency across distributed systems")
    print("✓ Hierarchical distribution scales to massive agent populations")
    print("✓ Retry mechanisms handle temporary failures gracefully")
    print("\nTHE POWER OF BROADCAST COMMUNICATION:")
    print("- Scales to millions of recipients with minimal latency")
    print("- Enables real-time coordination of massive agent swarms")
    print("- Provides consistency guarantees for critical operations")
    print("- Supports emergency response and crisis management")

if __name__ == "__main__":
    asyncio.run(main())
