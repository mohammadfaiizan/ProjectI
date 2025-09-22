#!/usr/bin/env python3
"""
Message Passing Systems: Reliable Agent Communication
====================================================

WHAT IS THE PROBLEM?
==================
Agents need to communicate reliably and efficiently, but traditional function calls don't work for:
- Distributed agents across networks
- Asynchronous communication
- Message ordering and delivery guarantees
- Fault tolerance and error recovery
- Load balancing and scalability

Example: Coordination Chaos
TRADITIONAL APPROACH (Fails):
- Agent A calls agent B's function directly
- Network failure breaks the connection
- No way to retry or recover
- Blocking calls prevent other work
- No message history or audit trail

REAL WORLD EXAMPLE:
=================
How do modern messaging systems like Slack or WhatsApp work?

SLACK MESSAGING SYSTEM:
Each message goes through:
1. SENDER: Types message and hits send
2. MESSAGE QUEUE: Message stored reliably
3. DELIVERY: System ensures message reaches recipient
4. ACKNOWLEDGMENT: Sender gets delivery confirmation
5. ORDERING: Messages arrive in correct order
6. PERSISTENCE: Messages stored for later retrieval

BENEFITS:
- Messages never get lost
- Works even when recipient is offline
- Handles millions of messages per second
- Provides delivery guarantees
- Scales globally across data centers

THE ALGORITHM:
=============
1. SEND: Agent packages message with metadata
2. ROUTE: System determines optimal delivery path
3. QUEUE: Message stored reliably until delivery
4. DELIVER: Message sent to recipient agent
5. ACKNOWLEDGE: Recipient confirms receipt
6. RETRY: System retries failed deliveries
7. ORDER: Messages delivered in correct sequence

MESSAGE TYPES:
- Point-to-point: Direct agent-to-agent
- Multicast: One sender, multiple recipients
- Broadcast: Message to all agents
- Request-Reply: Synchronous communication pattern

WHY IS THIS POWERFUL?
===================
- Enables reliable distributed agent systems
- Provides fault tolerance and error recovery
- Supports asynchronous, non-blocking communication
- Scales to thousands of agents
- Guarantees message delivery and ordering
- Enables complex multi-agent workflows
"""

import asyncio
import time
import json
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Callable, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import heapq
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class MessageType(Enum):
    """Types of messages in the system"""
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ACKNOWLEDGMENT = "acknowledgment"
    ERROR = "error"
    BROADCAST = "broadcast"

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class DeliveryGuarantee(Enum):
    """Message delivery guarantees"""
    AT_MOST_ONCE = "at_most_once"      # May lose messages, never duplicate
    AT_LEAST_ONCE = "at_least_once"    # Never lose, may duplicate
    EXACTLY_ONCE = "exactly_once"      # Never lose, never duplicate

@dataclass
class Message:
    """Core message structure for agent communication"""
    id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    correlation_id: Optional[str] = None  # For request-response pairing
    reply_to: Optional[str] = None        # Return address
    
    # Routing and delivery
    routing_key: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    ttl: Optional[float] = None           # Time to live
    
    # Tracking
    attempt_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    
    def __post_init__(self):
        """Initialize message with defaults"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.routing_key:
            self.routing_key = f"{self.sender_id}.{self.recipient_id}"
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.attempt_count < self.max_retries
    
    def increment_attempt(self) -> None:
        """Increment retry attempt counter"""
        self.attempt_count += 1
    
    def serialize(self) -> str:
        """Serialize message to JSON"""
        data = asdict(self)
        # Convert enums to strings
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        data['delivery_guarantee'] = self.delivery_guarantee.value
        return json.dumps(data)
    
    @classmethod
    def deserialize(cls, data: str) -> 'Message':
        """Deserialize message from JSON"""
        obj = json.loads(data)
        # Convert strings back to enums
        obj['message_type'] = MessageType(obj['message_type'])
        obj['priority'] = MessagePriority(obj['priority'])
        obj['delivery_guarantee'] = DeliveryGuarantee(obj['delivery_guarantee'])
        return cls(**obj)

class MessageHandler(ABC):
    """Abstract base class for message handlers"""
    
    @abstractmethod
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming message and optionally return response"""
        pass
    
    @abstractmethod
    def can_handle(self, message: Message) -> bool:
        """Check if this handler can process the message"""
        pass

class MessageRouter:
    """Routes messages to appropriate handlers"""
    
    def __init__(self):
        self.handlers: List[MessageHandler] = []
        self.routing_table: Dict[str, str] = {}  # routing_key -> agent_id
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)  # topic -> agent_ids
    
    def add_handler(self, handler: MessageHandler) -> None:
        """Add message handler"""
        self.handlers.append(handler)
    
    def register_agent(self, agent_id: str, routing_keys: List[str]) -> None:
        """Register agent for specific routing keys"""
        for key in routing_keys:
            self.routing_table[key] = agent_id
    
    def subscribe_to_topic(self, agent_id: str, topic: str) -> None:
        """Subscribe agent to topic for broadcast messages"""
        self.topic_subscribers[topic].add(agent_id)
    
    def unsubscribe_from_topic(self, agent_id: str, topic: str) -> None:
        """Unsubscribe agent from topic"""
        self.topic_subscribers[topic].discard(agent_id)
    
    async def route_message(self, message: Message) -> List[str]:
        """Route message and return list of target agent IDs"""
        targets = []
        
        if message.message_type == MessageType.BROADCAST:
            # Broadcast to all subscribers of the topic
            topic = message.headers.get('topic', 'default')
            targets.extend(self.topic_subscribers[topic])
        else:
            # Route to specific recipient
            if message.recipient_id:
                targets.append(message.recipient_id)
            elif message.routing_key in self.routing_table:
                targets.append(self.routing_table[message.routing_key])
        
        return targets
    
    async def find_handler(self, message: Message) -> Optional[MessageHandler]:
        """Find appropriate handler for message"""
        for handler in self.handlers:
            if handler.can_handle(message):
                return handler
        return None

class MessageQueue:
    """Priority queue for message storage and delivery"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queue: List[Tuple[int, float, Message]] = []  # (priority, timestamp, message)
        self.message_index: Dict[str, Message] = {}
        self.size = 0
        self.lock = threading.Lock()
    
    def put(self, message: Message) -> bool:
        """Add message to queue"""
        with self.lock:
            if self.size >= self.max_size:
                return False
            
            # Priority is negative for max-heap behavior
            priority = -message.priority.value
            timestamp = message.timestamp
            
            heapq.heappush(self.queue, (priority, timestamp, message))
            self.message_index[message.id] = message
            self.size += 1
            return True
    
    def get(self) -> Optional[Message]:
        """Get highest priority message"""
        with self.lock:
            while self.queue:
                priority, timestamp, message = heapq.heappop(self.queue)
                
                # Check if message is still valid
                if message.id in self.message_index:
                    del self.message_index[message.id]
                    self.size -= 1
                    
                    # Check if expired
                    if message.is_expired():
                        continue
                    
                    return message
            
            return None
    
    def remove(self, message_id: str) -> bool:
        """Remove message by ID"""
        with self.lock:
            if message_id in self.message_index:
                del self.message_index[message_id]
                self.size -= 1
                return True
            return False
    
    def peek(self) -> Optional[Message]:
        """Peek at next message without removing"""
        with self.lock:
            while self.queue:
                priority, timestamp, message = self.queue[0]
                
                if message.id in self.message_index:
                    if message.is_expired():
                        self.get()  # Remove expired message
                        continue
                    return message
                else:
                    heapq.heappop(self.queue)  # Remove invalid reference
            
            return None
    
    def get_size(self) -> int:
        """Get current queue size"""
        return self.size
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.size == 0

class MessageStore:
    """Persistent message storage for reliability"""
    
    def __init__(self):
        self.messages: Dict[str, Message] = {}
        self.agent_messages: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> message_ids
        self.pending_acks: Dict[str, Message] = {}  # message_id -> message
        self.delivery_receipts: Dict[str, Dict[str, Any]] = {}  # message_id -> receipt info
    
    def store_message(self, message: Message) -> None:
        """Store message persistently"""
        self.messages[message.id] = message
        self.agent_messages[message.sender_id].add(message.id)
        self.agent_messages[message.recipient_id].add(message.id)
        
        # Mark as pending acknowledgment if required
        if message.delivery_guarantee in [DeliveryGuarantee.AT_LEAST_ONCE, DeliveryGuarantee.EXACTLY_ONCE]:
            self.pending_acks[message.id] = message
    
    def get_message(self, message_id: str) -> Optional[Message]:
        """Retrieve message by ID"""
        return self.messages.get(message_id)
    
    def acknowledge_delivery(self, message_id: str, agent_id: str) -> bool:
        """Acknowledge message delivery"""
        if message_id in self.pending_acks:
            self.delivery_receipts[message_id] = {
                'acknowledged_by': agent_id,
                'timestamp': time.time()
            }
            del self.pending_acks[message_id]
            return True
        return False
    
    def get_pending_acks(self) -> List[Message]:
        """Get messages pending acknowledgment"""
        return list(self.pending_acks.values())
    
    def get_agent_messages(self, agent_id: str, limit: int = 100) -> List[Message]:
        """Get messages for specific agent"""
        message_ids = list(self.agent_messages[agent_id])[-limit:]
        return [self.messages[msg_id] for msg_id in message_ids if msg_id in self.messages]
    
    def cleanup_expired(self) -> int:
        """Remove expired messages and return count"""
        expired_ids = []
        for message_id, message in self.messages.items():
            if message.is_expired():
                expired_ids.append(message_id)
        
        for message_id in expired_ids:
            message = self.messages[message_id]
            del self.messages[message_id]
            self.agent_messages[message.sender_id].discard(message_id)
            self.agent_messages[message.recipient_id].discard(message_id)
            self.pending_acks.pop(message_id, None)
            self.delivery_receipts.pop(message_id, None)
        
        return len(expired_ids)

class MessagePassingSystem:
    """
    Complete message passing system for agent communication
    
    EXAMPLE USAGE:
    =============
    # Create messaging system
    messaging = MessagePassingSystem()
    
    # Register agents
    await messaging.register_agent("agent1")
    await messaging.register_agent("agent2")
    
    # Send message
    message = Message(
        id="",
        sender_id="agent1",
        recipient_id="agent2",
        message_type=MessageType.COMMAND,
        payload={"action": "process_data", "data": [1, 2, 3]}
    )
    
    success = await messaging.send_message(message)
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.router = MessageRouter()
        self.store = MessageStore()
        self.agents: Dict[str, MessageQueue] = {}
        self.handlers: Dict[str, List[MessageHandler]] = defaultdict(list)
        
        # System state
        self.running = False
        self.worker_count = 4
        self.executor = ThreadPoolExecutor(max_workers=self.worker_count)
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'messages_retried': 0
        }
        
        # Configuration
        self.max_queue_size = max_queue_size
        self.delivery_timeout = 30.0  # seconds
        self.retry_interval = 5.0     # seconds
        self.cleanup_interval = 300.0 # seconds
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start the messaging system"""
        self.running = True
        self.logger.info("Message passing system started")
        
        # Start background tasks
        asyncio.create_task(self.delivery_worker())
        asyncio.create_task(self.retry_worker())
        asyncio.create_task(self.cleanup_worker())
    
    async def stop(self) -> None:
        """Stop the messaging system"""
        self.running = False
        self.executor.shutdown(wait=True)
        self.logger.info("Message passing system stopped")
    
    async def register_agent(self, agent_id: str, routing_keys: Optional[List[str]] = None) -> None:
        """Register agent in the messaging system"""
        if agent_id not in self.agents:
            self.agents[agent_id] = MessageQueue(self.max_queue_size)
            if routing_keys:
                self.router.register_agent(agent_id, routing_keys)
            self.logger.info(f"Agent {agent_id} registered")
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister agent from the messaging system"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.handlers[agent_id]
            self.logger.info(f"Agent {agent_id} unregistered")
    
    def add_handler(self, agent_id: str, handler: MessageHandler) -> None:
        """Add message handler for agent"""
        self.handlers[agent_id].append(handler)
        self.router.add_handler(handler)
    
    async def send_message(self, message: Message) -> bool:
        """Send message through the system"""
        try:
            # Validate message
            if not message.sender_id or not message.recipient_id:
                self.logger.error(f"Invalid message: missing sender or recipient")
                return False
            
            # Store message persistently
            self.store.store_message(message)
            
            # Route message to target agents
            target_agents = await self.router.route_message(message)
            
            if not target_agents:
                self.logger.warning(f"No route found for message {message.id}")
                return False
            
            # Deliver to each target
            delivery_success = True
            for agent_id in target_agents:
                if agent_id in self.agents:
                    success = self.agents[agent_id].put(message)
                    if not success:
                        self.logger.warning(f"Queue full for agent {agent_id}")
                        delivery_success = False
                else:
                    self.logger.warning(f"Agent {agent_id} not found")
                    delivery_success = False
            
            if delivery_success:
                self.stats['messages_sent'] += 1
                self.logger.info(f"Message {message.id} queued for delivery")
            else:
                self.stats['messages_failed'] += 1
            
            return delivery_success
            
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            self.stats['messages_failed'] += 1
            return False
    
    async def receive_message(self, agent_id: str, timeout: float = 1.0) -> Optional[Message]:
        """Receive message for agent"""
        if agent_id not in self.agents:
            return None
        
        queue = self.agents[agent_id]
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            message = queue.get()
            if message:
                self.stats['messages_delivered'] += 1
                
                # Send acknowledgment if required
                if message.delivery_guarantee != DeliveryGuarantee.AT_MOST_ONCE:
                    self.store.acknowledge_delivery(message.id, agent_id)
                
                return message
            
            await asyncio.sleep(0.1)
        
        return None
    
    async def send_request(self, sender_id: str, recipient_id: str, 
                          payload: Dict[str, Any], timeout: float = 30.0) -> Optional[Message]:
        """Send request and wait for response"""
        correlation_id = str(uuid.uuid4())
        
        # Send request message
        request = Message(
            id="",
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.QUERY,
            payload=payload,
            correlation_id=correlation_id,
            reply_to=sender_id
        )
        
        success = await self.send_message(request)
        if not success:
            return None
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            message = await self.receive_message(sender_id, timeout=1.0)
            if (message and 
                message.message_type == MessageType.RESPONSE and 
                message.correlation_id == correlation_id):
                return message
        
        return None
    
    async def send_response(self, original_message: Message, 
                           payload: Dict[str, Any]) -> bool:
        """Send response to original message"""
        if not original_message.reply_to or not original_message.correlation_id:
            return False
        
        response = Message(
            id="",
            sender_id=original_message.recipient_id,
            recipient_id=original_message.reply_to,
            message_type=MessageType.RESPONSE,
            payload=payload,
            correlation_id=original_message.correlation_id
        )
        
        return await self.send_message(response)
    
    async def broadcast_message(self, sender_id: str, topic: str, 
                               payload: Dict[str, Any]) -> bool:
        """Broadcast message to all subscribers of topic"""
        message = Message(
            id="",
            sender_id=sender_id,
            recipient_id="",  # No specific recipient for broadcast
            message_type=MessageType.BROADCAST,
            payload=payload,
            headers={'topic': topic}
        )
        
        return await self.send_message(message)
    
    def subscribe_to_topic(self, agent_id: str, topic: str) -> None:
        """Subscribe agent to broadcast topic"""
        self.router.subscribe_to_topic(agent_id, topic)
    
    def unsubscribe_from_topic(self, agent_id: str, topic: str) -> None:
        """Unsubscribe agent from broadcast topic"""
        self.router.unsubscribe_from_topic(agent_id, topic)
    
    async def delivery_worker(self) -> None:
        """Background worker for message delivery"""
        while self.running:
            try:
                # Process pending deliveries
                for agent_id, queue in self.agents.items():
                    if not queue.is_empty():
                        message = queue.peek()
                        if message:
                            # Find and execute handler
                            handler = await self.router.find_handler(message)
                            if handler:
                                try:
                                    response = await handler.handle_message(message)
                                    if response:
                                        await self.send_message(response)
                                except Exception as e:
                                    self.logger.error(f"Handler error: {e}")
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Delivery worker error: {e}")
                await asyncio.sleep(1.0)
    
    async def retry_worker(self) -> None:
        """Background worker for message retries"""
        while self.running:
            try:
                # Check for messages needing retry
                pending_messages = self.store.get_pending_acks()
                
                for message in pending_messages:
                    if message.can_retry():
                        # Calculate retry delay (exponential backoff)
                        delay = self.retry_interval * (2 ** message.attempt_count)
                        
                        if time.time() - message.timestamp > delay:
                            message.increment_attempt()
                            await self.send_message(message)
                            self.stats['messages_retried'] += 1
                            self.logger.info(f"Retrying message {message.id}, attempt {message.attempt_count}")
                    else:
                        # Max retries reached
                        self.store.acknowledge_delivery(message.id, "SYSTEM_TIMEOUT")
                        self.stats['messages_failed'] += 1
                        self.logger.warning(f"Message {message.id} exceeded max retries")
                
                await asyncio.sleep(self.retry_interval)
                
            except Exception as e:
                self.logger.error(f"Retry worker error: {e}")
                await asyncio.sleep(1.0)
    
    async def cleanup_worker(self) -> None:
        """Background worker for system cleanup"""
        while self.running:
            try:
                # Clean up expired messages
                expired_count = self.store.cleanup_expired()
                if expired_count > 0:
                    self.logger.info(f"Cleaned up {expired_count} expired messages")
                
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Cleanup worker error: {e}")
                await asyncio.sleep(1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            **self.stats,
            'agents_registered': len(self.agents),
            'total_queue_size': sum(q.get_size() for q in self.agents.values()),
            'pending_acknowledgments': len(self.store.get_pending_acks()),
            'messages_stored': len(self.store.messages)
        }

# Example message handlers
class EchoMessageHandler(MessageHandler):
    """Simple echo handler for testing"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Echo the message back to sender"""
        if message.message_type == MessageType.QUERY:
            return Message(
                id="",
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                payload={'echo': message.payload, 'processed_by': self.agent_id},
                correlation_id=message.correlation_id
            )
        return None
    
    def can_handle(self, message: Message) -> bool:
        """Can handle query messages"""
        return (message.recipient_id == self.agent_id and 
                message.message_type == MessageType.QUERY)

class DataProcessorHandler(MessageHandler):
    """Handler for data processing commands"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Process data from message"""
        if message.message_type == MessageType.COMMAND:
            action = message.payload.get('action')
            
            if action == 'process_data':
                data = message.payload.get('data', [])
                # Simulate data processing
                result = [x * 2 for x in data if isinstance(x, (int, float))]
                
                if message.reply_to:
                    return Message(
                        id="",
                        sender_id=self.agent_id,
                        recipient_id=message.reply_to,
                        message_type=MessageType.RESPONSE,
                        payload={'result': result, 'processed_by': self.agent_id},
                        correlation_id=message.correlation_id
                    )
        
        return None
    
    def can_handle(self, message: Message) -> bool:
        """Can handle command messages for this agent"""
        return (message.recipient_id == self.agent_id and 
                message.message_type == MessageType.COMMAND)

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_messaging():
    """Demo: Basic point-to-point messaging"""
    print("\nDEMO 1: BASIC AGENT MESSAGING")
    print("=" * 50)
    
    # Create messaging system
    messaging = MessagePassingSystem()
    await messaging.start()
    
    # Register agents
    await messaging.register_agent("sender")
    await messaging.register_agent("receiver")
    
    # Add handlers
    echo_handler = EchoMessageHandler("receiver")
    messaging.add_handler("receiver", echo_handler)
    
    # Send simple message
    message = Message(
        id="",
        sender_id="sender",
        recipient_id="receiver",
        message_type=MessageType.QUERY,
        payload={"question": "What is 2 + 2?"}
    )
    
    print(f"Sending message: {message.payload}")
    success = await messaging.send_message(message)
    print(f"Message sent: {success}")
    
    # Wait for processing and response
    await asyncio.sleep(1.0)
    
    # Receive response
    response = await messaging.receive_message("sender", timeout=5.0)
    if response:
        print(f"Received response: {response.payload}")
    else:
        print("No response received")
    
    await messaging.stop()

async def demo_request_response():
    """Demo: Request-response pattern with timeout"""
    print("\nDEMO 2: REQUEST-RESPONSE PATTERN")
    print("=" * 50)
    
    messaging = MessagePassingSystem()
    await messaging.start()
    
    # Register agents
    await messaging.register_agent("client")
    await messaging.register_agent("server")
    
    # Add data processor handler
    processor_handler = DataProcessorHandler("server")
    messaging.add_handler("server", processor_handler)
    
    # Send request and wait for response
    payload = {"action": "process_data", "data": [1, 2, 3, 4, 5]}
    
    print(f"Sending request: {payload}")
    response = await messaging.send_request("client", "server", payload, timeout=10.0)
    
    if response:
        print(f"Received response: {response.payload}")
    else:
        print("Request timed out")
    
    await messaging.stop()

async def demo_broadcast_messaging():
    """Demo: Broadcast messaging to multiple subscribers"""
    print("\nDEMO 3: BROADCAST MESSAGING")
    print("=" * 50)
    
    messaging = MessagePassingSystem()
    await messaging.start()
    
    # Register multiple agents
    agents = ["broadcaster", "listener1", "listener2", "listener3"]
    for agent_id in agents:
        await messaging.register_agent(agent_id)
    
    # Subscribe listeners to news topic
    for listener in ["listener1", "listener2", "listener3"]:
        messaging.subscribe_to_topic(listener, "news")
    
    # Broadcast news update
    news_payload = {
        "headline": "AI Agents Achieve Breakthrough in Communication",
        "content": "New messaging protocols enable seamless agent coordination",
        "timestamp": time.time()
    }
    
    print(f"Broadcasting news: {news_payload['headline']}")
    success = await messaging.broadcast_message("broadcaster", "news", news_payload)
    print(f"Broadcast sent: {success}")
    
    # Wait for delivery
    await asyncio.sleep(1.0)
    
    # Check each listener received the broadcast
    for listener in ["listener1", "listener2", "listener3"]:
        message = await messaging.receive_message(listener, timeout=2.0)
        if message:
            print(f"{listener} received: {message.payload['headline']}")
        else:
            print(f"{listener} did not receive broadcast")
    
    await messaging.stop()

async def demo_reliable_delivery():
    """Demo: Reliable message delivery with retries"""
    print("\nDEMO 4: RELIABLE MESSAGE DELIVERY")
    print("=" * 50)
    
    messaging = MessagePassingSystem()
    await messaging.start()
    
    # Register agents
    await messaging.register_agent("sender")
    await messaging.register_agent("receiver")
    
    # Send critical message with guaranteed delivery
    critical_message = Message(
        id="",
        sender_id="sender",
        recipient_id="receiver",
        message_type=MessageType.COMMAND,
        payload={"action": "critical_task", "data": "important_data"},
        priority=MessagePriority.CRITICAL,
        delivery_guarantee=DeliveryGuarantee.EXACTLY_ONCE,
        ttl=60.0  # 1 minute to live
    )
    
    print(f"Sending critical message with guaranteed delivery")
    success = await messaging.send_message(critical_message)
    print(f"Message queued: {success}")
    
    # Show system statistics
    await asyncio.sleep(2.0)
    stats = messaging.get_statistics()
    print(f"System stats: {stats}")
    
    # Simulate message processing
    message = await messaging.receive_message("receiver", timeout=5.0)
    if message:
        print(f"Received critical message: {message.payload}")
        print(f"Priority: {message.priority.value}")
        print(f"Delivery guarantee: {message.delivery_guarantee.value}")
    
    await messaging.stop()

async def demo_message_patterns():
    """Demo: Various message patterns and types"""
    print("\nDEMO 5: MESSAGE PATTERNS SHOWCASE")
    print("=" * 50)
    
    messaging = MessagePassingSystem()
    await messaging.start()
    
    # Register agents for different patterns
    agents = ["coordinator", "worker1", "worker2", "monitor"]
    for agent_id in agents:
        await messaging.register_agent(agent_id)
    
    print("1. Command Pattern: Coordinator -> Workers")
    # Command pattern: coordinator sends commands to workers
    for i, worker in enumerate(["worker1", "worker2"]):
        command = Message(
            id="",
            sender_id="coordinator",
            recipient_id=worker,
            message_type=MessageType.COMMAND,
            payload={"action": "start_task", "task_id": f"task_{i+1}"}
        )
        await messaging.send_message(command)
        print(f"  Sent command to {worker}: task_{i+1}")
    
    print("\n2. Event Pattern: Workers -> Monitor")
    # Event pattern: workers send events to monitor
    for i, worker in enumerate(["worker1", "worker2"]):
        event = Message(
            id="",
            sender_id=worker,
            recipient_id="monitor",
            message_type=MessageType.EVENT,
            payload={"event": "task_completed", "worker": worker, "task_id": f"task_{i+1}"}
        )
        await messaging.send_message(event)
        print(f"  {worker} sent completion event")
    
    print("\n3. Heartbeat Pattern: All Agents")
    # Heartbeat pattern: agents send heartbeats
    for agent in agents:
        heartbeat = Message(
            id="",
            sender_id=agent,
            recipient_id="monitor",
            message_type=MessageType.HEARTBEAT,
            payload={"status": "alive", "timestamp": time.time()}
        )
        await messaging.send_message(heartbeat)
        print(f"  {agent} sent heartbeat")
    
    # Process messages
    await asyncio.sleep(1.0)
    
    print("\nReceived messages:")
    # Show received messages
    for agent in agents:
        while True:
            message = await messaging.receive_message(agent, timeout=0.1)
            if not message:
                break
            print(f"  {agent} received {message.message_type.value}: {message.payload}")
    
    await messaging.stop()

async def main():
    """
    Demonstrate Message Passing Systems for reliable agent communication
    
    WHAT YOU'LL LEARN:
    ================
    1. How to design reliable message passing systems
    2. How to implement various message delivery guarantees
    3. How to handle asynchronous agent communication
    4. How to build scalable messaging architectures
    5. How to handle message routing, retries, and failures
    
    REAL WORLD APPLICATIONS:
    =======================
    - Distributed agent systems and microservices
    - Multi-agent coordination and workflow systems
    - Real-time communication platforms
    - IoT device management and coordination
    - Event-driven architectures and message queues
    - Fault-tolerant distributed systems
    """
    
    print("MESSAGE PASSING SYSTEMS DEMONSTRATION")
    print("Showing how agents communicate reliably at scale!")
    
    await demo_basic_messaging()
    await demo_request_response()
    await demo_broadcast_messaging()
    await demo_reliable_delivery()
    await demo_message_patterns()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Message passing enables reliable distributed communication")
    print("✓ Different delivery guarantees for different use cases")
    print("✓ Asynchronous communication prevents blocking")
    print("✓ Message routing and queuing handle scalability")
    print("✓ Retry mechanisms provide fault tolerance")
    print("✓ Multiple communication patterns for different scenarios")
    print("\nTHE POWER OF MESSAGE PASSING:")
    print("- Scales to thousands of distributed agents")
    print("- Provides reliable communication guarantees")
    print("- Enables complex multi-agent workflows")
    print("- Handles network failures gracefully")

if __name__ == "__main__":
    asyncio.run(main())
