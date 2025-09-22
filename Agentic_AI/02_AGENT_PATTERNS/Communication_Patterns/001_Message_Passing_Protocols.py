#!/usr/bin/env python3
"""
Message Passing Protocols: Agent Communication Standards
=====================================================

Theory Summary:
Message passing protocols define standardized ways for agents to communicate,
ensuring reliable, structured, and meaningful exchanges. These protocols
handle message formatting, delivery guarantees, routing, and conversation
management between distributed agents.

Key Concepts:
- Standardized message formats and schemas
- Delivery guarantees (at-least-once, exactly-once)
- Message routing and addressing
- Conversation and session management
- Protocol versioning and compatibility

Use Cases:
- Distributed agent systems
- Microservices communication
- Event-driven architectures
- Real-time collaborative systems
- Cross-platform agent integration
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib

class MessageType(Enum):
    """Standard message types in agent communication"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    HEARTBEAT = "heartbeat"
    ACKNOWLEDGMENT = "acknowledgment"
    ERROR = "error"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PUBLISH = "publish"

class DeliveryGuarantee(Enum):
    """Message delivery guarantee levels"""
    AT_MOST_ONCE = "at_most_once"      # May be lost, never duplicated
    AT_LEAST_ONCE = "at_least_once"    # Never lost, may be duplicated
    EXACTLY_ONCE = "exactly_once"      # Never lost, never duplicated

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class MessageHeader:
    """Standard message header with metadata"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    timestamp: float = field(default_factory=time.time)
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE
    conversation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expires_at: Optional[float] = None
    version: str = "1.0"
    checksum: Optional[str] = None

@dataclass
class Message:
    """Complete message with header and payload"""
    header: MessageHeader
    payload: Any
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.header.checksum:
            self.header.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate message checksum for integrity verification"""
        content = f"{self.header.message_id}{self.header.sender_id}{self.header.recipient_id}{json.dumps(self.payload, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify message integrity using checksum"""
        calculated_checksum = self._calculate_checksum()
        return calculated_checksum == self.header.checksum
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.header.expires_at is None:
            return False
        return time.time() > self.header.expires_at

class MessageProtocol(ABC):
    """Abstract base class for message protocols"""
    
    @abstractmethod
    async def send_message(self, message: Message) -> bool:
        """Send a message - returns success status"""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[Message]:
        """Receive a message - returns None if no message available"""
        pass
    
    @abstractmethod
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register handler for specific message type"""
        pass

class ReliableMessageProtocol(MessageProtocol):
    """
    Reliable message protocol with delivery guarantees
    
    Provides features like acknowledgments, retries, deduplication,
    and conversation tracking for robust agent communication.
    """
    
    def __init__(self, agent_id: str, delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE):
        self.agent_id = agent_id
        self.delivery_guarantee = delivery_guarantee
        self.handlers: Dict[MessageType, Callable] = {}
        self.pending_messages: Dict[str, Message] = {}  # Awaiting acknowledgment
        self.received_messages: Dict[str, float] = {}   # For deduplication
        self.conversations: Dict[str, Dict] = {}        # Conversation tracking
        self.subscriptions: Dict[str, List[str]] = {}   # Topic subscriptions
        self.message_queue: List[Message] = []
        self.connected_agents: Dict[str, 'ReliableMessageProtocol'] = {}
        
        # Protocol statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "acknowledgments_sent": 0,
            "acknowledgments_received": 0,
            "duplicates_detected": 0
        }
        
        # Start background tasks
        self.background_tasks = []
        self.background_tasks.append(asyncio.create_task(self._retry_pending_messages()))
        self.background_tasks.append(asyncio.create_task(self._cleanup_old_messages()))
    
    def connect_to_agent(self, agent_protocol: 'ReliableMessageProtocol') -> None:
        """Connect to another agent's protocol"""
        self.connected_agents[agent_protocol.agent_id] = agent_protocol
        agent_protocol.connected_agents[self.agent_id] = self
    
    async def send_message(self, message: Message) -> bool:
        """Send message with delivery guarantees"""
        # Validate message
        if not self._validate_message(message):
            return False
        
        # Check if recipient is connected
        if message.header.recipient_id not in self.connected_agents:
            self.stats["messages_failed"] += 1
            return False
        
        try:
            # Send message to recipient
            recipient = self.connected_agents[message.header.recipient_id]
            recipient.message_queue.append(message)
            
            # Handle delivery guarantees
            if self.delivery_guarantee in [DeliveryGuarantee.AT_LEAST_ONCE, DeliveryGuarantee.EXACTLY_ONCE]:
                # Store for potential retry
                self.pending_messages[message.header.message_id] = message
            
            self.stats["messages_sent"] += 1
            return True
            
        except Exception as e:
            print(f"Error sending message: {e}")
            self.stats["messages_failed"] += 1
            return False
    
    async def receive_message(self) -> Optional[Message]:
        """Receive and process next message from queue"""
        if not self.message_queue:
            return None
        
        message = self.message_queue.pop(0)
        
        # Verify message integrity
        if not message.verify_integrity():
            print(f"Message integrity check failed: {message.header.message_id}")
            return None
        
        # Check if message is expired
        if message.is_expired():
            print(f"Message expired: {message.header.message_id}")
            return None
        
        # Handle deduplication for exactly-once delivery
        if self.delivery_guarantee == DeliveryGuarantee.EXACTLY_ONCE:
            if message.header.message_id in self.received_messages:
                self.stats["duplicates_detected"] += 1
                await self._send_acknowledgment(message)  # Acknowledge duplicate
                return None
        
        # Record message reception
        self.received_messages[message.header.message_id] = time.time()
        self.stats["messages_received"] += 1
        
        # Send acknowledgment if required
        if message.header.delivery_guarantee in [DeliveryGuarantee.AT_LEAST_ONCE, DeliveryGuarantee.EXACTLY_ONCE]:
            await self._send_acknowledgment(message)
        
        # Update conversation tracking
        if message.header.conversation_id:
            self._update_conversation(message)
        
        # Process message through handlers
        await self._process_message(message)
        
        return message
    
    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Register handler for specific message type"""
        self.handlers[message_type] = handler
    
    async def send_request(self, recipient_id: str, payload: Any, conversation_id: str = None) -> str:
        """Send request message and return message ID"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        message = self._create_message(
            recipient_id=recipient_id,
            message_type=MessageType.REQUEST,
            payload=payload,
            conversation_id=conversation_id
        )
        
        success = await self.send_message(message)
        return message.header.message_id if success else None
    
    async def send_response(self, request_message: Message, payload: Any) -> bool:
        """Send response to a request message"""
        response = self._create_message(
            recipient_id=request_message.header.sender_id,
            message_type=MessageType.RESPONSE,
            payload=payload,
            conversation_id=request_message.header.conversation_id,
            reply_to=request_message.header.message_id
        )
        
        return await self.send_message(response)
    
    async def send_notification(self, recipient_id: str, payload: Any) -> bool:
        """Send notification message (fire-and-forget)"""
        message = self._create_message(
            recipient_id=recipient_id,
            message_type=MessageType.NOTIFICATION,
            payload=payload,
            delivery_guarantee=DeliveryGuarantee.AT_MOST_ONCE
        )
        
        return await self.send_message(message)
    
    async def publish_message(self, topic: str, payload: Any) -> int:
        """Publish message to topic subscribers"""
        subscribers = self.subscriptions.get(topic, [])
        success_count = 0
        
        for subscriber_id in subscribers:
            message = self._create_message(
                recipient_id=subscriber_id,
                message_type=MessageType.PUBLISH,
                payload={"topic": topic, "data": payload}
            )
            
            if await self.send_message(message):
                success_count += 1
        
        return success_count
    
    async def subscribe_to_topic(self, topic: str, publisher_id: str) -> bool:
        """Subscribe to messages from a topic"""
        message = self._create_message(
            recipient_id=publisher_id,
            message_type=MessageType.SUBSCRIBE,
            payload={"topic": topic, "subscriber": self.agent_id}
        )
        
        return await self.send_message(message)
    
    def _create_message(self, recipient_id: str, message_type: MessageType, payload: Any,
                       conversation_id: str = None, reply_to: str = None,
                       priority: MessagePriority = MessagePriority.NORMAL,
                       delivery_guarantee: DeliveryGuarantee = None) -> Message:
        """Create a properly formatted message"""
        if delivery_guarantee is None:
            delivery_guarantee = self.delivery_guarantee
        
        header = MessageHeader(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            priority=priority,
            delivery_guarantee=delivery_guarantee,
            conversation_id=conversation_id,
            reply_to=reply_to,
            expires_at=time.time() + 300 if delivery_guarantee != DeliveryGuarantee.AT_MOST_ONCE else None  # 5 minute expiry
        )
        
        return Message(header=header, payload=payload)
    
    def _validate_message(self, message: Message) -> bool:
        """Validate message before sending"""
        # Check required fields
        if not all([message.header.message_id, message.header.sender_id, message.header.recipient_id]):
            return False
        
        # Check message size (simplified)
        try:
            message_size = len(json.dumps(message.payload))
            if message_size > 1024 * 1024:  # 1MB limit
                return False
        except:
            return False
        
        return True
    
    async def _send_acknowledgment(self, original_message: Message) -> None:
        """Send acknowledgment for received message"""
        ack_message = self._create_message(
            recipient_id=original_message.header.sender_id,
            message_type=MessageType.ACKNOWLEDGMENT,
            payload={"acknowledged_message_id": original_message.header.message_id},
            delivery_guarantee=DeliveryGuarantee.AT_MOST_ONCE
        )
        
        await self.send_message(ack_message)
        self.stats["acknowledgments_sent"] += 1
    
    async def _process_message(self, message: Message) -> None:
        """Process message through registered handlers"""
        message_type = message.header.message_type
        
        # Handle acknowledgments
        if message_type == MessageType.ACKNOWLEDGMENT:
            await self._handle_acknowledgment(message)
            return
        
        # Handle subscriptions
        if message_type == MessageType.SUBSCRIBE:
            await self._handle_subscription(message)
            return
        
        # Handle unsubscriptions
        if message_type == MessageType.UNSUBSCRIBE:
            await self._handle_unsubscription(message)
            return
        
        # Call registered handler
        if message_type in self.handlers:
            try:
                await self.handlers[message_type](message)
            except Exception as e:
                print(f"Error in message handler: {e}")
                # Send error response if it was a request
                if message_type == MessageType.REQUEST:
                    await self._send_error_response(message, str(e))
    
    async def _handle_acknowledgment(self, message: Message) -> None:
        """Handle acknowledgment message"""
        payload = message.payload
        acked_message_id = payload.get("acknowledged_message_id")
        
        if acked_message_id in self.pending_messages:
            del self.pending_messages[acked_message_id]
            self.stats["acknowledgments_received"] += 1
    
    async def _handle_subscription(self, message: Message) -> None:
        """Handle subscription request"""
        payload = message.payload
        topic = payload.get("topic")
        subscriber = payload.get("subscriber")
        
        if topic and subscriber:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = []
            
            if subscriber not in self.subscriptions[topic]:
                self.subscriptions[topic].append(subscriber)
    
    async def _handle_unsubscription(self, message: Message) -> None:
        """Handle unsubscription request"""
        payload = message.payload
        topic = payload.get("topic")
        subscriber = payload.get("subscriber")
        
        if topic in self.subscriptions and subscriber in self.subscriptions[topic]:
            self.subscriptions[topic].remove(subscriber)
    
    async def _send_error_response(self, request_message: Message, error_message: str) -> None:
        """Send error response for failed request"""
        error_response = self._create_message(
            recipient_id=request_message.header.sender_id,
            message_type=MessageType.ERROR,
            payload={"error": error_message, "original_message_id": request_message.header.message_id},
            conversation_id=request_message.header.conversation_id,
            reply_to=request_message.header.message_id
        )
        
        await self.send_message(error_response)
    
    def _update_conversation(self, message: Message) -> None:
        """Update conversation tracking"""
        conv_id = message.header.conversation_id
        
        if conv_id not in self.conversations:
            self.conversations[conv_id] = {
                "participants": set(),
                "messages": [],
                "started_at": time.time(),
                "last_activity": time.time()
            }
        
        conversation = self.conversations[conv_id]
        conversation["participants"].add(message.header.sender_id)
        conversation["participants"].add(message.header.recipient_id)
        conversation["messages"].append({
            "message_id": message.header.message_id,
            "sender": message.header.sender_id,
            "type": message.header.message_type.value,
            "timestamp": message.header.timestamp
        })
        conversation["last_activity"] = time.time()
    
    async def _retry_pending_messages(self) -> None:
        """Background task to retry unacknowledged messages"""
        while True:
            try:
                current_time = time.time()
                retry_messages = []
                
                for message_id, message in list(self.pending_messages.items()):
                    # Check if message needs retry (after 5 seconds)
                    if current_time - message.header.timestamp > 5.0:
                        if message.retry_count < message.max_retries:
                            message.retry_count += 1
                            retry_messages.append(message)
                        else:
                            # Max retries exceeded
                            del self.pending_messages[message_id]
                            self.stats["messages_failed"] += 1
                
                # Retry messages
                for message in retry_messages:
                    await self.send_message(message)
                
                await asyncio.sleep(1.0)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in retry task: {e}")
                await asyncio.sleep(1.0)
    
    async def _cleanup_old_messages(self) -> None:
        """Background task to clean up old message records"""
        while True:
            try:
                current_time = time.time()
                
                # Clean up old received message records (after 1 hour)
                old_messages = [
                    msg_id for msg_id, timestamp in self.received_messages.items()
                    if current_time - timestamp > 3600
                ]
                for msg_id in old_messages:
                    del self.received_messages[msg_id]
                
                # Clean up old conversations (after 1 hour of inactivity)
                old_conversations = [
                    conv_id for conv_id, conv in self.conversations.items()
                    if current_time - conv["last_activity"] > 3600
                ]
                for conv_id in old_conversations:
                    del self.conversations[conv_id]
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in cleanup task: {e}")
                await asyncio.sleep(300)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get protocol statistics"""
        return {
            "agent_id": self.agent_id,
            "delivery_guarantee": self.delivery_guarantee.value,
            "connected_agents": len(self.connected_agents),
            "pending_messages": len(self.pending_messages),
            "active_conversations": len(self.conversations),
            "topic_subscriptions": {topic: len(subscribers) for topic, subscribers in self.subscriptions.items()},
            "statistics": self.stats.copy()
        }
    
    async def shutdown(self) -> None:
        """Shutdown protocol and cleanup"""
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)

# Example usage and learning demonstration
async def main():
    """
    Demonstration of Message Passing Protocols
    
    Learning Objectives:
    1. Understand structured agent communication
    2. See delivery guarantees in action
    3. Learn conversation and session management
    4. Practice pub/sub messaging patterns
    """
    
    # Create agents with different delivery guarantees
    agent_a = ReliableMessageProtocol("agent_a", DeliveryGuarantee.EXACTLY_ONCE)
    agent_b = ReliableMessageProtocol("agent_b", DeliveryGuarantee.AT_LEAST_ONCE)
    agent_c = ReliableMessageProtocol("agent_c", DeliveryGuarantee.AT_MOST_ONCE)
    
    # Connect agents
    agent_a.connect_to_agent(agent_b)
    agent_a.connect_to_agent(agent_c)
    agent_b.connect_to_agent(agent_c)
    
    print("=== Message Passing Protocol Demo ===")
    print(f"Connected 3 agents with different delivery guarantees")
    
    # Register message handlers
    async def request_handler(message: Message):
        print(f"Received request: {message.payload}")
        await agent_b.send_response(message, {"result": "Request processed successfully"})
    
    async def response_handler(message: Message):
        print(f"Received response: {message.payload}")
    
    async def notification_handler(message: Message):
        print(f"Received notification: {message.payload}")
    
    async def publish_handler(message: Message):
        print(f"Received published message: {message.payload}")
    
    # Register handlers
    agent_b.register_handler(MessageType.REQUEST, request_handler)
    agent_a.register_handler(MessageType.RESPONSE, response_handler)
    agent_c.register_handler(MessageType.NOTIFICATION, notification_handler)
    agent_c.register_handler(MessageType.PUBLISH, publish_handler)
    
    # Example 1: Request-Response pattern
    print("\n=== Request-Response Example ===")
    request_id = await agent_a.send_request(
        "agent_b", 
        {"action": "process_data", "data": [1, 2, 3, 4, 5]}
    )
    print(f"Sent request: {request_id}")
    
    # Process messages
    await agent_b.receive_message()  # Process request
    await agent_a.receive_message()  # Process response
    
    # Example 2: Notification pattern
    print("\n=== Notification Example ===")
    await agent_a.send_notification(
        "agent_c",
        {"event": "system_update", "version": "2.1.0"}
    )
    await agent_c.receive_message()  # Process notification
    
    # Example 3: Publish-Subscribe pattern
    print("\n=== Publish-Subscribe Example ===")
    # Subscribe to topic
    await agent_c.subscribe_to_topic("news_updates", "agent_a")
    await agent_a.receive_message()  # Process subscription
    
    # Publish to topic
    published_count = await agent_a.publish_message(
        "news_updates",
        {"headline": "New AI breakthrough announced", "priority": "high"}
    )
    print(f"Published message to {published_count} subscribers")
    await agent_c.receive_message()  # Process published message
    
    # Example 4: Show protocol statistics
    print("\n=== Protocol Statistics ===")
    for agent in [agent_a, agent_b, agent_c]:
        stats = agent.get_statistics()
        print(f"\n{stats['agent_id']}:")
        print(f"  Delivery guarantee: {stats['delivery_guarantee']}")
        print(f"  Messages sent: {stats['statistics']['messages_sent']}")
        print(f"  Messages received: {stats['statistics']['messages_received']}")
        print(f"  Acknowledgments sent: {stats['statistics']['acknowledgments_sent']}")
        print(f"  Acknowledgments received: {stats['statistics']['acknowledgments_received']}")
        print(f"  Active conversations: {stats['active_conversations']}")
    
    # Cleanup
    await agent_a.shutdown()
    await agent_b.shutdown()
    await agent_c.shutdown()
    
    # Summary of learning
    print("\n=== What We Learned ===")
    print("1. Message protocols ensure reliable agent communication")
    print("2. Delivery guarantees provide different reliability levels")
    print("3. Conversation tracking enables complex interactions")
    print("4. Pub/sub patterns enable scalable message distribution")
    print("5. Protocol statistics help monitor system health")

if __name__ == "__main__":
    asyncio.run(main())
