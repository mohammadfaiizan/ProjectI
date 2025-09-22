#!/usr/bin/env python3
"""
Publish-Subscribe Patterns: Scalable Agent Broadcasting
======================================================

WHAT IS THE PROBLEM?
==================
Traditional one-to-one communication doesn't work for:
- Broadcasting updates to multiple agents
- Dynamic subscriber lists
- Loose coupling between publishers and consumers
- Scalable fan-out communication
- Topic-based filtering and routing

Example: Newsletter Nightmare
TRADITIONAL APPROACH (Breaks):
- News service maintains list of all subscribers
- Directly calls each subscriber individually
- Adding new subscriber requires code changes
- Removing subscriber risks breaking system
- No way to handle different interests/topics

REAL WORLD EXAMPLE:
=================
How does YouTube notify millions of subscribers?

YOUTUBE NOTIFICATION SYSTEM:
When PewDiePie uploads a video:
1. PUBLISH: Video upload triggers notification event
2. TOPIC: Event published to "PewDiePie_uploads" topic
3. SUBSCRIBERS: 100+ million subscribers get notified
4. FILTERING: Users can choose notification preferences
5. DELIVERY: Push notifications, emails, app badges
6. SCALING: System handles millions of notifications instantly

BENEFITS:
- Publishers don't know who's listening
- Subscribers can join/leave dynamically
- Topic-based filtering for relevance
- Massive fan-out with minimal latency
- Independent scaling of publishers and subscribers

THE ALGORITHM:
=============
1. SUBSCRIBE: Agents subscribe to topics of interest
2. PUBLISH: Agents publish messages to topics
3. MATCH: System finds all topic subscribers
4. FILTER: Apply subscriber-specific filters
5. DELIVER: Fan-out messages to all matches
6. SCALE: Parallelize delivery across many workers
7. MONITOR: Track delivery success and failures

PATTERNS:
- Topic-based: Subscribe to specific subjects
- Content-based: Filter by message content
- Hierarchical: Tree-structured topic organization
- Wild-card: Pattern matching for subscriptions

WHY IS THIS REVOLUTIONARY?
========================
- Enables true decoupling of system components
- Scales to millions of publishers and subscribers
- Supports dynamic system reconfiguration
- Provides natural load distribution
- Enables real-time data streaming at scale
- Powers modern event architectures
"""

import asyncio
import time
import json
import uuid
import re
from typing import Dict, List, Any, Optional, Callable, Set, Pattern
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
import heapq
import fnmatch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class SubscriptionType(Enum):
    """Types of subscriptions"""
    TOPIC_EXACT = "topic_exact"           # Exact topic match
    TOPIC_PATTERN = "topic_pattern"       # Pattern-based topic matching
    CONTENT_FILTER = "content_filter"     # Content-based filtering
    HIERARCHICAL = "hierarchical"         # Tree-based topic hierarchy
    WILDCARD = "wildcard"                # Wildcard pattern matching

class DeliveryMode(Enum):
    """Message delivery modes"""
    FIRE_AND_FORGET = "fire_and_forget"   # No delivery confirmation
    AT_LEAST_ONCE = "at_least_once"       # Guaranteed delivery with possible duplicates
    AT_MOST_ONCE = "at_most_once"         # No duplicates but possible message loss
    EXACTLY_ONCE = "exactly_once"         # Guaranteed exactly one delivery

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class PubSubMessage:
    """Message for publish-subscribe communication"""
    id: str
    topic: str
    payload: Dict[str, Any]
    publisher_id: str
    
    # Message metadata
    timestamp: float = field(default_factory=time.time)
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_mode: DeliveryMode = DeliveryMode.FIRE_AND_FORGET
    
    # Content attributes for filtering
    content_type: str = "application/json"
    tags: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Delivery control
    ttl: Optional[float] = None           # Time to live in seconds
    retain: bool = False                  # Retain for new subscribers
    correlation_id: Optional[str] = None
    
    # Tracking
    delivery_count: int = 0
    failed_deliveries: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize message with defaults"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def add_tag(self, tag: str) -> None:
        """Add tag to message"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if message has tag"""
        return tag in self.tags
    
    def increment_delivery(self) -> None:
        """Increment delivery counter"""
        self.delivery_count += 1
    
    def record_delivery_failure(self, subscriber_id: str) -> None:
        """Record delivery failure"""
        if subscriber_id not in self.failed_deliveries:
            self.failed_deliveries.append(subscriber_id)
    
    def serialize(self) -> str:
        """Serialize message to JSON"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['delivery_mode'] = self.delivery_mode.value
        return json.dumps(data)
    
    @classmethod
    def deserialize(cls, data: str) -> 'PubSubMessage':
        """Deserialize message from JSON"""
        obj = json.loads(data)
        obj['priority'] = MessagePriority(obj['priority'])
        obj['delivery_mode'] = DeliveryMode(obj['delivery_mode'])
        return cls(**obj)

class ContentFilter:
    """Filter for content-based subscriptions"""
    
    def __init__(self, conditions: Dict[str, Any] = None):
        self.conditions = conditions or {}
    
    def matches(self, message: PubSubMessage) -> bool:
        """Check if message matches filter conditions"""
        for key, expected_value in self.conditions.items():
            if key.startswith("header."):
                # Header filtering
                header_key = key[7:]  # Remove "header." prefix
                actual_value = message.headers.get(header_key)
            elif key.startswith("tag."):
                # Tag filtering
                tag = key[4:]  # Remove "tag." prefix
                return message.has_tag(tag)
            elif key.startswith("payload."):
                # Payload filtering
                payload_key = key[8:]  # Remove "payload." prefix
                actual_value = message.payload.get(payload_key)
            elif key == "priority":
                actual_value = message.priority.value
            elif key == "publisher_id":
                actual_value = message.publisher_id
            elif key == "content_type":
                actual_value = message.content_type
            else:
                continue
            
            # Value matching
            if isinstance(expected_value, str) and expected_value.startswith("regex:"):
                pattern = expected_value[6:]  # Remove "regex:" prefix
                if not re.match(pattern, str(actual_value)):
                    return False
            elif isinstance(expected_value, list):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False
        
        return True

class Subscription:
    """Subscription configuration for a subscriber"""
    
    def __init__(self, subscriber_id: str, topic_pattern: str,
                 subscription_type: SubscriptionType = SubscriptionType.TOPIC_EXACT,
                 content_filter: Optional[ContentFilter] = None,
                 delivery_mode: DeliveryMode = DeliveryMode.FIRE_AND_FORGET,
                 max_delivery_rate: Optional[float] = None):
        
        self.id = str(uuid.uuid4())
        self.subscriber_id = subscriber_id
        self.topic_pattern = topic_pattern
        self.subscription_type = subscription_type
        self.content_filter = content_filter
        self.delivery_mode = delivery_mode
        self.max_delivery_rate = max_delivery_rate  # Messages per second
        
        # Subscription state
        self.created_at = time.time()
        self.active = True
        self.message_count = 0
        self.last_delivery = 0.0
        self.delivery_queue: deque = deque()
        
        # Compile pattern for efficiency
        if subscription_type == SubscriptionType.TOPIC_PATTERN:
            self.compiled_pattern = re.compile(topic_pattern)
        elif subscription_type == SubscriptionType.WILDCARD:
            # Convert shell-style wildcards to regex
            regex_pattern = fnmatch.translate(topic_pattern)
            self.compiled_pattern = re.compile(regex_pattern)
        else:
            self.compiled_pattern = None
    
    def matches_topic(self, topic: str) -> bool:
        """Check if topic matches subscription"""
        if self.subscription_type == SubscriptionType.TOPIC_EXACT:
            return topic == self.topic_pattern
        
        elif self.subscription_type == SubscriptionType.HIERARCHICAL:
            # Hierarchical matching (e.g., "sports.football" matches "sports.*")
            topic_parts = topic.split('.')
            pattern_parts = self.topic_pattern.split('.')
            
            if len(pattern_parts) > len(topic_parts):
                return False
            
            for i, pattern_part in enumerate(pattern_parts):
                if pattern_part == "*":
                    continue
                elif pattern_part == "**":
                    return True  # Match remaining hierarchy
                elif i >= len(topic_parts) or pattern_part != topic_parts[i]:
                    return False
            
            return len(pattern_parts) == len(topic_parts)
        
        elif self.subscription_type in [SubscriptionType.TOPIC_PATTERN, SubscriptionType.WILDCARD]:
            return bool(self.compiled_pattern.match(topic))
        
        return False
    
    def matches_message(self, message: PubSubMessage) -> bool:
        """Check if message matches subscription"""
        # Check topic match
        if not self.matches_topic(message.topic):
            return False
        
        # Check content filter
        if self.content_filter and not self.content_filter.matches(message):
            return False
        
        # Check delivery rate limit
        if self.max_delivery_rate:
            current_time = time.time()
            if current_time - self.last_delivery < (1.0 / self.max_delivery_rate):
                return False
        
        return True
    
    def can_deliver_now(self) -> bool:
        """Check if we can deliver message now (rate limiting)"""
        if not self.max_delivery_rate:
            return True
        
        current_time = time.time()
        return current_time - self.last_delivery >= (1.0 / self.max_delivery_rate)
    
    def record_delivery(self) -> None:
        """Record successful delivery"""
        self.message_count += 1
        self.last_delivery = time.time()

class Subscriber(ABC):
    """Abstract base class for message subscribers"""
    
    @abstractmethod
    async def handle_message(self, message: PubSubMessage) -> bool:
        """Handle received message. Return True if successful."""
        pass
    
    @abstractmethod
    def get_subscriber_id(self) -> str:
        """Get unique subscriber identifier"""
        pass

class Topic:
    """Topic for organizing messages"""
    
    def __init__(self, name: str, max_retained_messages: int = 1000):
        self.name = name
        self.max_retained_messages = max_retained_messages
        
        # Subscribers for this topic
        self.subscriptions: List[Subscription] = []
        
        # Retained messages for new subscribers
        self.retained_messages: deque = deque(maxlen=max_retained_messages)
        
        # Topic statistics
        self.total_published = 0
        self.total_delivered = 0
        self.total_failed = 0
        self.created_at = time.time()
        
        self.lock = threading.Lock()
    
    def add_subscription(self, subscription: Subscription) -> None:
        """Add subscription to topic"""
        with self.lock:
            self.subscriptions.append(subscription)
    
    def remove_subscription(self, subscription_id: str) -> bool:
        """Remove subscription from topic"""
        with self.lock:
            for i, sub in enumerate(self.subscriptions):
                if sub.id == subscription_id:
                    del self.subscriptions[i]
                    return True
            return False
    
    def get_matching_subscriptions(self, message: PubSubMessage) -> List[Subscription]:
        """Get subscriptions that match the message"""
        with self.lock:
            return [sub for sub in self.subscriptions 
                   if sub.active and sub.matches_message(message)]
    
    def add_retained_message(self, message: PubSubMessage) -> None:
        """Add message to retained messages"""
        if message.retain:
            with self.lock:
                self.retained_messages.append(message)
    
    def get_retained_messages(self) -> List[PubSubMessage]:
        """Get all retained messages for new subscribers"""
        with self.lock:
            return list(self.retained_messages)
    
    def update_stats(self, published: int = 0, delivered: int = 0, failed: int = 0) -> None:
        """Update topic statistics"""
        with self.lock:
            self.total_published += published
            self.total_delivered += delivered
            self.total_failed += failed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get topic statistics"""
        with self.lock:
            return {
                'name': self.name,
                'subscribers': len(self.subscriptions),
                'retained_messages': len(self.retained_messages),
                'total_published': self.total_published,
                'total_delivered': self.total_delivered,
                'total_failed': self.total_failed,
                'created_at': self.created_at
            }

class MessageBroker:
    """Message broker for publish-subscribe communication"""
    
    def __init__(self, max_concurrent_deliveries: int = 100):
        self.topics: Dict[str, Topic] = {}
        self.subscriptions: Dict[str, Subscription] = {}  # subscription_id -> subscription
        self.subscribers: Dict[str, Subscriber] = {}       # subscriber_id -> subscriber
        
        # Delivery infrastructure
        self.delivery_queue: deque = deque()
        self.max_concurrent_deliveries = max_concurrent_deliveries
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_deliveries)
        
        # Broker state
        self.running = False
        self.delivery_workers = 4
        
        # Statistics
        self.stats = {
            'messages_published': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'active_subscriptions': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
    
    def create_topic(self, topic_name: str, max_retained: int = 1000) -> Topic:
        """Create or get existing topic"""
        with self.lock:
            if topic_name not in self.topics:
                self.topics[topic_name] = Topic(topic_name, max_retained)
                self.logger.info(f"Topic created: {topic_name}")
            return self.topics[topic_name]
    
    def delete_topic(self, topic_name: str) -> bool:
        """Delete topic and all its subscriptions"""
        with self.lock:
            if topic_name in self.topics:
                # Remove all subscriptions for this topic
                topic = self.topics[topic_name]
                for subscription in topic.subscriptions:
                    self.subscriptions.pop(subscription.id, None)
                
                del self.topics[topic_name]
                self.logger.info(f"Topic deleted: {topic_name}")
                return True
            return False
    
    def register_subscriber(self, subscriber: Subscriber) -> None:
        """Register subscriber with broker"""
        subscriber_id = subscriber.get_subscriber_id()
        self.subscribers[subscriber_id] = subscriber
        self.logger.info(f"Subscriber registered: {subscriber_id}")
    
    def unregister_subscriber(self, subscriber_id: str) -> None:
        """Unregister subscriber and remove all subscriptions"""
        # Remove subscriber
        self.subscribers.pop(subscriber_id, None)
        
        # Remove all subscriptions for this subscriber
        to_remove = []
        for subscription_id, subscription in self.subscriptions.items():
            if subscription.subscriber_id == subscriber_id:
                to_remove.append(subscription_id)
        
        for subscription_id in to_remove:
            self.unsubscribe(subscription_id)
        
        self.logger.info(f"Subscriber unregistered: {subscriber_id}")
    
    def subscribe(self, subscriber_id: str, topic_pattern: str,
                  subscription_type: SubscriptionType = SubscriptionType.TOPIC_EXACT,
                  content_filter: Optional[ContentFilter] = None,
                  delivery_mode: DeliveryMode = DeliveryMode.FIRE_AND_FORGET) -> str:
        """Subscribe to topic pattern"""
        
        subscription = Subscription(
            subscriber_id=subscriber_id,
            topic_pattern=topic_pattern,
            subscription_type=subscription_type,
            content_filter=content_filter,
            delivery_mode=delivery_mode
        )
        
        self.subscriptions[subscription.id] = subscription
        
        # Add to matching topics
        for topic_name, topic in self.topics.items():
            if subscription.matches_topic(topic_name):
                topic.add_subscription(subscription)
                
                # Deliver retained messages if any
                if subscription_type == SubscriptionType.TOPIC_EXACT:
                    retained_messages = topic.get_retained_messages()
                    for message in retained_messages:
                        if subscription.matches_message(message):
                            self.delivery_queue.append((subscription, message))
        
        self.stats['active_subscriptions'] = len(self.subscriptions)
        self.logger.info(f"Subscription created: {subscription.id} for pattern: {topic_pattern}")
        
        return subscription.id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove subscription"""
        if subscription_id not in self.subscriptions:
            return False
        
        subscription = self.subscriptions[subscription_id]
        
        # Remove from all topics
        for topic in self.topics.values():
            topic.remove_subscription(subscription_id)
        
        del self.subscriptions[subscription_id]
        self.stats['active_subscriptions'] = len(self.subscriptions)
        
        self.logger.info(f"Subscription removed: {subscription_id}")
        return True
    
    async def publish(self, topic_name: str, payload: Dict[str, Any],
                     publisher_id: str, **kwargs) -> str:
        """Publish message to topic"""
        
        # Create message
        message = PubSubMessage(
            id="",
            topic=topic_name,
            payload=payload,
            publisher_id=publisher_id,
            **kwargs
        )
        
        # Ensure topic exists
        topic = self.create_topic(topic_name)
        
        # Add to retained messages if needed
        topic.add_retained_message(message)
        
        # Find all matching subscriptions across all topics
        matching_subscriptions = []
        
        for subscription in self.subscriptions.values():
            if subscription.matches_message(message):
                matching_subscriptions.append(subscription)
        
        # Queue for delivery
        for subscription in matching_subscriptions:
            self.delivery_queue.append((subscription, message))
        
        # Update statistics
        self.stats['messages_published'] += 1
        topic.update_stats(published=1)
        
        self.logger.info(f"Message published to {topic_name}: {len(matching_subscriptions)} recipients")
        
        return message.id
    
    async def start(self) -> None:
        """Start message broker"""
        self.running = True
        self.logger.info("Message broker started")
        
        # Start delivery workers
        for i in range(self.delivery_workers):
            asyncio.create_task(self._delivery_worker(f"worker-{i}"))
    
    async def stop(self) -> None:
        """Stop message broker"""
        self.running = False
        self.executor.shutdown(wait=True)
        self.logger.info("Message broker stopped")
    
    async def _delivery_worker(self, worker_id: str) -> None:
        """Background delivery worker"""
        while self.running:
            try:
                if self.delivery_queue:
                    subscription, message = self.delivery_queue.popleft()
                    await self._deliver_message(subscription, message)
                else:
                    await asyncio.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Delivery worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)
    
    async def _deliver_message(self, subscription: Subscription, message: PubSubMessage) -> None:
        """Deliver message to subscriber"""
        subscriber_id = subscription.subscriber_id
        
        if subscriber_id not in self.subscribers:
            self.logger.warning(f"Subscriber not found: {subscriber_id}")
            return
        
        # Check rate limiting
        if not subscription.can_deliver_now():
            # Re-queue for later delivery
            self.delivery_queue.append((subscription, message))
            return
        
        subscriber = self.subscribers[subscriber_id]
        
        try:
            # Attempt delivery
            success = await subscriber.handle_message(message)
            
            if success:
                subscription.record_delivery()
                message.increment_delivery()
                self.stats['messages_delivered'] += 1
                
                # Update topic stats
                if message.topic in self.topics:
                    self.topics[message.topic].update_stats(delivered=1)
                
            else:
                message.record_delivery_failure(subscriber_id)
                self.stats['messages_failed'] += 1
                
                # Update topic stats
                if message.topic in self.topics:
                    self.topics[message.topic].update_stats(failed=1)
                
                # Retry logic based on delivery mode
                if (subscription.delivery_mode == DeliveryMode.AT_LEAST_ONCE and
                    not message.is_expired()):
                    self.delivery_queue.append((subscription, message))
        
        except Exception as e:
            self.logger.error(f"Delivery error to {subscriber_id}: {e}")
            self.stats['messages_failed'] += 1
    
    def get_broker_stats(self) -> Dict[str, Any]:
        """Get comprehensive broker statistics"""
        topic_stats = {name: topic.get_stats() for name, topic in self.topics.items()}
        
        return {
            'broker_stats': self.stats,
            'topic_count': len(self.topics),
            'subscriber_count': len(self.subscribers),
            'subscription_count': len(self.subscriptions),
            'topic_details': topic_stats,
            'delivery_queue_size': len(self.delivery_queue)
        }

class PubSubSystem:
    """
    Complete publish-subscribe system
    
    EXAMPLE USAGE:
    =============
    # Create pub-sub system
    pubsub = PubSubSystem()
    await pubsub.start()
    
    # Register subscribers
    news_subscriber = NewsSubscriber()
    pubsub.register_subscriber(news_subscriber)
    
    # Subscribe to topics
    pubsub.subscribe("news_service", "news.sports.*")
    
    # Publish messages
    await pubsub.publish("news.sports.football", 
                        {"headline": "Team wins championship!"})
    """
    
    def __init__(self):
        self.broker = MessageBroker()
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start pub-sub system"""
        await self.broker.start()
        self.logger.info("Pub-Sub system started")
    
    async def stop(self) -> None:
        """Stop pub-sub system"""
        await self.broker.stop()
        self.logger.info("Pub-Sub system stopped")
    
    def register_subscriber(self, subscriber: Subscriber) -> None:
        """Register subscriber"""
        self.broker.register_subscriber(subscriber)
    
    def unregister_subscriber(self, subscriber_id: str) -> None:
        """Unregister subscriber"""
        self.broker.unregister_subscriber(subscriber_id)
    
    def subscribe(self, subscriber_id: str, topic_pattern: str,
                  subscription_type: SubscriptionType = SubscriptionType.TOPIC_EXACT,
                  content_filter: Dict[str, Any] = None) -> str:
        """Subscribe to topic with optional content filtering"""
        
        filter_obj = ContentFilter(content_filter) if content_filter else None
        
        return self.broker.subscribe(
            subscriber_id=subscriber_id,
            topic_pattern=topic_pattern,
            subscription_type=subscription_type,
            content_filter=filter_obj
        )
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from topic"""
        return self.broker.unsubscribe(subscription_id)
    
    async def publish(self, topic: str, payload: Dict[str, Any], 
                     publisher_id: str = "system", **kwargs) -> str:
        """Publish message to topic"""
        return await self.broker.publish(topic, payload, publisher_id, **kwargs)
    
    def create_topic(self, topic_name: str) -> None:
        """Create topic explicitly"""
        self.broker.create_topic(topic_name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return self.broker.get_broker_stats()

# Example subscribers for demonstrations
class NewsSubscriber(Subscriber):
    """Subscriber for news updates"""
    
    def __init__(self, subscriber_id: str = "news_reader"):
        self.subscriber_id = subscriber_id
        self.received_news: List[Dict[str, Any]] = []
    
    async def handle_message(self, message: PubSubMessage) -> bool:
        """Handle news message"""
        try:
            news_item = {
                'topic': message.topic,
                'headline': message.payload.get('headline', 'No headline'),
                'content': message.payload.get('content', ''),
                'timestamp': message.timestamp,
                'publisher': message.publisher_id
            }
            
            self.received_news.append(news_item)
            
            print(f"📰 NEWS: {news_item['headline']} (from {news_item['publisher']})")
            return True
            
        except Exception as e:
            print(f"Error processing news: {e}")
            return False
    
    def get_subscriber_id(self) -> str:
        return self.subscriber_id

class AlertSubscriber(Subscriber):
    """Subscriber for system alerts"""
    
    def __init__(self, subscriber_id: str = "alert_monitor"):
        self.subscriber_id = subscriber_id
        self.alerts_received: List[Dict[str, Any]] = []
    
    async def handle_message(self, message: PubSubMessage) -> bool:
        """Handle alert message"""
        try:
            alert = {
                'topic': message.topic,
                'level': message.payload.get('level', 'INFO'),
                'message': message.payload.get('message', ''),
                'source': message.publisher_id,
                'timestamp': message.timestamp
            }
            
            self.alerts_received.append(alert)
            
            level_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "NORMAL": "ℹ️", "LOW": "💡"}
            emoji = level_emoji.get(alert['level'], "📢")
            
            print(f"{emoji} ALERT [{alert['level']}]: {alert['message']} (from {alert['source']})")
            return True
            
        except Exception as e:
            print(f"Error processing alert: {e}")
            return False
    
    def get_subscriber_id(self) -> str:
        return self.subscriber_id

class MetricsSubscriber(Subscriber):
    """Subscriber for system metrics"""
    
    def __init__(self, subscriber_id: str = "metrics_collector"):
        self.subscriber_id = subscriber_id
        self.metrics: Dict[str, List[float]] = defaultdict(list)
    
    async def handle_message(self, message: PubSubMessage) -> bool:
        """Handle metrics message"""
        try:
            metric_name = message.payload.get('metric')
            metric_value = message.payload.get('value')
            
            if metric_name and metric_value is not None:
                self.metrics[metric_name].append(float(metric_value))
                
                print(f"📊 METRIC: {metric_name} = {metric_value}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error processing metric: {e}")
            return False
    
    def get_subscriber_id(self) -> str:
        return self.subscriber_id

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_pubsub():
    """Demo: Basic publish-subscribe communication"""
    print("\nDEMO 1: BASIC PUBLISH-SUBSCRIBE")
    print("=" * 50)
    
    # Create pub-sub system
    pubsub = PubSubSystem()
    await pubsub.start()
    
    # Register subscribers
    news_subscriber = NewsSubscriber("news_reader_1")
    pubsub.register_subscriber(news_subscriber)
    
    # Subscribe to news topics
    pubsub.subscribe("news_reader_1", "news.sports")
    pubsub.subscribe("news_reader_1", "news.tech")
    
    # Publish news
    news_items = [
        ("news.sports", {"headline": "Local team wins championship!", "content": "Great victory!"}),
        ("news.tech", {"headline": "New AI breakthrough announced", "content": "Revolutionary technology"}),
        ("news.politics", {"headline": "Election results", "content": "Candidate wins"}),  # Not subscribed
        ("news.sports", {"headline": "Transfer news", "content": "Player moves to new team"})
    ]
    
    print("Publishing news items:")
    for topic, payload in news_items:
        await pubsub.publish(topic, payload, "news_agency")
        print(f"  Published to {topic}: {payload['headline']}")
    
    # Wait for delivery
    await asyncio.sleep(1.0)
    
    print(f"\nReceived news items: {len(news_subscriber.received_news)}")
    for news in news_subscriber.received_news:
        print(f"  - {news['headline']} (topic: {news['topic']})")
    
    await pubsub.stop()

async def demo_pattern_subscriptions():
    """Demo: Pattern-based topic subscriptions"""
    print("\nDEMO 2: PATTERN-BASED SUBSCRIPTIONS")
    print("=" * 50)
    
    pubsub = PubSubSystem()
    await pubsub.start()
    
    # Register multiple subscribers with different patterns
    subscribers = [
        (NewsSubscriber("sports_fan"), "sports.*", SubscriptionType.WILDCARD),
        (NewsSubscriber("tech_enthusiast"), "tech.**", SubscriptionType.HIERARCHICAL),
        (AlertSubscriber("system_admin"), "alert.*", SubscriptionType.WILDCARD),
        (NewsSubscriber("all_news"), "news.*", SubscriptionType.WILDCARD)
    ]
    
    for subscriber, pattern, sub_type in subscribers:
        pubsub.register_subscriber(subscriber)
        pubsub.subscribe(subscriber.get_subscriber_id(), pattern, sub_type)
        print(f"Subscribed {subscriber.get_subscriber_id()} to pattern: {pattern}")
    
    # Publish to various topics
    publications = [
        ("sports.football", {"headline": "Championship final tonight", "league": "Premier"}),
        ("sports.basketball", {"headline": "New season starts", "league": "NBA"}),
        ("tech.ai.breakthrough", {"headline": "AI solves protein folding", "impact": "high"}),
        ("tech.mobile.release", {"headline": "New smartphone launched", "price": 999}),
        ("alert.system.cpu", {"level": "HIGH", "message": "CPU usage at 95%"}),
        ("alert.security.breach", {"level": "CRITICAL", "message": "Unauthorized access detected"}),
        ("news.weather", {"headline": "Storm warning issued", "severity": "moderate"})
    ]
    
    print(f"\nPublishing {len(publications)} messages:")
    for topic, payload in publications:
        await pubsub.publish(topic, payload, "content_system")
        print(f"  Published to {topic}")
    
    # Wait for delivery
    await asyncio.sleep(1.0)
    
    # Show what each subscriber received
    print(f"\nDelivery results:")
    for subscriber, pattern, sub_type in subscribers:
        if isinstance(subscriber, NewsSubscriber):
            count = len(subscriber.received_news)
            print(f"  {subscriber.get_subscriber_id()}: {count} news items")
        elif isinstance(subscriber, AlertSubscriber):
            count = len(subscriber.alerts_received)
            print(f"  {subscriber.get_subscriber_id()}: {count} alerts")
    
    await pubsub.stop()

async def demo_content_filtering():
    """Demo: Content-based message filtering"""
    print("\nDEMO 3: CONTENT-BASED FILTERING")
    print("=" * 50)
    
    pubsub = PubSubSystem()
    await pubsub.start()
    
    # Register subscribers with content filters
    high_priority_subscriber = AlertSubscriber("high_priority_monitor")
    critical_alerts_subscriber = AlertSubscriber("critical_alerts")
    tech_ai_subscriber = NewsSubscriber("ai_researcher")
    
    pubsub.register_subscriber(high_priority_subscriber)
    pubsub.register_subscriber(critical_alerts_subscriber)
    pubsub.register_subscriber(tech_ai_subscriber)
    
    # Subscribe with content filters
    pubsub.subscribe("high_priority_monitor", "alert.*",
                    content_filter={"payload.level": ["HIGH", "CRITICAL"]})
    
    pubsub.subscribe("critical_alerts", "alert.*",
                    content_filter={"payload.level": "CRITICAL"})
    
    pubsub.subscribe("ai_researcher", "tech.*",
                    content_filter={"payload.category": "AI"})
    
    print("Subscribers registered with content filters:")
    print("  - high_priority_monitor: HIGH and CRITICAL alerts only")
    print("  - critical_alerts: CRITICAL alerts only")
    print("  - ai_researcher: AI-related tech news only")
    
    # Publish messages with varying content
    messages = [
        ("alert.system", {"level": "LOW", "message": "Disk space 80% full"}),
        ("alert.system", {"level": "HIGH", "message": "Memory usage critical"}),
        ("alert.security", {"level": "CRITICAL", "message": "Security breach detected"}),
        ("tech.mobile", {"headline": "New phone released", "category": "Hardware"}),
        ("tech.ai", {"headline": "Neural network breakthrough", "category": "AI"}),
        ("tech.blockchain", {"headline": "New crypto protocol", "category": "Blockchain"}),
        ("alert.network", {"level": "NORMAL", "message": "Network latency high"}),
        ("tech.ai", {"headline": "AI ethics guidelines published", "category": "AI"})
    ]
    
    print(f"\nPublishing {len(messages)} messages:")
    for topic, payload in messages:
        await pubsub.publish(topic, payload, "monitoring_system")
        level = payload.get('level', 'N/A')
        category = payload.get('category', 'N/A')
        print(f"  {topic}: level={level}, category={category}")
    
    # Wait for delivery
    await asyncio.sleep(1.0)
    
    # Show filtered results
    print(f"\nFiltered delivery results:")
    print(f"  high_priority_monitor: {len(high_priority_subscriber.alerts_received)} alerts")
    print(f"  critical_alerts: {len(critical_alerts_subscriber.alerts_received)} alerts")
    print(f"  ai_researcher: {len(tech_ai_subscriber.received_news)} tech items")
    
    await pubsub.stop()

async def demo_retained_messages():
    """Demo: Retained messages for new subscribers"""
    print("\nDEMO 4: RETAINED MESSAGES")
    print("=" * 50)
    
    pubsub = PubSubSystem()
    await pubsub.start()
    
    # Publish retained messages before any subscribers
    print("Publishing retained messages before subscribers join:")
    
    retained_news = [
        {"headline": "Breaking: Major announcement", "importance": "high"},
        {"headline": "Weather alert: Storm approaching", "importance": "medium"},
        {"headline": "Sports: Championship results", "importance": "low"}
    ]
    
    for i, news in enumerate(retained_news):
        await pubsub.publish("news.breaking", news, "news_service", 
                           retain=True, priority=MessagePriority.HIGH)
        print(f"  Published retained message {i+1}")
    
    # Wait a moment
    await asyncio.sleep(0.5)
    
    # Now register subscriber
    print("\nRegistering new subscriber...")
    late_subscriber = NewsSubscriber("late_joiner")
    pubsub.register_subscriber(late_subscriber)
    pubsub.subscribe("late_joiner", "news.breaking")
    
    # Wait for retained message delivery
    await asyncio.sleep(1.0)
    
    print(f"\nLate subscriber received {len(late_subscriber.received_news)} retained messages:")
    for news in late_subscriber.received_news:
        print(f"  - {news['headline']}")
    
    # Publish new message
    print("\nPublishing new message:")
    await pubsub.publish("news.breaking", 
                        {"headline": "Live: New development", "importance": "high"}, 
                        "news_service")
    
    await asyncio.sleep(0.5)
    
    print(f"Total messages received: {len(late_subscriber.received_news)}")
    
    await pubsub.stop()

async def demo_scalable_fanout():
    """Demo: Scalable fan-out to many subscribers"""
    print("\nDEMO 5: SCALABLE FAN-OUT")
    print("=" * 50)
    
    pubsub = PubSubSystem()
    await pubsub.start()
    
    # Create many subscribers
    num_subscribers = 50
    subscribers = []
    
    print(f"Creating {num_subscribers} subscribers...")
    
    for i in range(num_subscribers):
        subscriber = NewsSubscriber(f"subscriber_{i}")
        subscribers.append(subscriber)
        pubsub.register_subscriber(subscriber)
        
        # Subscribe to different topic patterns
        if i % 3 == 0:
            pubsub.subscribe(f"subscriber_{i}", "news.*", SubscriptionType.WILDCARD)
        elif i % 3 == 1:
            pubsub.subscribe(f"subscriber_{i}", "alert.*", SubscriptionType.WILDCARD)
        else:
            pubsub.subscribe(f"subscriber_{i}", "metrics.*", SubscriptionType.WILDCARD)
    
    # Publish to multiple topics simultaneously
    print(f"\nPublishing to multiple topics for fan-out:")
    
    start_time = time.time()
    
    topics_to_publish = [
        ("news.global", {"headline": "Global news update", "region": "worldwide"}),
        ("alert.system", {"level": "HIGH", "message": "System maintenance starting"}),
        ("metrics.performance", {"metric": "response_time", "value": 150})
    ]
    
    for topic, payload in topics_to_publish:
        await pubsub.publish(topic, payload, "broadcast_system")
        print(f"  Published to {topic}")
    
    # Wait for all deliveries
    await asyncio.sleep(2.0)
    
    delivery_time = time.time() - start_time
    
    # Count total deliveries
    total_deliveries = 0
    for subscriber in subscribers:
        if hasattr(subscriber, 'received_news'):
            total_deliveries += len(subscriber.received_news)
    
    # Show statistics
    stats = pubsub.get_statistics()
    
    print(f"\nFan-out Performance:")
    print(f"  - Subscribers: {num_subscribers}")
    print(f"  - Messages published: {len(topics_to_publish)}")
    print(f"  - Total deliveries: {total_deliveries}")
    print(f"  - Delivery time: {delivery_time:.2f}s")
    print(f"  - Deliveries per second: {total_deliveries/delivery_time:.0f}")
    print(f"  - Broker stats: {stats['broker_stats']}")
    
    await pubsub.stop()

async def main():
    """
    Demonstrate Publish-Subscribe Patterns for scalable agent broadcasting
    
    WHAT YOU'LL LEARN:
    ================
    1. How to design scalable broadcast communication systems
    2. How to implement topic-based and content-based filtering
    3. How to handle pattern matching and hierarchical topics
    4. How to build reliable message delivery with fan-out
    5. How to manage dynamic subscriber lists efficiently
    
    REAL WORLD APPLICATIONS:
    =======================
    - News and content distribution systems
    - Real-time monitoring and alerting platforms
    - IoT device coordination and management
    - Social media feeds and notifications
    - Financial market data distribution
    - Distributed system event coordination
    """
    
    print("PUBLISH-SUBSCRIBE PATTERNS DEMONSTRATION")
    print("Showing how agents broadcast and receive at massive scale!")
    
    await demo_basic_pubsub()
    await demo_pattern_subscriptions()
    await demo_content_filtering()
    await demo_retained_messages()
    await demo_scalable_fanout()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Pub-Sub enables massive fan-out communication")
    print("✓ Topic patterns provide flexible subscription models")
    print("✓ Content filtering allows selective message delivery")
    print("✓ Retained messages help late-joining subscribers")
    print("✓ Decoupling publishers and subscribers improves scalability")
    print("✓ Pattern matching supports complex routing scenarios")
    print("\nTHE POWER OF PUBLISH-SUBSCRIBE:")
    print("- Scales to millions of publishers and subscribers")
    print("- Enables real-time data streaming and broadcasting")
    print("- Provides natural load distribution across subscribers")
    print("- Supports dynamic system reconfiguration")

if __name__ == "__main__":
    asyncio.run(main())
