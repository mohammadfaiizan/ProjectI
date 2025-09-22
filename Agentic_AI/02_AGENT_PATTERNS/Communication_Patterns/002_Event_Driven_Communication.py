#!/usr/bin/env python3
"""
Event-Driven Communication: Reactive Agent Coordination
======================================================

WHAT IS THE PROBLEM?
==================
Traditional request-response patterns create tight coupling and don't scale:
- Agents must know exactly who to call
- Synchronous calls block execution
- Hard to add new agents to workflow
- Difficult to handle dynamic scenarios
- No way to react to state changes automatically

Example: Rigid Workflow Chaos
TRADITIONAL APPROACH (Brittle):
- Order agent calls payment agent directly
- Payment agent calls inventory agent directly
- Inventory agent calls shipping agent directly
- Adding fraud detection requires changing all agents
- If any agent fails, entire workflow breaks

REAL WORLD EXAMPLE:
=================
How does a modern e-commerce system handle orders?

AMAZON ORDER PROCESSING:
When you place an order, events flow naturally:
1. ORDER_PLACED event triggers payment processing
2. PAYMENT_CONFIRMED event triggers inventory check
3. INVENTORY_RESERVED event triggers fraud detection
4. FRAUD_CHECK_PASSED event triggers fulfillment
5. ITEM_SHIPPED event triggers customer notification
6. DELIVERY_CONFIRMED event triggers review request

BENEFITS:
- Loose coupling: services don't need to know each other
- Easy to add new features (just subscribe to events)
- Automatic failure handling and retries
- Real-time responsiveness to state changes
- Scales to millions of events per second

THE ALGORITHM:
=============
1. PUBLISH: Agent publishes event when something happens
2. DISCOVER: Event system finds all interested subscribers
3. ROUTE: Events delivered to relevant handlers
4. PROCESS: Each subscriber processes event independently
5. REACT: Subscribers may publish new events as reactions
6. CASCADE: Event chains create complex workflows
7. MONITOR: System tracks event flows and performance

EVENT PATTERNS:
- Domain Events: Business state changes
- Integration Events: Cross-system communication
- System Events: Infrastructure notifications
- Command Events: Action requests

WHY IS THIS REVOLUTIONARY?
========================
- Enables reactive, responsive systems
- Supports complex workflows without tight coupling
- Allows dynamic system evolution
- Provides natural audit trails
- Scales horizontally across many agents
- Enables real-time processing at massive scale
"""

import asyncio
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Set, Type, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import weakref
import inspect

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class EventType(Enum):
    """Types of events in the system"""
    DOMAIN_EVENT = "domain_event"           # Business state changes
    INTEGRATION_EVENT = "integration_event" # Cross-system communication
    SYSTEM_EVENT = "system_event"          # Infrastructure events
    COMMAND_EVENT = "command_event"        # Action requests
    NOTIFICATION_EVENT = "notification_event" # Informational events

class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Event:
    """Core event structure for event-driven communication"""
    id: str
    event_type: str
    source: str
    timestamp: float = field(default_factory=time.time)
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Event properties
    priority: EventPriority = EventPriority.NORMAL
    category: EventType = EventType.DOMAIN_EVENT
    version: str = "1.0"
    
    # Routing and filtering
    tags: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None  # ID of event that caused this one
    
    # Delivery control
    ttl: Optional[float] = None          # Time to live
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        """Initialize event with defaults"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
    
    def is_expired(self) -> bool:
        """Check if event has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def can_retry(self) -> bool:
        """Check if event can be retried"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry counter"""
        self.retry_count += 1
    
    def add_tag(self, tag: str) -> None:
        """Add tag to event"""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def has_tag(self, tag: str) -> bool:
        """Check if event has specific tag"""
        return tag in self.tags
    
    def serialize(self) -> str:
        """Serialize event to JSON"""
        data = asdict(self)
        data['category'] = self.category.value
        data['priority'] = self.priority.value
        return json.dumps(data)
    
    @classmethod
    def deserialize(cls, data: str) -> 'Event':
        """Deserialize event from JSON"""
        obj = json.loads(data)
        obj['category'] = EventType(obj['category'])
        obj['priority'] = EventPriority(obj['priority'])
        return cls(**obj)

class EventHandler(ABC):
    """Abstract base class for event handlers"""
    
    @abstractmethod
    async def handle(self, event: Event) -> Optional[List[Event]]:
        """Handle event and optionally return new events"""
        pass
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can process the event"""
        pass
    
    @abstractmethod
    def get_subscriptions(self) -> List[str]:
        """Get list of event types this handler subscribes to"""
        pass

class EventFilter:
    """Filter for event subscription"""
    
    def __init__(self, event_types: List[str] = None, 
                 tags: List[str] = None,
                 source_patterns: List[str] = None,
                 priority_threshold: EventPriority = EventPriority.LOW):
        self.event_types = event_types or []
        self.tags = tags or []
        self.source_patterns = source_patterns or []
        self.priority_threshold = priority_threshold
    
    def matches(self, event: Event) -> bool:
        """Check if event matches this filter"""
        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        # Check tags
        if self.tags and not any(tag in event.tags for tag in self.tags):
            return False
        
        # Check source patterns
        if self.source_patterns:
            matches_source = any(
                pattern in event.source for pattern in self.source_patterns
            )
            if not matches_source:
                return False
        
        # Check priority
        if event.priority.value < self.priority_threshold.value:
            return False
        
        return True

class EventSubscription:
    """Event subscription with handler and filter"""
    
    def __init__(self, handler: EventHandler, 
                 event_filter: Optional[EventFilter] = None,
                 subscriber_id: str = None):
        self.handler = handler
        self.filter = event_filter or EventFilter()
        self.subscriber_id = subscriber_id or str(uuid.uuid4())
        self.created_at = time.time()
        self.message_count = 0
        self.last_processed = None
        self.active = True
    
    def matches_event(self, event: Event) -> bool:
        """Check if subscription matches event"""
        return self.active and self.filter.matches(event)
    
    async def deliver_event(self, event: Event) -> Optional[List[Event]]:
        """Deliver event to handler"""
        if not self.matches_event(event):
            return None
        
        try:
            result = await self.handler.handle(event)
            self.message_count += 1
            self.last_processed = time.time()
            return result
        except Exception as e:
            logging.error(f"Error in event handler {self.subscriber_id}: {e}")
            return None

class EventStore:
    """Store for event persistence and replay"""
    
    def __init__(self, max_events: int = 100000):
        self.events: Dict[str, Event] = {}
        self.event_streams: Dict[str, List[str]] = defaultdict(list)  # source -> event_ids
        self.event_types: Dict[str, List[str]] = defaultdict(list)    # type -> event_ids
        self.max_events = max_events
        self.lock = threading.Lock()
    
    def store_event(self, event: Event) -> None:
        """Store event persistently"""
        with self.lock:
            # Remove oldest events if at capacity
            if len(self.events) >= self.max_events:
                self._cleanup_old_events()
            
            self.events[event.id] = event
            self.event_streams[event.source].append(event.id)
            self.event_types[event.event_type].append(event.id)
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get event by ID"""
        return self.events.get(event_id)
    
    def get_events_by_source(self, source: str, limit: int = 100) -> List[Event]:
        """Get events from specific source"""
        event_ids = self.event_streams[source][-limit:]
        return [self.events[eid] for eid in event_ids if eid in self.events]
    
    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[Event]:
        """Get events of specific type"""
        event_ids = self.event_types[event_type][-limit:]
        return [self.events[eid] for eid in event_ids if eid in self.events]
    
    def get_events_by_correlation(self, correlation_id: str) -> List[Event]:
        """Get all events with same correlation ID"""
        return [event for event in self.events.values() 
                if event.correlation_id == correlation_id]
    
    def _cleanup_old_events(self) -> None:
        """Remove oldest 10% of events"""
        sorted_events = sorted(self.events.values(), key=lambda e: e.timestamp)
        cleanup_count = len(sorted_events) // 10
        
        for event in sorted_events[:cleanup_count]:
            del self.events[event.id]
            self.event_streams[event.source].remove(event.id)
            self.event_types[event.event_type].remove(event.id)

class EventBus:
    """Event bus for publishing and subscribing to events"""
    
    def __init__(self, buffer_size: int = 10000):
        self.subscriptions: List[EventSubscription] = []
        self.event_buffer: deque = deque(maxlen=buffer_size)
        self.event_store = EventStore()
        
        # Processing state
        self.running = False
        self.worker_count = 4
        self.executor = ThreadPoolExecutor(max_workers=self.worker_count)
        
        # Statistics
        self.stats = {
            'events_published': 0,
            'events_delivered': 0,
            'events_failed': 0,
            'handlers_active': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def subscribe(self, handler: EventHandler, 
                  event_filter: Optional[EventFilter] = None) -> str:
        """Subscribe handler to events"""
        subscription = EventSubscription(handler, event_filter)
        self.subscriptions.append(subscription)
        self.stats['handlers_active'] = len(self.subscriptions)
        
        self.logger.info(f"Handler subscribed: {subscription.subscriber_id}")
        return subscription.subscriber_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe handler"""
        for i, subscription in enumerate(self.subscriptions):
            if subscription.subscriber_id == subscription_id:
                del self.subscriptions[i]
                self.stats['handlers_active'] = len(self.subscriptions)
                self.logger.info(f"Handler unsubscribed: {subscription_id}")
                return True
        return False
    
    async def publish(self, event: Event) -> None:
        """Publish event to the bus"""
        # Store event
        self.event_store.store_event(event)
        
        # Add to processing buffer
        self.event_buffer.append(event)
        self.stats['events_published'] += 1
        
        self.logger.info(f"Event published: {event.event_type} from {event.source}")
    
    async def publish_multiple(self, events: List[Event]) -> None:
        """Publish multiple events atomically"""
        for event in events:
            await self.publish(event)
    
    async def start_processing(self) -> None:
        """Start event processing"""
        self.running = True
        self.logger.info("Event bus started processing")
        
        # Start processing tasks
        for i in range(self.worker_count):
            asyncio.create_task(self._event_processor(f"worker-{i}"))
    
    async def stop_processing(self) -> None:
        """Stop event processing"""
        self.running = False
        self.executor.shutdown(wait=True)
        self.logger.info("Event bus stopped processing")
    
    async def _event_processor(self, worker_id: str) -> None:
        """Background event processor"""
        while self.running:
            try:
                if self.event_buffer:
                    event = self.event_buffer.popleft()
                    await self._process_event(event)
                else:
                    await asyncio.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Event processor {worker_id} error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_event(self, event: Event) -> None:
        """Process single event through all matching subscriptions"""
        matching_subscriptions = [
            sub for sub in self.subscriptions 
            if sub.matches_event(event)
        ]
        
        if not matching_subscriptions:
            self.logger.debug(f"No handlers for event: {event.event_type}")
            return
        
        # Process event through all matching handlers
        for subscription in matching_subscriptions:
            try:
                new_events = await subscription.deliver_event(event)
                self.stats['events_delivered'] += 1
                
                # Publish any new events generated by handler
                if new_events:
                    for new_event in new_events:
                        # Set causation chain
                        new_event.causation_id = event.id
                        new_event.correlation_id = event.correlation_id
                        await self.publish(new_event)
                
            except Exception as e:
                self.stats['events_failed'] += 1
                self.logger.error(f"Handler error for event {event.id}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            **self.stats,
            'events_buffered': len(self.event_buffer),
            'events_stored': len(self.event_store.events),
            'active_subscriptions': len(self.subscriptions)
        }
    
    def get_events_by_correlation(self, correlation_id: str) -> List[Event]:
        """Get event chain by correlation ID"""
        return self.event_store.get_events_by_correlation(correlation_id)

class EventDrivenSystem:
    """
    Complete event-driven communication system
    
    EXAMPLE USAGE:
    =============
    # Create event system
    system = EventDrivenSystem()
    await system.start()
    
    # Register handlers
    order_handler = OrderProcessingHandler()
    system.register_handler(order_handler)
    
    # Publish events
    event = Event(
        id="",
        event_type="order.placed",
        source="order_service",
        data={"order_id": "12345", "customer_id": "67890"}
    )
    
    await system.publish_event(event)
    """
    
    def __init__(self):
        self.event_bus = EventBus()
        self.handlers: Dict[str, EventHandler] = {}
        self.agent_subscriptions: Dict[str, List[str]] = defaultdict(list)  # agent_id -> subscription_ids
        
        # System monitoring
        self.event_metrics: Dict[str, int] = defaultdict(int)
        self.handler_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self) -> None:
        """Start the event-driven system"""
        await self.event_bus.start_processing()
        self.logger.info("Event-driven system started")
    
    async def stop(self) -> None:
        """Stop the event-driven system"""
        await self.event_bus.stop_processing()
        self.logger.info("Event-driven system stopped")
    
    def register_handler(self, handler: EventHandler, agent_id: str = None) -> str:
        """Register event handler"""
        handler_id = str(uuid.uuid4())
        self.handlers[handler_id] = handler
        
        # Create filter from handler subscriptions
        event_types = handler.get_subscriptions()
        event_filter = EventFilter(event_types=event_types)
        
        # Subscribe to event bus
        subscription_id = self.event_bus.subscribe(handler, event_filter)
        
        if agent_id:
            self.agent_subscriptions[agent_id].append(subscription_id)
        
        self.logger.info(f"Handler registered: {handler_id} for types: {event_types}")
        return handler_id
    
    def unregister_handler(self, handler_id: str) -> bool:
        """Unregister event handler"""
        if handler_id in self.handlers:
            # Find and remove subscriptions
            handler = self.handlers[handler_id]
            for agent_id, subscriptions in self.agent_subscriptions.items():
                subscriptions[:] = [s for s in subscriptions if s != handler_id]
            
            del self.handlers[handler_id]
            self.logger.info(f"Handler unregistered: {handler_id}")
            return True
        return False
    
    async def publish_event(self, event: Event) -> None:
        """Publish event to the system"""
        await self.event_bus.publish(event)
        self.event_metrics[event.event_type] += 1
    
    async def publish_domain_event(self, source: str, event_type: str, 
                                  data: Dict[str, Any], 
                                  correlation_id: str = None) -> Event:
        """Publish domain event with convenience method"""
        event = Event(
            id="",
            event_type=event_type,
            source=source,
            data=data,
            category=EventType.DOMAIN_EVENT,
            correlation_id=correlation_id
        )
        
        await self.publish_event(event)
        return event
    
    async def publish_command_event(self, source: str, command: str, 
                                   target: str, data: Dict[str, Any]) -> Event:
        """Publish command event"""
        event = Event(
            id="",
            event_type=f"command.{command}",
            source=source,
            data={**data, "target": target},
            category=EventType.COMMAND_EVENT,
            priority=EventPriority.HIGH
        )
        
        await self.publish_event(event)
        return event
    
    def get_event_chain(self, correlation_id: str) -> List[Event]:
        """Get complete event chain by correlation ID"""
        return self.event_bus.get_events_by_correlation(correlation_id)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            'event_bus_stats': self.event_bus.get_statistics(),
            'event_type_counts': dict(self.event_metrics),
            'total_handlers': len(self.handlers),
            'total_agents': len(self.agent_subscriptions)
        }

# Example event handlers for common patterns
class OrderProcessingHandler(EventHandler):
    """Handler for order processing events"""
    
    def __init__(self, agent_id: str = "order_processor"):
        self.agent_id = agent_id
        self.processed_orders: Set[str] = set()
    
    async def handle(self, event: Event) -> Optional[List[Event]]:
        """Handle order-related events"""
        new_events = []
        
        if event.event_type == "order.placed":
            order_id = event.data.get("order_id")
            customer_id = event.data.get("customer_id")
            
            # Process order
            self.processed_orders.add(order_id)
            
            # Generate payment request event
            payment_event = Event(
                id="",
                event_type="payment.requested",
                source=self.agent_id,
                data={
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "amount": event.data.get("amount", 100.0)
                }
            )
            new_events.append(payment_event)
            
            print(f"  Order {order_id} processed, payment requested")
        
        elif event.event_type == "payment.confirmed":
            order_id = event.data.get("order_id")
            
            # Generate inventory check event
            inventory_event = Event(
                id="",
                event_type="inventory.check_requested",
                source=self.agent_id,
                data={
                    "order_id": order_id,
                    "items": event.data.get("items", [])
                }
            )
            new_events.append(inventory_event)
            
            print(f"  Payment confirmed for order {order_id}, checking inventory")
        
        return new_events if new_events else None
    
    def can_handle(self, event: Event) -> bool:
        """Can handle order and payment events"""
        return event.event_type in ["order.placed", "payment.confirmed"]
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to order and payment events"""
        return ["order.placed", "payment.confirmed"]

class PaymentHandler(EventHandler):
    """Handler for payment processing"""
    
    def __init__(self, agent_id: str = "payment_processor"):
        self.agent_id = agent_id
        self.processed_payments: Dict[str, float] = {}
    
    async def handle(self, event: Event) -> Optional[List[Event]]:
        """Handle payment events"""
        if event.event_type == "payment.requested":
            order_id = event.data.get("order_id")
            amount = event.data.get("amount")
            
            # Simulate payment processing
            await asyncio.sleep(0.1)  # Processing delay
            
            self.processed_payments[order_id] = amount
            
            # Generate payment confirmation
            confirmation_event = Event(
                id="",
                event_type="payment.confirmed",
                source=self.agent_id,
                data={
                    "order_id": order_id,
                    "amount": amount,
                    "transaction_id": str(uuid.uuid4())
                }
            )
            
            print(f"  Payment of ${amount} processed for order {order_id}")
            return [confirmation_event]
        
        return None
    
    def can_handle(self, event: Event) -> bool:
        """Can handle payment request events"""
        return event.event_type == "payment.requested"
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to payment request events"""
        return ["payment.requested"]

class InventoryHandler(EventHandler):
    """Handler for inventory management"""
    
    def __init__(self, agent_id: str = "inventory_manager"):
        self.agent_id = agent_id
        self.inventory = {"item1": 100, "item2": 50, "item3": 200}
    
    async def handle(self, event: Event) -> Optional[List[Event]]:
        """Handle inventory events"""
        if event.event_type == "inventory.check_requested":
            order_id = event.data.get("order_id")
            items = event.data.get("items", ["item1"])  # Default item
            
            # Check inventory availability
            available = all(self.inventory.get(item, 0) > 0 for item in items)
            
            if available:
                # Reserve items
                for item in items:
                    self.inventory[item] -= 1
                
                # Generate confirmation
                confirmation_event = Event(
                    id="",
                    event_type="inventory.reserved",
                    source=self.agent_id,
                    data={
                        "order_id": order_id,
                        "items": items,
                        "status": "reserved"
                    }
                )
                
                print(f"  Inventory reserved for order {order_id}: {items}")
                return [confirmation_event]
            else:
                # Generate out of stock event
                out_of_stock_event = Event(
                    id="",
                    event_type="inventory.out_of_stock",
                    source=self.agent_id,
                    data={
                        "order_id": order_id,
                        "items": items,
                        "status": "unavailable"
                    }
                )
                
                print(f"  Inventory unavailable for order {order_id}: {items}")
                return [out_of_stock_event]
        
        return None
    
    def can_handle(self, event: Event) -> bool:
        """Can handle inventory check events"""
        return event.event_type == "inventory.check_requested"
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to inventory events"""
        return ["inventory.check_requested"]

class NotificationHandler(EventHandler):
    """Handler for sending notifications"""
    
    def __init__(self, agent_id: str = "notification_service"):
        self.agent_id = agent_id
        self.notifications_sent: List[Dict[str, Any]] = []
    
    async def handle(self, event: Event) -> Optional[List[Event]]:
        """Handle notification events"""
        notification_data = None
        
        if event.event_type == "inventory.reserved":
            order_id = event.data.get("order_id")
            notification_data = {
                "type": "order_confirmed",
                "message": f"Your order {order_id} has been confirmed and items reserved",
                "order_id": order_id
            }
        
        elif event.event_type == "inventory.out_of_stock":
            order_id = event.data.get("order_id")
            notification_data = {
                "type": "order_cancelled",
                "message": f"Your order {order_id} has been cancelled due to insufficient inventory",
                "order_id": order_id
            }
        
        if notification_data:
            self.notifications_sent.append(notification_data)
            print(f"  📧 Notification sent: {notification_data['message']}")
        
        return None
    
    def can_handle(self, event: Event) -> bool:
        """Can handle inventory result events"""
        return event.event_type in ["inventory.reserved", "inventory.out_of_stock"]
    
    def get_subscriptions(self) -> List[str]:
        """Subscribe to inventory result events"""
        return ["inventory.reserved", "inventory.out_of_stock"]

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_event_flow():
    """Demo: Basic event-driven workflow"""
    print("\nDEMO 1: BASIC EVENT-DRIVEN WORKFLOW")
    print("=" * 50)
    
    # Create event system
    system = EventDrivenSystem()
    await system.start()
    
    # Register handlers
    order_handler = OrderProcessingHandler()
    payment_handler = PaymentHandler()
    
    system.register_handler(order_handler, "order_service")
    system.register_handler(payment_handler, "payment_service")
    
    # Publish order placed event
    order_event = Event(
        id="",
        event_type="order.placed",
        source="e_commerce_app",
        data={
            "order_id": "ORD-12345",
            "customer_id": "CUST-67890",
            "amount": 99.99,
            "items": ["laptop", "mouse"]
        }
    )
    
    print(f"Publishing event: {order_event.event_type}")
    print(f"Order data: {order_event.data}")
    
    await system.publish_event(order_event)
    
    # Wait for processing
    await asyncio.sleep(1.0)
    
    # Show metrics
    metrics = system.get_system_metrics()
    print(f"Events processed: {metrics['event_bus_stats']['events_delivered']}")
    print(f"Event types: {list(metrics['event_type_counts'].keys())}")
    
    await system.stop()

async def demo_complete_order_workflow():
    """Demo: Complete e-commerce order workflow"""
    print("\nDEMO 2: COMPLETE ORDER WORKFLOW")
    print("=" * 50)
    
    system = EventDrivenSystem()
    await system.start()
    
    # Register all handlers
    handlers = [
        OrderProcessingHandler(),
        PaymentHandler(),
        InventoryHandler(),
        NotificationHandler()
    ]
    
    for handler in handlers:
        system.register_handler(handler)
    
    # Process multiple orders
    orders = [
        {"order_id": "ORD-001", "customer_id": "CUST-001", "amount": 150.00, "items": ["item1"]},
        {"order_id": "ORD-002", "customer_id": "CUST-002", "amount": 75.50, "items": ["item2"]},
        {"order_id": "ORD-003", "customer_id": "CUST-003", "amount": 200.00, "items": ["item999"]}  # Out of stock
    ]
    
    print("Processing orders through event-driven workflow:")
    
    for order_data in orders:
        print(f"\n--- Processing Order {order_data['order_id']} ---")
        
        order_event = Event(
            id="",
            event_type="order.placed",
            source="online_store",
            data=order_data
        )
        
        await system.publish_event(order_event)
        
        # Wait for workflow completion
        await asyncio.sleep(0.5)
    
    # Show final metrics
    await asyncio.sleep(1.0)
    metrics = system.get_system_metrics()
    print(f"\nFinal Statistics:")
    print(f"- Total events: {metrics['event_bus_stats']['events_published']}")
    print(f"- Events delivered: {metrics['event_bus_stats']['events_delivered']}")
    print(f"- Event types processed: {len(metrics['event_type_counts'])}")
    
    await system.stop()

async def demo_event_correlation():
    """Demo: Event correlation and tracking"""
    print("\nDEMO 3: EVENT CORRELATION AND TRACKING")
    print("=" * 50)
    
    system = EventDrivenSystem()
    await system.start()
    
    # Register handlers
    system.register_handler(OrderProcessingHandler())
    system.register_handler(PaymentHandler())
    system.register_handler(InventoryHandler())
    
    # Create order with specific correlation ID
    correlation_id = str(uuid.uuid4())
    
    order_event = Event(
        id="",
        event_type="order.placed",
        source="mobile_app",
        data={"order_id": "ORD-TRACE", "customer_id": "CUST-TRACE", "amount": 125.00},
        correlation_id=correlation_id
    )
    
    print(f"Starting order workflow with correlation ID: {correlation_id[:8]}...")
    await system.publish_event(order_event)
    
    # Wait for processing
    await asyncio.sleep(1.0)
    
    # Get complete event chain
    event_chain = system.get_event_chain(correlation_id)
    
    print(f"\nEvent Chain ({len(event_chain)} events):")
    for i, event in enumerate(event_chain, 1):
        causation = f" (caused by: {event.causation_id[:8]}...)" if event.causation_id else ""
        print(f"  {i}. {event.event_type} from {event.source}{causation}")
        print(f"     Time: {time.strftime('%H:%M:%S', time.localtime(event.timestamp))}")
        print(f"     Data: {event.data}")
    
    await system.stop()

async def demo_event_filtering():
    """Demo: Event filtering and selective subscription"""
    print("\nDEMO 4: EVENT FILTERING AND SUBSCRIPTION")
    print("=" * 50)
    
    system = EventDrivenSystem()
    await system.start()
    
    # Create specialized handlers with filters
    class HighPriorityHandler(EventHandler):
        def __init__(self):
            self.handled_events = []
        
        async def handle(self, event: Event) -> Optional[List[Event]]:
            self.handled_events.append(event.event_type)
            print(f"  🔥 HIGH PRIORITY: {event.event_type} from {event.source}")
            return None
        
        def can_handle(self, event: Event) -> bool:
            return event.priority == EventPriority.HIGH
        
        def get_subscriptions(self) -> List[str]:
            return []  # Handle all types, filtered by priority
    
    class PaymentEventsHandler(EventHandler):
        def __init__(self):
            self.handled_events = []
        
        async def handle(self, event: Event) -> Optional[List[Event]]:
            self.handled_events.append(event.event_type)
            print(f"  💳 PAYMENT: {event.event_type} - {event.data}")
            return None
        
        def can_handle(self, event: Event) -> bool:
            return "payment" in event.event_type
        
        def get_subscriptions(self) -> List[str]:
            return ["payment.requested", "payment.confirmed", "payment.failed"]
    
    # Register handlers
    high_priority_handler = HighPriorityHandler()
    payment_handler = PaymentEventsHandler()
    
    system.register_handler(high_priority_handler)
    system.register_handler(payment_handler)
    
    # Publish various events
    events = [
        Event("", "order.placed", "app", data={"order_id": "1"}, priority=EventPriority.NORMAL),
        Event("", "payment.requested", "order_service", data={"amount": 100}, priority=EventPriority.HIGH),
        Event("", "inventory.check", "inventory", data={"item": "widget"}, priority=EventPriority.LOW),
        Event("", "payment.confirmed", "payment_gateway", data={"transaction": "TX123"}, priority=EventPriority.NORMAL),
        Event("", "system.alert", "monitoring", data={"alert": "disk_full"}, priority=EventPriority.CRITICAL)
    ]
    
    print("Publishing events with different priorities and types:")
    
    for event in events:
        print(f"Publishing: {event.event_type} (priority: {event.priority.name})")
        await system.publish_event(event)
    
    # Wait for processing
    await asyncio.sleep(1.0)
    
    print(f"\nHigh priority handler processed: {len(high_priority_handler.handled_events)} events")
    print(f"Payment handler processed: {len(payment_handler.handled_events)} events")
    
    await system.stop()

async def demo_reactive_scaling():
    """Demo: Reactive system scaling based on events"""
    print("\nDEMO 5: REACTIVE SYSTEM SCALING")
    print("=" * 50)
    
    system = EventDrivenSystem()
    await system.start()
    
    # Create load monitoring handler
    class LoadMonitor(EventHandler):
        def __init__(self):
            self.event_counts = defaultdict(int)
            self.alert_threshold = 3
        
        async def handle(self, event: Event) -> Optional[List[Event]]:
            self.event_counts[event.event_type] += 1
            
            # Check if we need to scale up
            if self.event_counts[event.event_type] >= self.alert_threshold:
                scale_event = Event(
                    id="",
                    event_type="system.scale_up_requested",
                    source="load_monitor",
                    data={
                        "service": event.source,
                        "event_type": event.event_type,
                        "current_load": self.event_counts[event.event_type]
                    },
                    priority=EventPriority.HIGH
                )
                
                # Reset counter after scaling request
                self.event_counts[event.event_type] = 0
                
                print(f"  📈 SCALING: Requesting scale-up for {event.source}")
                return [scale_event]
            
            return None
        
        def can_handle(self, event: Event) -> bool:
            return not event.event_type.startswith("system.")
        
        def get_subscriptions(self) -> List[str]:
            return []  # Monitor all non-system events
    
    class ScalingController(EventHandler):
        def __init__(self):
            self.instance_counts = defaultdict(int)
        
        async def handle(self, event: Event) -> Optional[List[Event]]:
            if event.event_type == "system.scale_up_requested":
                service = event.data.get("service")
                self.instance_counts[service] += 1
                
                print(f"  🚀 SCALED: {service} now has {self.instance_counts[service]} instances")
                
                # Confirm scaling
                confirm_event = Event(
                    id="",
                    event_type="system.scaled",
                    source="scaling_controller",
                    data={
                        "service": service,
                        "new_instance_count": self.instance_counts[service]
                    }
                )
                return [confirm_event]
            
            return None
        
        def can_handle(self, event: Event) -> bool:
            return event.event_type == "system.scale_up_requested"
        
        def get_subscriptions(self) -> List[str]:
            return ["system.scale_up_requested"]
    
    # Register monitoring handlers
    system.register_handler(LoadMonitor())
    system.register_handler(ScalingController())
    
    # Simulate high load
    print("Simulating high load on payment service:")
    
    for i in range(5):
        payment_event = Event(
            id="",
            event_type="payment.processed",
            source="payment_service",
            data={"transaction_id": f"TX{i+1}"}
        )
        
        await system.publish_event(payment_event)
        await asyncio.sleep(0.2)
    
    # Wait for scaling events
    await asyncio.sleep(1.0)
    
    await system.stop()

async def main():
    """
    Demonstrate Event-Driven Communication for reactive agent systems
    
    WHAT YOU'LL LEARN:
    ================
    1. How to design reactive, event-driven agent systems
    2. How to implement loose coupling through events
    3. How to handle complex workflows with event chains
    4. How to build scalable, responsive communication
    5. How to track and correlate events across system
    
    REAL WORLD APPLICATIONS:
    =======================
    - E-commerce order processing workflows
    - Microservices coordination and integration
    - Real-time monitoring and alerting systems
    - IoT device management and automation
    - Financial trading and risk management
    - Supply chain and logistics coordination
    """
    
    print("EVENT-DRIVEN COMMUNICATION DEMONSTRATION")
    print("Showing how agents react and coordinate through events!")
    
    await demo_basic_event_flow()
    await demo_complete_order_workflow()
    await demo_event_correlation()
    await demo_event_filtering()
    await demo_reactive_scaling()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Events enable loose coupling between agents")
    print("✓ Event chains create complex workflows naturally")
    print("✓ Reactive systems respond automatically to changes")
    print("✓ Event correlation provides complete traceability")
    print("✓ Filtering allows selective event processing")
    print("✓ Event-driven architecture scales horizontally")
    print("\nTHE POWER OF EVENT-DRIVEN SYSTEMS:")
    print("- Enables real-time reactive behavior")
    print("- Provides natural audit trails and monitoring")
    print("- Supports dynamic system evolution")
    print("- Scales to millions of events per second")

if __name__ == "__main__":
    asyncio.run(main())
