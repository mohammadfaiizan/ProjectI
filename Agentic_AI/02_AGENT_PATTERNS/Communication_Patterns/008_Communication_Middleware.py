#!/usr/bin/env python3
"""
Communication Middleware: Agent Infrastructure Layer
===================================================

WHAT IS THE PROBLEM?
==================
Building agent communication from scratch is complex and error-prone:
- Reinventing networking, serialization, and routing for every project
- Managing different communication patterns in a unified way
- Handling failures, retries, and error recovery consistently
- Scaling communication infrastructure across many agents
- Providing service discovery and load balancing

Example: Manual Communication Nightmare
BUILDING FROM SCRATCH (Painful):
- Agent A writes custom TCP socket code
- Agent B uses different message format
- No automatic retry when messages fail
- Manual service discovery via hardcoded IPs
- Each agent implements its own error handling
- Different timeouts and reliability guarantees

REAL WORLD EXAMPLE:
=================
How does Slack handle billions of messages daily?

SLACK'S MIDDLEWARE ARCHITECTURE:
Message Infrastructure:
1. API GATEWAY: Single entry point for all communication
2. MESSAGE ROUTER: Routes messages to appropriate services
3. RELIABILITY LAYER: Handles retries, deduplication, ordering
4. PROTOCOL ADAPTER: Supports WebSocket, HTTP, mobile protocols
5. LOAD BALANCER: Distributes load across service instances
6. SERVICE DISCOVERY: Automatic finding and health checking

BENEFITS:
- Developers focus on business logic, not networking
- Consistent reliability and error handling across all services
- Automatic scaling and load distribution
- Support for multiple protocols and devices
- Centralized monitoring and debugging

THE ARCHITECTURE:
================
1. ABSTRACTION: Hide low-level networking complexity
2. STANDARDIZATION: Common interfaces for all communication
3. RELIABILITY: Built-in retry, timeout, and error handling
4. SCALABILITY: Automatic load balancing and service discovery
5. OBSERVABILITY: Logging, metrics, and tracing throughout
6. FLEXIBILITY: Support multiple protocols and patterns
7. SECURITY: Authentication, encryption, and authorization

MIDDLEWARE LAYERS:
- Transport Layer: TCP, UDP, WebSocket, HTTP protocols
- Serialization Layer: JSON, Protocol Buffers, MessagePack
- Routing Layer: Service discovery, load balancing, failover
- Reliability Layer: Retries, circuit breakers, timeouts
- Security Layer: Authentication, encryption, authorization
- Observability Layer: Logging, metrics, distributed tracing

WHY IS THIS ESSENTIAL?
====================
- Reduces development time by 80% for distributed systems
- Provides consistent reliability and performance guarantees
- Enables rapid scaling from prototype to production
- Standardizes communication patterns across entire organization
- Provides foundation for microservices and agent architectures
- Essential for modern cloud-native applications
"""

import asyncio
import time
import json
import uuid
import logging
import inspect
from typing import Dict, List, Any, Optional, Callable, Union, Type, Protocol
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
from collections import defaultdict, deque
import ssl
import hashlib
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TransportType(Enum):
    """Types of transport protocols"""
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    HTTP = "http"
    MEMORY = "memory"              # In-memory for testing
    REDIS = "redis"                # Redis pub/sub
    AMQP = "amqp"                 # RabbitMQ/AMQP

class SerializationType(Enum):
    """Message serialization formats"""
    JSON = "json"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    PICKLE = "pickle"

class RoutingStrategy(Enum):
    """Message routing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    CONSISTENT_HASH = "consistent_hash"
    GEOGRAPHIC = "geographic"
    RANDOM = "random"

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

@dataclass
class MiddlewareMessage:
    """Standardized message format for middleware"""
    id: str
    source_service: str
    target_service: str
    method: str
    
    # Message content
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Middleware metadata
    timestamp: float = field(default_factory=time.time)
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Routing and delivery
    routing_key: str = ""
    reply_to: Optional[str] = None
    ttl: Optional[float] = None
    
    # Quality of Service
    delivery_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    
    # Tracing and observability
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize message with defaults"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())
        if not self.routing_key:
            self.routing_key = f"{self.source_service}.{self.target_service}.{self.method}"
    
    def increment_delivery_count(self) -> None:
        """Increment delivery attempt counter"""
        self.delivery_count += 1
    
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.delivery_count < self.max_retries
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def add_header(self, key: str, value: str) -> None:
        """Add header to message"""
        self.headers[key] = value
    
    def get_header(self, key: str, default: str = None) -> Optional[str]:
        """Get header value"""
        return self.headers.get(key, default)

@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    service_id: str
    host: str
    port: int
    protocol: TransportType
    
    # Service metadata
    version: str = "1.0.0"
    weight: int = 1                        # Load balancing weight
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Health and status
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: float = field(default_factory=time.time)
    response_time: float = 0.0
    error_rate: float = 0.0
    
    # Connection tracking
    active_connections: int = 0
    total_requests: int = 0
    
    def get_address(self) -> str:
        """Get full address of endpoint"""
        return f"{self.protocol.value}://{self.host}:{self.port}"
    
    def is_healthy(self, max_age: float = 60.0) -> bool:
        """Check if endpoint is healthy"""
        is_recent = time.time() - self.last_health_check < max_age
        return self.status == ServiceStatus.HEALTHY and is_recent
    
    def update_health(self, status: ServiceStatus, response_time: float = 0.0) -> None:
        """Update health status"""
        self.status = status
        self.last_health_check = time.time()
        if response_time > 0:
            # Exponential moving average
            self.response_time = (self.response_time * 0.8) + (response_time * 0.2)

class MessageSerializer(ABC):
    """Abstract message serializer"""
    
    @abstractmethod
    def serialize(self, message: MiddlewareMessage) -> bytes:
        """Serialize message to bytes"""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> MiddlewareMessage:
        """Deserialize bytes to message"""
        pass

class JSONSerializer(MessageSerializer):
    """JSON message serializer"""
    
    def serialize(self, message: MiddlewareMessage) -> bytes:
        """Serialize message to JSON bytes"""
        data = asdict(message)
        data['priority'] = message.priority.value
        return json.dumps(data).encode('utf-8')
    
    def deserialize(self, data: bytes) -> MiddlewareMessage:
        """Deserialize JSON bytes to message"""
        obj = json.loads(data.decode('utf-8'))
        obj['priority'] = MessagePriority(obj['priority'])
        return MiddlewareMessage(**obj)

class Transport(ABC):
    """Abstract transport layer"""
    
    @abstractmethod
    async def send(self, endpoint: ServiceEndpoint, message: MiddlewareMessage) -> bool:
        """Send message to endpoint"""
        pass
    
    @abstractmethod
    async def receive(self, timeout: float = 1.0) -> Optional[MiddlewareMessage]:
        """Receive message with timeout"""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start transport"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop transport"""
        pass

class MemoryTransport(Transport):
    """In-memory transport for testing and local communication"""
    
    def __init__(self):
        self.message_queues: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue())
        self.running = False
    
    async def start(self) -> None:
        """Start memory transport"""
        self.running = True
    
    async def stop(self) -> None:
        """Stop memory transport"""
        self.running = False
    
    async def send(self, endpoint: ServiceEndpoint, message: MiddlewareMessage) -> bool:
        """Send message via memory queue"""
        if not self.running:
            return False
        
        try:
            service_queue = self.message_queues[endpoint.service_id]
            await service_queue.put(message)
            return True
        except Exception:
            return False
    
    async def receive(self, service_id: str, timeout: float = 1.0) -> Optional[MiddlewareMessage]:
        """Receive message from service queue"""
        if not self.running:
            return None
        
        try:
            service_queue = self.message_queues[service_id]
            return await asyncio.wait_for(service_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

class ServiceRegistry:
    """Service discovery and registry"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceEndpoint]] = defaultdict(list)
        self.lock = threading.Lock()
        
        # Health checking
        self.health_check_interval = 30.0
        self.health_check_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self) -> None:
        """Start service registry"""
        self.running = True
        self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self) -> None:
        """Stop service registry"""
        self.running = False
        if self.health_check_task:
            self.health_check_task.cancel()
    
    def register_service(self, endpoint: ServiceEndpoint) -> None:
        """Register service endpoint"""
        with self.lock:
            service_endpoints = self.services[endpoint.service_id]
            
            # Remove existing endpoint with same host:port
            service_endpoints[:] = [
                ep for ep in service_endpoints 
                if not (ep.host == endpoint.host and ep.port == endpoint.port)
            ]
            
            # Add new endpoint
            service_endpoints.append(endpoint)
            
        logging.info(f"Registered service: {endpoint.service_id} at {endpoint.get_address()}")
    
    def unregister_service(self, service_id: str, host: str, port: int) -> None:
        """Unregister service endpoint"""
        with self.lock:
            if service_id in self.services:
                self.services[service_id][:] = [
                    ep for ep in self.services[service_id]
                    if not (ep.host == host and ep.port == port)
                ]
                
                if not self.services[service_id]:
                    del self.services[service_id]
        
        logging.info(f"Unregistered service: {service_id} at {host}:{port}")
    
    def discover_service(self, service_id: str) -> List[ServiceEndpoint]:
        """Discover healthy endpoints for service"""
        with self.lock:
            endpoints = self.services.get(service_id, [])
            return [ep for ep in endpoints if ep.is_healthy()]
    
    def get_all_services(self) -> Dict[str, List[ServiceEndpoint]]:
        """Get all registered services"""
        with self.lock:
            return {
                service_id: endpoints.copy()
                for service_id, endpoints in self.services.items()
            }
    
    async def _health_check_loop(self) -> None:
        """Periodic health checking of services"""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logging.error(f"Health check error: {e}")
                await asyncio.sleep(5.0)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all endpoints"""
        with self.lock:
            all_endpoints = [
                ep for endpoints in self.services.values() 
                for ep in endpoints
            ]
        
        for endpoint in all_endpoints:
            try:
                # Simple health check - in production, this would be a real health probe
                is_healthy = await self._check_endpoint_health(endpoint)
                status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
                endpoint.update_health(status)
            except Exception:
                endpoint.update_health(ServiceStatus.UNHEALTHY)
    
    async def _check_endpoint_health(self, endpoint: ServiceEndpoint) -> bool:
        """Check if endpoint is healthy"""
        # Simplified health check - always return True for demo
        # In production, this would make actual health check requests
        return True

class LoadBalancer:
    """Load balancer for service endpoints"""
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
    
    def select_endpoint(self, service_id: str, endpoints: List[ServiceEndpoint]) -> Optional[ServiceEndpoint]:
        """Select endpoint based on load balancing strategy"""
        if not endpoints:
            return None
        
        healthy_endpoints = [ep for ep in endpoints if ep.is_healthy()]
        if not healthy_endpoints:
            # Fall back to any endpoint if none are healthy
            healthy_endpoints = endpoints
        
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(service_id, healthy_endpoints)
        elif self.strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_endpoints)
        elif self.strategy == RoutingStrategy.WEIGHTED:
            return self._weighted_select(healthy_endpoints)
        elif self.strategy == RoutingStrategy.RANDOM:
            return self._random_select(healthy_endpoints)
        else:
            return healthy_endpoints[0]  # Default to first
    
    def _round_robin_select(self, service_id: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round-robin endpoint selection"""
        counter = self.round_robin_counters[service_id]
        selected = endpoints[counter % len(endpoints)]
        self.round_robin_counters[service_id] = counter + 1
        return selected
    
    def _least_connections_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select endpoint with least active connections"""
        return min(endpoints, key=lambda ep: ep.active_connections)
    
    def _weighted_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted random selection"""
        import random
        total_weight = sum(ep.weight for ep in endpoints)
        if total_weight == 0:
            return random.choice(endpoints)
        
        target = random.uniform(0, total_weight)
        current = 0
        
        for endpoint in endpoints:
            current += endpoint.weight
            if current >= target:
                return endpoint
        
        return endpoints[-1]  # Fallback
    
    def _random_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Random endpoint selection"""
        import random
        return random.choice(endpoints)

class MessageRouter:
    """Routes messages to appropriate services"""
    
    def __init__(self, service_registry: ServiceRegistry, load_balancer: LoadBalancer):
        self.service_registry = service_registry
        self.load_balancer = load_balancer
        self.routing_table: Dict[str, str] = {}  # routing_key -> service_id
        
        # Routing statistics
        self.stats = {
            'messages_routed': 0,
            'routing_errors': 0,
            'service_not_found': 0
        }
    
    def add_route(self, routing_key: str, service_id: str) -> None:
        """Add explicit routing rule"""
        self.routing_table[routing_key] = service_id
    
    def remove_route(self, routing_key: str) -> None:
        """Remove routing rule"""
        self.routing_table.pop(routing_key, None)
    
    async def route_message(self, message: MiddlewareMessage) -> Optional[ServiceEndpoint]:
        """Route message to appropriate endpoint"""
        self.stats['messages_routed'] += 1
        
        # Check explicit routing table first
        target_service = self.routing_table.get(message.routing_key)
        if not target_service:
            target_service = message.target_service
        
        if not target_service:
            self.stats['routing_errors'] += 1
            return None
        
        # Discover service endpoints
        endpoints = self.service_registry.discover_service(target_service)
        if not endpoints:
            self.stats['service_not_found'] += 1
            return None
        
        # Select endpoint using load balancer
        selected_endpoint = self.load_balancer.select_endpoint(target_service, endpoints)
        return selected_endpoint
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return self.stats.copy()

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        
        # Circuit state
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True
    
    def record_success(self) -> None:
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self) -> None:
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class MessageBroker:
    """Central message broker with reliability features"""
    
    def __init__(self, transport: Transport, serializer: MessageSerializer):
        self.transport = transport
        self.serializer = serializer
        self.service_registry = ServiceRegistry()
        self.load_balancer = LoadBalancer()
        self.router = MessageRouter(self.service_registry, self.load_balancer)
        
        # Reliability features
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_queue: asyncio.Queue = asyncio.Queue()
        self.dead_letter_queue: List[MiddlewareMessage] = []
        
        # Message tracking
        self.in_flight_messages: Dict[str, MiddlewareMessage] = {}
        self.message_history: deque = deque(maxlen=1000)
        
        # Workers
        self.running = False
        self.retry_worker_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'retries_attempted': 0,
            'dead_letter_count': 0
        }
        
        self.logger = logging.getLogger("MessageBroker")
    
    async def start(self) -> None:
        """Start message broker"""
        self.running = True
        
        await self.transport.start()
        await self.service_registry.start()
        
        # Start background workers
        self.retry_worker_task = asyncio.create_task(self._retry_worker())
        
        self.logger.info("Message broker started")
    
    async def stop(self) -> None:
        """Stop message broker"""
        self.running = False
        
        if self.retry_worker_task:
            self.retry_worker_task.cancel()
        
        await self.transport.stop()
        await self.service_registry.stop()
        
        self.logger.info("Message broker stopped")
    
    def register_service(self, endpoint: ServiceEndpoint) -> None:
        """Register service endpoint"""
        self.service_registry.register_service(endpoint)
        
        # Initialize circuit breaker for service
        if endpoint.service_id not in self.circuit_breakers:
            self.circuit_breakers[endpoint.service_id] = CircuitBreaker()
    
    async def send_message(self, message: MiddlewareMessage) -> bool:
        """Send message with reliability features"""
        self.stats['messages_sent'] += 1
        
        # Route message to endpoint
        endpoint = await self.router.route_message(message)
        if not endpoint:
            self.logger.error(f"No endpoint found for service: {message.target_service}")
            self.stats['messages_failed'] += 1
            return False
        
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(endpoint.service_id)
        if circuit_breaker and not circuit_breaker.can_execute():
            self.logger.warning(f"Circuit breaker open for service: {endpoint.service_id}")
            await self._handle_failed_message(message)
            return False
        
        # Track in-flight message
        self.in_flight_messages[message.id] = message
        
        try:
            # Attempt delivery
            success = await self.transport.send(endpoint, message)
            
            if success:
                self.stats['messages_delivered'] += 1
                endpoint.total_requests += 1
                
                if circuit_breaker:
                    circuit_breaker.record_success()
                
                # Remove from in-flight tracking
                self.in_flight_messages.pop(message.id, None)
                
                # Add to history
                self.message_history.append({
                    'message_id': message.id,
                    'status': 'delivered',
                    'timestamp': time.time(),
                    'endpoint': endpoint.get_address()
                })
                
                return True
            else:
                raise Exception("Transport send failed")
        
        except Exception as e:
            self.logger.error(f"Failed to send message {message.id}: {e}")
            
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            await self._handle_failed_message(message)
            return False
    
    async def _handle_failed_message(self, message: MiddlewareMessage) -> None:
        """Handle failed message delivery"""
        message.increment_delivery_count()
        
        if message.can_retry() and not message.is_expired():
            # Add to retry queue
            await self.retry_queue.put(message)
            self.logger.info(f"Queued message {message.id} for retry (attempt {message.delivery_count})")
        else:
            # Move to dead letter queue
            self.dead_letter_queue.append(message)
            self.stats['dead_letter_count'] += 1
            self.in_flight_messages.pop(message.id, None)
            
            self.logger.warning(f"Message {message.id} moved to dead letter queue")
    
    async def _retry_worker(self) -> None:
        """Background worker for message retries"""
        while self.running:
            try:
                # Get message from retry queue with timeout
                message = await asyncio.wait_for(self.retry_queue.get(), timeout=1.0)
                
                self.stats['retries_attempted'] += 1
                
                # Add exponential backoff delay
                delay = min(2 ** message.delivery_count, 60)  # Max 60 seconds
                await asyncio.sleep(delay)
                
                # Retry sending
                await self.send_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Retry worker error: {e}")
                await asyncio.sleep(1.0)
    
    def get_broker_stats(self) -> Dict[str, Any]:
        """Get comprehensive broker statistics"""
        return {
            'broker_stats': self.stats,
            'routing_stats': self.router.get_routing_stats(),
            'in_flight_messages': len(self.in_flight_messages),
            'retry_queue_size': self.retry_queue.qsize(),
            'dead_letter_count': len(self.dead_letter_queue),
            'circuit_breaker_states': {
                service_id: cb.state 
                for service_id, cb in self.circuit_breakers.items()
            },
            'registered_services': len(self.service_registry.get_all_services())
        }

class CommunicationMiddleware:
    """
    Complete communication middleware system
    
    EXAMPLE USAGE:
    =============
    # Create middleware
    middleware = CommunicationMiddleware()
    await middleware.start()
    
    # Register services
    user_service = ServiceEndpoint("user_service", "localhost", 8001, TransportType.HTTP)
    middleware.register_service(user_service)
    
    # Send message
    message = MiddlewareMessage(
        id="",
        source_service="web_app",
        target_service="user_service",
        method="get_user",
        payload={"user_id": "12345"}
    )
    
    success = await middleware.send_message(message)
    """
    
    def __init__(self, transport_type: TransportType = TransportType.MEMORY,
                 serialization_type: SerializationType = SerializationType.JSON):
        
        # Create transport and serializer
        if transport_type == TransportType.MEMORY:
            self.transport = MemoryTransport()
        else:
            raise ValueError(f"Unsupported transport: {transport_type}")
        
        if serialization_type == SerializationType.JSON:
            self.serializer = JSONSerializer()
        else:
            raise ValueError(f"Unsupported serialization: {serialization_type}")
        
        # Create message broker
        self.broker = MessageBroker(self.transport, self.serializer)
        
        # Service clients
        self.service_clients: Dict[str, 'ServiceClient'] = {}
        
        self.logger = logging.getLogger("CommunicationMiddleware")
    
    async def start(self) -> None:
        """Start middleware"""
        await self.broker.start()
        self.logger.info("Communication middleware started")
    
    async def stop(self) -> None:
        """Stop middleware"""
        await self.broker.stop()
        self.logger.info("Communication middleware stopped")
    
    def register_service(self, endpoint: ServiceEndpoint) -> None:
        """Register service endpoint"""
        self.broker.register_service(endpoint)
    
    async def send_message(self, message: MiddlewareMessage) -> bool:
        """Send message through middleware"""
        return await self.broker.send_message(message)
    
    def create_service_client(self, service_id: str) -> 'ServiceClient':
        """Create client for specific service"""
        client = ServiceClient(service_id, self)
        self.service_clients[service_id] = client
        return client
    
    def add_route(self, routing_key: str, service_id: str) -> None:
        """Add explicit routing rule"""
        self.broker.router.add_route(routing_key, service_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        return self.broker.get_broker_stats()

class ServiceClient:
    """Client wrapper for calling remote services"""
    
    def __init__(self, service_id: str, middleware: CommunicationMiddleware):
        self.service_id = service_id
        self.middleware = middleware
        self.client_id = str(uuid.uuid4())
        
        # Client statistics
        self.stats = {
            'requests_sent': 0,
            'responses_received': 0,
            'errors': 0,
            'average_response_time': 0.0
        }
        
        self.response_times: List[float] = []
    
    async def call(self, method: str, payload: Dict[str, Any] = None,
                   timeout: float = 30.0, headers: Dict[str, str] = None) -> Any:
        """Call remote service method"""
        
        start_time = time.time()
        
        # Create request message
        message = MiddlewareMessage(
            id="",
            source_service=self.client_id,
            target_service=self.service_id,
            method=method,
            payload=payload or {},
            headers=headers or {},
            timeout=timeout
        )
        
        self.stats['requests_sent'] += 1
        
        try:
            # Send message
            success = await self.middleware.send_message(message)
            
            if success:
                # In a real implementation, we would wait for response
                # For demo, we'll simulate response
                await asyncio.sleep(0.01)  # Simulate network delay
                
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                
                # Update average response time
                if self.response_times:
                    self.stats['average_response_time'] = sum(self.response_times) / len(self.response_times)
                
                self.stats['responses_received'] += 1
                
                # Return mock successful response
                return {
                    'status': 'success',
                    'data': f'Response from {self.service_id}.{method}',
                    'timestamp': time.time()
                }
            else:
                raise Exception("Failed to send message")
                
        except Exception as e:
            self.stats['errors'] += 1
            raise Exception(f"Service call failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return self.stats.copy()

# Example service implementations
class UserService:
    """Example user service"""
    
    def __init__(self, middleware: CommunicationMiddleware):
        self.middleware = middleware
        self.service_id = "user_service"
        
        # Mock user database
        self.users = {
            "user_1": {"id": "user_1", "name": "Alice", "email": "alice@example.com"},
            "user_2": {"id": "user_2", "name": "Bob", "email": "bob@example.com"},
            "user_3": {"id": "user_3", "name": "Charlie", "email": "charlie@example.com"}
        }
        
        # Register service
        endpoint = ServiceEndpoint(
            service_id=self.service_id,
            host="localhost",
            port=8001,
            protocol=TransportType.HTTP,
            tags=["user", "authentication"]
        )
        
        self.middleware.register_service(endpoint)
    
    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get user by ID"""
        await asyncio.sleep(0.05)  # Simulate database query
        
        user = self.users.get(user_id)
        if user:
            return {"status": "success", "user": user}
        else:
            return {"status": "error", "message": "User not found"}
    
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new user"""
        await asyncio.sleep(0.1)  # Simulate database write
        
        user_id = f"user_{len(self.users) + 1}"
        user = {
            "id": user_id,
            "name": user_data.get("name"),
            "email": user_data.get("email")
        }
        
        self.users[user_id] = user
        return {"status": "success", "user": user}

class OrderService:
    """Example order service"""
    
    def __init__(self, middleware: CommunicationMiddleware):
        self.middleware = middleware
        self.service_id = "order_service"
        self.user_client = middleware.create_service_client("user_service")
        
        # Mock order database
        self.orders = {}
        self.order_counter = 1
        
        # Register service
        endpoint = ServiceEndpoint(
            service_id=self.service_id,
            host="localhost",
            port=8002,
            protocol=TransportType.HTTP,
            tags=["order", "commerce"]
        )
        
        self.middleware.register_service(endpoint)
    
    async def create_order(self, user_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create new order"""
        
        # Verify user exists by calling user service
        try:
            user_response = await self.user_client.call("get_user", {"user_id": user_id})
            if user_response.get("status") != "success":
                return {"status": "error", "message": "Invalid user"}
        except Exception as e:
            return {"status": "error", "message": f"User verification failed: {e}"}
        
        # Create order
        order_id = f"order_{self.order_counter}"
        self.order_counter += 1
        
        order = {
            "id": order_id,
            "user_id": user_id,
            "items": items,
            "status": "pending",
            "created_at": time.time()
        }
        
        self.orders[order_id] = order
        
        await asyncio.sleep(0.08)  # Simulate processing
        
        return {"status": "success", "order": order}
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order by ID"""
        await asyncio.sleep(0.03)  # Simulate database query
        
        order = self.orders.get(order_id)
        if order:
            return {"status": "success", "order": order}
        else:
            return {"status": "error", "message": "Order not found"}

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_middleware():
    """Demo: Basic middleware setup and service communication"""
    print("\nDEMO 1: BASIC MIDDLEWARE COMMUNICATION")
    print("=" * 50)
    
    # Create middleware
    middleware = CommunicationMiddleware()
    await middleware.start()
    
    # Create services
    user_service = UserService(middleware)
    order_service = OrderService(middleware)
    
    print("Created user and order services")
    
    # Create client for user service
    user_client = middleware.create_service_client("user_service")
    
    # Test direct service calls
    print("\nTesting service calls:")
    
    try:
        # Get existing user
        response = await user_client.call("get_user", {"user_id": "user_1"})
        print(f"  Get user_1: {response}")
        
        # Create new user
        new_user_data = {"name": "Diana", "email": "diana@example.com"}
        response = await user_client.call("create_user", new_user_data)
        print(f"  Create user: {response}")
        
        # Try to get non-existent user
        response = await user_client.call("get_user", {"user_id": "user_999"})
        print(f"  Get non-existent user: {response}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Show client statistics
    stats = user_client.get_stats()
    print(f"\nClient statistics: {stats}")
    
    await middleware.stop()

async def demo_service_discovery():
    """Demo: Service discovery and load balancing"""
    print("\nDEMO 2: SERVICE DISCOVERY AND LOAD BALANCING")
    print("=" * 50)
    
    middleware = CommunicationMiddleware()
    await middleware.start()
    
    # Register multiple instances of the same service
    service_instances = [
        ServiceEndpoint("user_service", "localhost", 8001, TransportType.HTTP, weight=1),
        ServiceEndpoint("user_service", "localhost", 8002, TransportType.HTTP, weight=2),
        ServiceEndpoint("user_service", "localhost", 8003, TransportType.HTTP, weight=1),
    ]
    
    print(f"Registering {len(service_instances)} user service instances:")
    for instance in service_instances:
        middleware.register_service(instance)
        print(f"  - {instance.get_address()} (weight: {instance.weight})")
    
    # Get service registry state
    all_services = middleware.broker.service_registry.get_all_services()
    print(f"\nDiscovered services:")
    for service_id, endpoints in all_services.items():
        print(f"  {service_id}: {len(endpoints)} endpoints")
    
    # Test load balancing by sending multiple requests
    user_client = middleware.create_service_client("user_service")
    
    print(f"\nTesting load balancing with 10 requests:")
    for i in range(10):
        try:
            response = await user_client.call("get_user", {"user_id": f"user_{i % 3 + 1}"})
            print(f"  Request {i+1}: Success")
        except Exception as e:
            print(f"  Request {i+1}: Error - {e}")
        
        await asyncio.sleep(0.1)
    
    await middleware.stop()

async def demo_service_communication():
    """Demo: Inter-service communication"""
    print("\nDEMO 3: INTER-SERVICE COMMUNICATION")
    print("=" * 50)
    
    middleware = CommunicationMiddleware()
    await middleware.start()
    
    # Create services that communicate with each other
    user_service = UserService(middleware)
    order_service = OrderService(middleware)
    
    print("Created interconnected services")
    
    # Create order client
    order_client = middleware.create_service_client("order_service")
    
    print("\nCreating order (involves user service call):")
    
    # Create order - this will trigger user service validation
    order_items = [
        {"product": "laptop", "quantity": 1, "price": 999.99},
        {"product": "mouse", "quantity": 2, "price": 29.99}
    ]
    
    try:
        response = await order_client.call("create_order", {
            "user_id": "user_1",
            "items": order_items
        })
        
        if response.get("status") == "success":
            order = response.get("order")
            print(f"  Order created successfully:")
            print(f"    Order ID: {order['id']}")
            print(f"    User ID: {order['user_id']}")
            print(f"    Items: {len(order['items'])}")
        else:
            print(f"  Order creation failed: {response.get('message')}")
    
    except Exception as e:
        print(f"  Error creating order: {e}")
    
    # Try creating order with invalid user
    print("\nTrying to create order with invalid user:")
    try:
        response = await order_client.call("create_order", {
            "user_id": "invalid_user",
            "items": order_items
        })
        print(f"  Result: {response}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    await middleware.stop()

async def demo_reliability_features():
    """Demo: Reliability features like retries and circuit breakers"""
    print("\nDEMO 4: RELIABILITY FEATURES")
    print("=" * 50)
    
    middleware = CommunicationMiddleware()
    await middleware.start()
    
    # Register a service endpoint that will "fail"
    unreliable_service = ServiceEndpoint(
        service_id="unreliable_service",
        host="localhost",
        port=9999,  # Non-existent port
        protocol=TransportType.HTTP
    )
    
    middleware.register_service(unreliable_service)
    print("Registered unreliable service (will fail)")
    
    # Create client and attempt calls
    client = middleware.create_service_client("unreliable_service")
    
    print("\nAttempting calls to unreliable service:")
    
    for i in range(3):
        try:
            # These calls will fail and trigger retries
            response = await client.call("test_method", {"data": f"attempt_{i+1}"})
            print(f"  Attempt {i+1}: Success - {response}")
        except Exception as e:
            print(f"  Attempt {i+1}: Failed - {e}")
        
        await asyncio.sleep(1.0)  # Wait between attempts
    
    # Check circuit breaker state
    circuit_breakers = middleware.broker.circuit_breakers
    if "unreliable_service" in circuit_breakers:
        cb_state = circuit_breakers["unreliable_service"].state
        print(f"\nCircuit breaker state: {cb_state}")
    
    # Show broker statistics
    stats = middleware.get_statistics()
    print(f"\nBroker statistics:")
    print(f"  Messages sent: {stats['broker_stats']['messages_sent']}")
    print(f"  Messages failed: {stats['broker_stats']['messages_failed']}")
    print(f"  Retries attempted: {stats['broker_stats']['retries_attempted']}")
    print(f"  Dead letter count: {stats['broker_stats']['dead_letter_count']}")
    
    await middleware.stop()

async def demo_middleware_observability():
    """Demo: Observability and monitoring features"""
    print("\nDEMO 5: MIDDLEWARE OBSERVABILITY")
    print("=" * 50)
    
    middleware = CommunicationMiddleware()
    await middleware.start()
    
    # Create services
    user_service = UserService(middleware)
    order_service = OrderService(middleware)
    
    print("Services created with observability")
    
    # Generate some traffic
    user_client = middleware.create_service_client("user_service")
    order_client = middleware.create_service_client("order_service")
    
    print("\nGenerating traffic for observability demonstration:")
    
    # Multiple service calls
    for i in range(5):
        try:
            # User service calls
            await user_client.call("get_user", {"user_id": f"user_{i % 3 + 1}"})
            
            # Order service calls (which call user service internally)
            if i % 2 == 0:
                await order_client.call("create_order", {
                    "user_id": f"user_{i % 3 + 1}",
                    "items": [{"product": f"item_{i}", "quantity": 1}]
                })
            
            print(f"  Completed request batch {i+1}")
            
        except Exception as e:
            print(f"  Request batch {i+1} failed: {e}")
        
        await asyncio.sleep(0.2)
    
    # Collect comprehensive statistics
    print(f"\nMIDDLEWARE OBSERVABILITY REPORT:")
    print("=" * 40)
    
    # Broker statistics
    broker_stats = middleware.get_statistics()
    print(f"\nBroker Statistics:")
    for key, value in broker_stats['broker_stats'].items():
        print(f"  {key}: {value}")
    
    # Service registry statistics
    all_services = middleware.broker.service_registry.get_all_services()
    print(f"\nService Registry:")
    for service_id, endpoints in all_services.items():
        print(f"  {service_id}: {len(endpoints)} endpoints")
        for endpoint in endpoints:
            print(f"    - {endpoint.get_address()} ({endpoint.status.value})")
            print(f"      Requests: {endpoint.total_requests}, Response time: {endpoint.response_time:.3f}s")
    
    # Client statistics
    print(f"\nClient Statistics:")
    for service_id, client in middleware.service_clients.items():
        stats = client.get_stats()
        print(f"  {service_id}_client:")
        for key, value in stats.items():
            print(f"    {key}: {value}")
    
    # Routing statistics
    routing_stats = broker_stats['routing_stats']
    print(f"\nRouting Statistics:")
    for key, value in routing_stats.items():
        print(f"  {key}: {value}")
    
    await middleware.stop()

async def main():
    """
    Demonstrate Communication Middleware for agent infrastructure
    
    WHAT YOU'LL LEARN:
    ================
    1. How to design scalable communication middleware
    2. How to implement service discovery and load balancing
    3. How to build reliability features (retries, circuit breakers)
    4. How to create inter-service communication patterns
    5. How to add observability and monitoring to middleware
    
    REAL WORLD APPLICATIONS:
    =======================
    - Microservices communication infrastructure
    - Distributed agent system coordination
    - API gateway and service mesh implementations
    - Cloud-native application platforms
    - Enterprise service bus architectures
    - IoT device communication platforms
    """
    
    print("COMMUNICATION MIDDLEWARE DEMONSTRATION")
    print("Showing how to build scalable agent communication infrastructure!")
    
    await demo_basic_middleware()
    await demo_service_discovery()
    await demo_service_communication()
    await demo_reliability_features()
    await demo_middleware_observability()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Middleware abstracts away networking complexity")
    print("✓ Service discovery enables dynamic service location")
    print("✓ Load balancing distributes traffic efficiently")
    print("✓ Reliability features handle failures gracefully")
    print("✓ Observability provides deep system insights")
    print("✓ Standardized interfaces enable interoperability")
    print("\nTHE POWER OF COMMUNICATION MIDDLEWARE:")
    print("- Reduces development time by providing reusable infrastructure")
    print("- Enables scalable microservices and agent architectures")
    print("- Provides consistent reliability and performance guarantees")
    print("- Powers modern cloud-native and distributed applications")

if __name__ == "__main__":
    asyncio.run(main())
