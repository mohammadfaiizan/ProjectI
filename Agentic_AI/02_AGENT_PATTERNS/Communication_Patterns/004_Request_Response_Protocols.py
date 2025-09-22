#!/usr/bin/env python3
"""
Request-Response Protocols: Synchronous Agent Communication
===========================================================

WHAT IS THE PROBLEM?
==================
Agents need synchronous communication for:
- Getting immediate responses to queries
- Ensuring data consistency across operations
- Implementing transactional workflows
- Coordinating time-sensitive operations
- Building interactive agent conversations

Example: Banking Transaction Chaos
FIRE-AND-FORGET APPROACH (Dangerous):
- Transfer agent sends "move $1000" message
- No confirmation if transfer succeeded
- No way to handle insufficient funds
- Account balances become inconsistent
- Customer sees money vanish into void

REAL WORLD EXAMPLE:
=================
How does your credit card payment work?

CREDIT CARD TRANSACTION FLOW:
When you buy coffee for $5:
1. REQUEST: Merchant sends authorization request
2. ROUTE: Payment processor forwards to your bank
3. VALIDATE: Bank checks balance and fraud patterns
4. RESPONSE: Bank returns approval/decline + auth code
5. CONFIRM: Merchant receives response within 2 seconds
6. COMPLETE: Transaction completes or fails atomically

BENEFITS:
- Immediate feedback on operation success/failure
- Guaranteed response within timeout period
- Enables complex multi-step workflows
- Maintains data consistency and integrity
- Supports interactive user experiences

THE ALGORITHM:
=============
1. REQUEST: Client sends request with unique ID
2. ROUTE: System finds appropriate handler agent
3. PROCESS: Handler executes request logic
4. RESPOND: Handler returns success/failure result
5. CORRELATE: Response matched to original request
6. TIMEOUT: Handle cases where response never comes
7. RETRY: Implement retry logic for failed requests

PATTERNS:
- Synchronous: Block until response received
- Asynchronous: Continue work, handle response later  
- Streaming: Multiple responses to single request
- Batch: Multiple requests in single round-trip

WHY IS THIS ESSENTIAL?
====================
- Enables reliable distributed transactions
- Provides immediate feedback for user interactions
- Supports complex query and command operations
- Maintains system consistency and data integrity
- Powers real-time agent conversations and negotiations
"""

import asyncio
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from abc import ABC, abstractmethod
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RequestType(Enum):
    """Types of requests"""
    QUERY = "query"               # Read-only data requests
    COMMAND = "command"           # State-changing operations
    STREAM = "stream"             # Streaming responses
    BATCH = "batch"               # Multiple operations
    HEALTH_CHECK = "health_check" # Service availability check

class ResponseStatus(Enum):
    """Response status codes"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"
    INVALID_REQUEST = "invalid_request"
    SERVICE_UNAVAILABLE = "service_unavailable"

class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Request:
    """Request message for request-response communication"""
    id: str
    request_type: RequestType
    method: str                    # Operation to perform
    target_service: str           # Target agent/service
    
    # Request data
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Request metadata
    sender_id: str = ""
    timestamp: float = field(default_factory=time.time)
    priority: RequestPriority = RequestPriority.NORMAL
    
    # Control parameters
    timeout: float = 30.0         # Request timeout in seconds
    retry_count: int = 0
    max_retries: int = 3
    correlation_id: Optional[str] = None
    
    # Callback tracking
    callback_address: Optional[str] = None
    
    def __post_init__(self):
        """Initialize request with defaults"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
    
    def is_expired(self) -> bool:
        """Check if request has timed out"""
        return time.time() - self.timestamp > self.timeout
    
    def can_retry(self) -> bool:
        """Check if request can be retried"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry counter"""
        self.retry_count += 1
    
    def serialize(self) -> str:
        """Serialize request to JSON"""
        data = asdict(self)
        data['request_type'] = self.request_type.value
        data['priority'] = self.priority.value
        return json.dumps(data)
    
    @classmethod
    def deserialize(cls, data: str) -> 'Request':
        """Deserialize request from JSON"""
        obj = json.loads(data)
        obj['request_type'] = RequestType(obj['request_type'])
        obj['priority'] = RequestPriority(obj['priority'])
        return cls(**obj)

@dataclass
class Response:
    """Response message for request-response communication"""
    request_id: str
    status: ResponseStatus
    
    # Response data
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # Response metadata
    responder_id: str = ""
    timestamp: float = field(default_factory=time.time)
    processing_time: float = 0.0  # Time taken to process request
    
    # Additional info
    headers: Dict[str, str] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize response"""
        if self.status == ResponseStatus.SUCCESS and self.error_message:
            # Inconsistent state - fix it
            self.status = ResponseStatus.ERROR
    
    def is_success(self) -> bool:
        """Check if response indicates success"""
        return self.status == ResponseStatus.SUCCESS
    
    def is_error(self) -> bool:
        """Check if response indicates error"""
        return self.status == ResponseStatus.ERROR
    
    def serialize(self) -> str:
        """Serialize response to JSON"""
        data = asdict(self)
        data['status'] = self.status.value
        return json.dumps(data)
    
    @classmethod
    def deserialize(cls, data: str) -> 'Response':
        """Deserialize response from JSON"""
        obj = json.loads(data)
        obj['status'] = ResponseStatus(obj['status'])
        return cls(**obj)

class RequestHandler(ABC):
    """Abstract base class for request handlers"""
    
    @abstractmethod
    async def handle_request(self, request: Request) -> Response:
        """Handle request and return response"""
        pass
    
    @abstractmethod
    def can_handle(self, request: Request) -> bool:
        """Check if this handler can process the request"""
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """Get the service name this handler represents"""
        pass

class PendingRequest:
    """Tracks a pending request waiting for response"""
    
    def __init__(self, request: Request):
        self.request = request
        self.future: asyncio.Future = asyncio.Future()
        self.created_at = time.time()
        self.retries = 0
    
    def is_expired(self) -> bool:
        """Check if request has expired"""
        return time.time() - self.created_at > self.request.timeout
    
    def set_response(self, response: Response) -> None:
        """Set the response for this request"""
        if not self.future.done():
            self.future.set_result(response)
    
    def set_timeout(self) -> None:
        """Mark request as timed out"""
        if not self.future.done():
            timeout_response = Response(
                request_id=self.request.id,
                status=ResponseStatus.TIMEOUT,
                error_message=f"Request timed out after {self.request.timeout}s",
                correlation_id=self.request.correlation_id
            )
            self.future.set_result(timeout_response)
    
    def set_error(self, error: Exception) -> None:
        """Set an error for this request"""
        if not self.future.done():
            error_response = Response(
                request_id=self.request.id,
                status=ResponseStatus.ERROR,
                error_message=str(error),
                correlation_id=self.request.correlation_id
            )
            self.future.set_result(error_response)

class RequestRouter:
    """Routes requests to appropriate handlers"""
    
    def __init__(self):
        self.handlers: Dict[str, RequestHandler] = {}  # service_name -> handler
        self.method_handlers: Dict[str, RequestHandler] = {}  # method -> handler
        self.load_balancers: Dict[str, List[RequestHandler]] = defaultdict(list)  # service -> handlers
        
        # Routing statistics
        self.routing_stats = defaultdict(int)
        self.handler_stats = defaultdict(lambda: defaultdict(int))
    
    def register_handler(self, handler: RequestHandler) -> None:
        """Register request handler"""
        service_name = handler.get_service_name()
        self.handlers[service_name] = handler
        self.load_balancers[service_name].append(handler)
    
    def register_method_handler(self, method: str, handler: RequestHandler) -> None:
        """Register handler for specific method"""
        self.method_handlers[method] = handler
    
    def unregister_handler(self, service_name: str) -> bool:
        """Unregister handler"""
        if service_name in self.handlers:
            handler = self.handlers[service_name]
            del self.handlers[service_name]
            
            # Remove from load balancer
            if service_name in self.load_balancers:
                self.load_balancers[service_name].remove(handler)
                if not self.load_balancers[service_name]:
                    del self.load_balancers[service_name]
            
            return True
        return False
    
    async def route_request(self, request: Request) -> Optional[RequestHandler]:
        """Find appropriate handler for request"""
        self.routing_stats['total_requests'] += 1
        
        # Try method-specific handler first
        if request.method in self.method_handlers:
            handler = self.method_handlers[request.method]
            if handler.can_handle(request):
                self.routing_stats['method_routed'] += 1
                return handler
        
        # Try service-specific handler
        if request.target_service in self.handlers:
            handler = self.handlers[request.target_service]
            if handler.can_handle(request):
                self.routing_stats['service_routed'] += 1
                return handler
        
        # Try load-balanced handlers
        if request.target_service in self.load_balancers:
            handlers = self.load_balancers[request.target_service]
            for handler in handlers:
                if handler.can_handle(request):
                    self.routing_stats['load_balanced'] += 1
                    return handler
        
        # No handler found
        self.routing_stats['no_handler'] += 1
        return None
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return dict(self.routing_stats)

class RequestResponseProtocol:
    """
    Complete request-response protocol implementation
    
    EXAMPLE USAGE:
    =============
    # Create protocol
    protocol = RequestResponseProtocol()
    await protocol.start()
    
    # Register handlers
    calculator = CalculatorHandler()
    protocol.register_handler(calculator)
    
    # Send request
    request = Request(
        id="",
        request_type=RequestType.QUERY,
        method="add",
        target_service="calculator",
        payload={"a": 5, "b": 3}
    )
    
    response = await protocol.send_request(request)
    """
    
    def __init__(self, max_concurrent_requests: int = 1000):
        self.router = RequestRouter()
        self.pending_requests: Dict[str, PendingRequest] = {}
        self.max_concurrent_requests = max_concurrent_requests
        
        # Protocol state
        self.running = False
        self.worker_count = 8
        self.executor = ThreadPoolExecutor(max_workers=self.worker_count)
        
        # Statistics
        self.stats = {
            'requests_sent': 0,
            'responses_received': 0,
            'requests_timed_out': 0,
            'requests_failed': 0,
            'average_response_time': 0.0
        }
        
        # Response time tracking
        self.response_times: List[float] = []
        self.max_response_time_samples = 1000
        
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
    
    async def start(self) -> None:
        """Start the request-response protocol"""
        self.running = True
        self.logger.info("Request-response protocol started")
        
        # Start background workers
        asyncio.create_task(self._timeout_monitor())
        asyncio.create_task(self._statistics_updater())
    
    async def stop(self) -> None:
        """Stop the protocol"""
        self.running = False
        
        # Cancel all pending requests
        for pending in self.pending_requests.values():
            pending.set_timeout()
        
        self.pending_requests.clear()
        self.executor.shutdown(wait=True)
        self.logger.info("Request-response protocol stopped")
    
    def register_handler(self, handler: RequestHandler) -> None:
        """Register request handler"""
        self.router.register_handler(handler)
        self.logger.info(f"Handler registered: {handler.get_service_name()}")
    
    def register_method_handler(self, method: str, handler: RequestHandler) -> None:
        """Register method-specific handler"""
        self.router.register_method_handler(method, handler)
        self.logger.info(f"Method handler registered: {method}")
    
    def unregister_handler(self, service_name: str) -> bool:
        """Unregister handler"""
        result = self.router.unregister_handler(service_name)
        if result:
            self.logger.info(f"Handler unregistered: {service_name}")
        return result
    
    async def send_request(self, request: Request, sender_id: str = "client") -> Response:
        """Send request and wait for response"""
        
        # Check concurrent request limit
        if len(self.pending_requests) >= self.max_concurrent_requests:
            return Response(
                request_id=request.id,
                status=ResponseStatus.SERVICE_UNAVAILABLE,
                error_message="Too many concurrent requests",
                correlation_id=request.correlation_id
            )
        
        request.sender_id = sender_id
        
        # Create pending request tracker
        pending = PendingRequest(request)
        
        with self.lock:
            self.pending_requests[request.id] = pending
        
        try:
            # Route and process request
            handler = await self.router.route_request(request)
            
            if not handler:
                return Response(
                    request_id=request.id,
                    status=ResponseStatus.NOT_FOUND,
                    error_message=f"No handler found for service: {request.target_service}",
                    correlation_id=request.correlation_id
                )
            
            # Process request asynchronously
            asyncio.create_task(self._process_request(request, handler, pending))
            
            # Wait for response or timeout
            response = await pending.future
            
            # Update statistics
            self.stats['requests_sent'] += 1
            if response.is_success():
                self.stats['responses_received'] += 1
            elif response.status == ResponseStatus.TIMEOUT:
                self.stats['requests_timed_out'] += 1
            else:
                self.stats['requests_failed'] += 1
            
            # Track response time
            response_time = time.time() - pending.created_at
            self.response_times.append(response_time)
            if len(self.response_times) > self.max_response_time_samples:
                self.response_times.pop(0)
            
            return response
            
        finally:
            # Clean up pending request
            with self.lock:
                self.pending_requests.pop(request.id, None)
    
    async def send_request_async(self, request: Request, 
                                callback: Callable[[Response], None],
                                sender_id: str = "client") -> str:
        """Send request asynchronously with callback"""
        
        async def handle_response():
            response = await self.send_request(request, sender_id)
            callback(response)
        
        asyncio.create_task(handle_response())
        return request.id
    
    async def send_batch_request(self, requests: List[Request], 
                                sender_id: str = "client") -> List[Response]:
        """Send multiple requests in parallel"""
        
        tasks = []
        for request in requests:
            task = asyncio.create_task(self.send_request(request, sender_id))
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_response = Response(
                    request_id=requests[i].id,
                    status=ResponseStatus.ERROR,
                    error_message=str(response),
                    correlation_id=requests[i].correlation_id
                )
                final_responses.append(error_response)
            else:
                final_responses.append(response)
        
        return final_responses
    
    async def _process_request(self, request: Request, handler: RequestHandler, 
                              pending: PendingRequest) -> None:
        """Process request with handler"""
        
        try:
            start_time = time.time()
            
            # Handle the request
            response = await handler.handle_request(request)
            
            # Set processing time
            response.processing_time = time.time() - start_time
            response.correlation_id = request.correlation_id
            
            # Deliver response
            pending.set_response(response)
            
        except Exception as e:
            self.logger.error(f"Error processing request {request.id}: {e}")
            pending.set_error(e)
    
    async def _timeout_monitor(self) -> None:
        """Monitor and handle request timeouts"""
        while self.running:
            try:
                current_time = time.time()
                expired_requests = []
                
                with self.lock:
                    for request_id, pending in self.pending_requests.items():
                        if pending.is_expired():
                            expired_requests.append(request_id)
                
                # Handle expired requests
                for request_id in expired_requests:
                    with self.lock:
                        pending = self.pending_requests.get(request_id)
                        if pending:
                            pending.set_timeout()
                            self.logger.warning(f"Request {request_id} timed out")
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Timeout monitor error: {e}")
                await asyncio.sleep(1.0)
    
    async def _statistics_updater(self) -> None:
        """Update rolling statistics"""
        while self.running:
            try:
                if self.response_times:
                    avg_response_time = sum(self.response_times) / len(self.response_times)
                    self.stats['average_response_time'] = avg_response_time
                
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Statistics updater error: {e}")
                await asyncio.sleep(5.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive protocol statistics"""
        with self.lock:
            pending_count = len(self.pending_requests)
        
        return {
            'protocol_stats': self.stats,
            'pending_requests': pending_count,
            'routing_stats': self.router.get_routing_stats(),
            'response_time_samples': len(self.response_times),
            'max_response_time': max(self.response_times) if self.response_times else 0,
            'min_response_time': min(self.response_times) if self.response_times else 0
        }

# Example request handlers
class CalculatorHandler(RequestHandler):
    """Handler for calculator operations"""
    
    def __init__(self):
        self.operations_count = 0
    
    async def handle_request(self, request: Request) -> Response:
        """Handle calculator requests"""
        
        try:
            method = request.method
            payload = request.payload
            
            if method == "add":
                a = payload.get('a', 0)
                b = payload.get('b', 0)
                result = a + b
                
            elif method == "subtract":
                a = payload.get('a', 0)
                b = payload.get('b', 0)
                result = a - b
                
            elif method == "multiply":
                a = payload.get('a', 0)
                b = payload.get('b', 0)
                result = a * b
                
            elif method == "divide":
                a = payload.get('a', 0)
                b = payload.get('b', 0)
                if b == 0:
                    return Response(
                        request_id=request.id,
                        status=ResponseStatus.ERROR,
                        error_message="Division by zero",
                        error_code="DIVISION_BY_ZERO"
                    )
                result = a / b
                
            else:
                return Response(
                    request_id=request.id,
                    status=ResponseStatus.INVALID_REQUEST,
                    error_message=f"Unknown operation: {method}"
                )
            
            self.operations_count += 1
            
            return Response(
                request_id=request.id,
                status=ResponseStatus.SUCCESS,
                data={'result': result, 'operation': method},
                responder_id="calculator"
            )
            
        except Exception as e:
            return Response(
                request_id=request.id,
                status=ResponseStatus.ERROR,
                error_message=str(e),
                responder_id="calculator"
            )
    
    def can_handle(self, request: Request) -> bool:
        """Can handle calculator operations"""
        return (request.target_service == "calculator" and
                request.method in ["add", "subtract", "multiply", "divide"])
    
    def get_service_name(self) -> str:
        return "calculator"

class DatabaseHandler(RequestHandler):
    """Handler for database operations"""
    
    def __init__(self):
        # Simple in-memory database
        self.data: Dict[str, Any] = {}
        self.query_count = 0
    
    async def handle_request(self, request: Request) -> Response:
        """Handle database requests"""
        
        try:
            method = request.method
            payload = request.payload
            
            if method == "get":
                key = payload.get('key')
                if key in self.data:
                    value = self.data[key]
                    return Response(
                        request_id=request.id,
                        status=ResponseStatus.SUCCESS,
                        data={'key': key, 'value': value},
                        responder_id="database"
                    )
                else:
                    return Response(
                        request_id=request.id,
                        status=ResponseStatus.NOT_FOUND,
                        error_message=f"Key not found: {key}",
                        responder_id="database"
                    )
            
            elif method == "set":
                key = payload.get('key')
                value = payload.get('value')
                self.data[key] = value
                
                return Response(
                    request_id=request.id,
                    status=ResponseStatus.SUCCESS,
                    data={'key': key, 'value': value, 'action': 'stored'},
                    responder_id="database"
                )
            
            elif method == "delete":
                key = payload.get('key')
                if key in self.data:
                    del self.data[key]
                    return Response(
                        request_id=request.id,
                        status=ResponseStatus.SUCCESS,
                        data={'key': key, 'action': 'deleted'},
                        responder_id="database"
                    )
                else:
                    return Response(
                        request_id=request.id,
                        status=ResponseStatus.NOT_FOUND,
                        error_message=f"Key not found: {key}",
                        responder_id="database"
                    )
            
            elif method == "list":
                return Response(
                    request_id=request.id,
                    status=ResponseStatus.SUCCESS,
                    data={'keys': list(self.data.keys()), 'count': len(self.data)},
                    responder_id="database"
                )
            
            else:
                return Response(
                    request_id=request.id,
                    status=ResponseStatus.INVALID_REQUEST,
                    error_message=f"Unknown operation: {method}"
                )
            
        except Exception as e:
            return Response(
                request_id=request.id,
                status=ResponseStatus.ERROR,
                error_message=str(e),
                responder_id="database"
            )
    
    def can_handle(self, request: Request) -> bool:
        """Can handle database operations"""
        return (request.target_service == "database" and
                request.method in ["get", "set", "delete", "list"])
    
    def get_service_name(self) -> str:
        return "database"

class WeatherHandler(RequestHandler):
    """Handler for weather information"""
    
    def __init__(self):
        # Mock weather data
        self.weather_data = {
            "new_york": {"temperature": 22, "condition": "sunny", "humidity": 65},
            "london": {"temperature": 15, "condition": "rainy", "humidity": 80},
            "tokyo": {"temperature": 28, "condition": "cloudy", "humidity": 70},
            "sydney": {"temperature": 18, "condition": "windy", "humidity": 55}
        }
    
    async def handle_request(self, request: Request) -> Response:
        """Handle weather requests"""
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        try:
            method = request.method
            payload = request.payload
            
            if method == "current":
                city = payload.get('city', '').lower()
                if city in self.weather_data:
                    weather = self.weather_data[city]
                    return Response(
                        request_id=request.id,
                        status=ResponseStatus.SUCCESS,
                        data={'city': city, 'weather': weather},
                        responder_id="weather_service"
                    )
                else:
                    return Response(
                        request_id=request.id,
                        status=ResponseStatus.NOT_FOUND,
                        error_message=f"Weather data not available for: {city}",
                        responder_id="weather_service"
                    )
            
            elif method == "forecast":
                city = payload.get('city', '').lower()
                days = payload.get('days', 1)
                
                if city in self.weather_data:
                    # Mock forecast data
                    base_weather = self.weather_data[city]
                    forecast = []
                    
                    for i in range(days):
                        day_weather = {
                            'day': i + 1,
                            'temperature': base_weather['temperature'] + (i - 2),
                            'condition': base_weather['condition'],
                            'humidity': base_weather['humidity'] + (i * 2)
                        }
                        forecast.append(day_weather)
                    
                    return Response(
                        request_id=request.id,
                        status=ResponseStatus.SUCCESS,
                        data={'city': city, 'forecast': forecast},
                        responder_id="weather_service"
                    )
                else:
                    return Response(
                        request_id=request.id,
                        status=ResponseStatus.NOT_FOUND,
                        error_message=f"Weather data not available for: {city}",
                        responder_id="weather_service"
                    )
            
            else:
                return Response(
                    request_id=request.id,
                    status=ResponseStatus.INVALID_REQUEST,
                    error_message=f"Unknown weather operation: {method}"
                )
                
        except Exception as e:
            return Response(
                request_id=request.id,
                status=ResponseStatus.ERROR,
                error_message=str(e),
                responder_id="weather_service"
            )
    
    def can_handle(self, request: Request) -> bool:
        """Can handle weather requests"""
        return (request.target_service == "weather" and
                request.method in ["current", "forecast"])
    
    def get_service_name(self) -> str:
        return "weather"

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_basic_request_response():
    """Demo: Basic request-response communication"""
    print("\nDEMO 1: BASIC REQUEST-RESPONSE")
    print("=" * 50)
    
    # Create protocol
    protocol = RequestResponseProtocol()
    await protocol.start()
    
    # Register handlers
    calculator = CalculatorHandler()
    protocol.register_handler(calculator)
    
    # Send calculation requests
    requests_data = [
        ("add", {"a": 10, "b": 5}),
        ("subtract", {"a": 20, "b": 8}),
        ("multiply", {"a": 7, "b": 6}),
        ("divide", {"a": 15, "b": 3}),
        ("divide", {"a": 10, "b": 0})  # Division by zero
    ]
    
    print("Sending calculation requests:")
    
    for method, payload in requests_data:
        request = Request(
            id="",
            request_type=RequestType.QUERY,
            method=method,
            target_service="calculator",
            payload=payload
        )
        
        print(f"  Request: {method}({payload})")
        response = await protocol.send_request(request)
        
        if response.is_success():
            result = response.data.get('result')
            print(f"  Response: {result}")
        else:
            print(f"  Error: {response.error_message}")
    
    await protocol.stop()

async def demo_database_operations():
    """Demo: Database operations with error handling"""
    print("\nDEMO 2: DATABASE OPERATIONS")
    print("=" * 50)
    
    protocol = RequestResponseProtocol()
    await protocol.start()
    
    # Register database handler
    database = DatabaseHandler()
    protocol.register_handler(database)
    
    # Database operations
    operations = [
        ("set", {"key": "user_1", "value": {"name": "Alice", "age": 30}}),
        ("set", {"key": "user_2", "value": {"name": "Bob", "age": 25}}),
        ("get", {"key": "user_1"}),
        ("get", {"key": "nonexistent"}),
        ("list", {}),
        ("delete", {"key": "user_2"}),
        ("list", {})
    ]
    
    print("Performing database operations:")
    
    for method, payload in operations:
        request = Request(
            id="",
            request_type=RequestType.COMMAND if method in ["set", "delete"] else RequestType.QUERY,
            method=method,
            target_service="database",
            payload=payload
        )
        
        print(f"  Operation: {method} {payload}")
        response = await protocol.send_request(request)
        
        if response.is_success():
            print(f"  Result: {response.data}")
        else:
            print(f"  Error: {response.error_message}")
    
    await protocol.stop()

async def demo_concurrent_requests():
    """Demo: Concurrent request handling"""
    print("\nDEMO 3: CONCURRENT REQUESTS")
    print("=" * 50)
    
    protocol = RequestResponseProtocol()
    await protocol.start()
    
    # Register multiple handlers
    calculator = CalculatorHandler()
    weather = WeatherHandler()
    database = DatabaseHandler()
    
    protocol.register_handler(calculator)
    protocol.register_handler(weather)
    protocol.register_handler(database)
    
    # Create concurrent requests
    concurrent_requests = [
        Request("", RequestType.QUERY, "add", "calculator", {"a": 5, "b": 3}),
        Request("", RequestType.QUERY, "current", "weather", {"city": "new_york"}),
        Request("", RequestType.COMMAND, "set", "database", {"key": "test", "value": "data"}),
        Request("", RequestType.QUERY, "multiply", "calculator", {"a": 4, "b": 7}),
        Request("", RequestType.QUERY, "forecast", "weather", {"city": "london", "days": 3}),
        Request("", RequestType.QUERY, "get", "database", {"key": "test"})
    ]
    
    print(f"Sending {len(concurrent_requests)} concurrent requests...")
    
    start_time = time.time()
    
    # Send all requests concurrently
    responses = await protocol.send_batch_request(concurrent_requests)
    
    end_time = time.time()
    
    print(f"All responses received in {end_time - start_time:.2f}s")
    print("\nResults:")
    
    for i, (request, response) in enumerate(zip(concurrent_requests, responses)):
        print(f"  {i+1}. {request.method} -> {response.status.value}")
        if response.is_success():
            print(f"     Data: {response.data}")
        else:
            print(f"     Error: {response.error_message}")
    
    await protocol.stop()

async def demo_timeout_handling():
    """Demo: Request timeout and retry handling"""
    print("\nDEMO 4: TIMEOUT HANDLING")
    print("=" * 50)
    
    protocol = RequestResponseProtocol()
    await protocol.start()
    
    # Create slow handler that sometimes times out
    class SlowHandler(RequestHandler):
        def __init__(self):
            self.call_count = 0
        
        async def handle_request(self, request: Request) -> Response:
            self.call_count += 1
            delay = request.payload.get('delay', 1.0)
            
            print(f"    Processing request with {delay}s delay...")
            await asyncio.sleep(delay)
            
            return Response(
                request_id=request.id,
                status=ResponseStatus.SUCCESS,
                data={'message': f'Completed after {delay}s delay', 'call_count': self.call_count},
                responder_id="slow_service"
            )
        
        def can_handle(self, request: Request) -> bool:
            return request.target_service == "slow_service"
        
        def get_service_name(self) -> str:
            return "slow_service"
    
    slow_handler = SlowHandler()
    protocol.register_handler(slow_handler)
    
    # Test requests with different timeouts
    test_cases = [
        {"delay": 0.5, "timeout": 2.0, "description": "Fast request (should succeed)"},
        {"delay": 3.0, "timeout": 1.0, "description": "Slow request (should timeout)"},
        {"delay": 1.5, "timeout": 2.0, "description": "Medium request (should succeed)"}
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['description']}")
        
        request = Request(
            id="",
            request_type=RequestType.QUERY,
            method="process",
            target_service="slow_service",
            payload={"delay": case["delay"]},
            timeout=case["timeout"]
        )
        
        start_time = time.time()
        response = await protocol.send_request(request)
        elapsed = time.time() - start_time
        
        print(f"  Elapsed time: {elapsed:.2f}s")
        print(f"  Status: {response.status.value}")
        
        if response.is_success():
            print(f"  Response: {response.data}")
        else:
            print(f"  Error: {response.error_message}")
    
    await protocol.stop()

async def demo_streaming_responses():
    """Demo: Streaming response pattern"""
    print("\nDEMO 5: STREAMING RESPONSES")
    print("=" * 50)
    
    # Note: This is a simplified streaming demo
    # In a real implementation, you'd use AsyncIterator or similar
    
    protocol = RequestResponseProtocol()
    await protocol.start()
    
    class DataStreamHandler(RequestHandler):
        async def handle_request(self, request: Request) -> Response:
            count = request.payload.get('count', 5)
            interval = request.payload.get('interval', 0.5)
            
            # Simulate streaming by collecting data over time
            stream_data = []
            
            for i in range(count):
                await asyncio.sleep(interval)
                data_point = {
                    'sequence': i + 1,
                    'timestamp': time.time(),
                    'value': f"data_item_{i+1}"
                }
                stream_data.append(data_point)
                print(f"    Generated: {data_point['value']}")
            
            return Response(
                request_id=request.id,
                status=ResponseStatus.SUCCESS,
                data={'stream_data': stream_data, 'total_items': count},
                responder_id="stream_service"
            )
        
        def can_handle(self, request: Request) -> bool:
            return request.target_service == "stream_service"
        
        def get_service_name(self) -> str:
            return "stream_service"
    
    stream_handler = DataStreamHandler()
    protocol.register_handler(stream_handler)
    
    # Request streaming data
    print("Requesting streaming data (5 items with 0.3s intervals):")
    
    request = Request(
        id="",
        request_type=RequestType.STREAM,
        method="generate",
        target_service="stream_service",
        payload={"count": 5, "interval": 0.3}
    )
    
    start_time = time.time()
    response = await protocol.send_request(request)
    elapsed = time.time() - start_time
    
    print(f"\nStreaming completed in {elapsed:.2f}s")
    
    if response.is_success():
        stream_data = response.data.get('stream_data', [])
        print(f"Received {len(stream_data)} data items:")
        for item in stream_data:
            print(f"  - {item['value']} (seq: {item['sequence']})")
    
    await protocol.stop()

async def main():
    """
    Demonstrate Request-Response Protocols for synchronous agent communication
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement reliable request-response communication
    2. How to handle timeouts, retries, and error responses
    3. How to route requests to appropriate handlers
    4. How to manage concurrent requests efficiently
    5. How to build synchronous workflows with async agents
    
    REAL WORLD APPLICATIONS:
    =======================
    - API gateways and microservice communication
    - Database query and transaction systems
    - Interactive agent conversations and negotiations
    - Real-time data retrieval and processing
    - Synchronous workflow orchestration
    - User interface backend communication
    """
    
    print("REQUEST-RESPONSE PROTOCOLS DEMONSTRATION")
    print("Showing how agents communicate synchronously and reliably!")
    
    await demo_basic_request_response()
    await demo_database_operations()
    await demo_concurrent_requests()
    await demo_timeout_handling()
    await demo_streaming_responses()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Request-response enables synchronous agent communication")
    print("✓ Timeouts prevent hanging requests and ensure responsiveness")
    print("✓ Error handling provides clear feedback on operation failures")
    print("✓ Concurrent processing improves system throughput")
    print("✓ Request routing enables scalable service architectures")
    print("✓ Correlation IDs track request-response pairs")
    print("\nTHE POWER OF REQUEST-RESPONSE:")
    print("- Enables reliable distributed transactions")
    print("- Provides immediate feedback for user interactions")
    print("- Supports complex query and command operations")
    print("- Maintains data consistency across operations")

if __name__ == "__main__":
    asyncio.run(main())
