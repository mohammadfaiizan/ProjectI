"""
CHAIN OF RESPONSIBILITY PATTERN - Behavioral Design Pattern
===========================================================

Problem Statement:
Implement the Chain of Responsibility pattern to pass requests along a chain
of handlers, where each handler decides either to process the request or pass
it to the next handler in the chain:
- Request handling chain with multiple processors
- Decoupling sender from receiver
- Dynamic handler configuration and ordering
- Request filtering and processing pipeline
- Error handling and fallback mechanisms

Learning Objectives:
- Understand Chain of Responsibility vs Command pattern differences
- Implement handler chains with flexible configuration
- Design request processing pipelines
- Handle dynamic chain modification
- Create robust error handling and logging systems
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable, Type
import time
import json
from datetime import datetime
from enum import Enum
import logging
import re


# ============================================================================
# CHAIN OF RESPONSIBILITY BASE CLASSES
# ============================================================================

class Request:
    """Base request class for chain processing."""
    
    def __init__(self, request_type: str, data: Dict[str, Any], priority: int = 1):
        self.request_type = request_type
        self.data = data
        self.priority = priority
        self.timestamp = datetime.now()
        self.processing_history: List[Dict[str, Any]] = []
        self.metadata = {}
        
    def add_processing_step(self, handler_name: str, action: str, result: Any = None) -> None:
        """Add processing step to history."""
        step = {
            'handler': handler_name,
            'action': action,
            'result': str(result) if result is not None else None,
            'timestamp': datetime.now().isoformat()
        }
        self.processing_history.append(step)
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of request processing."""
        return {
            'request_type': self.request_type,
            'priority': self.priority,
            'created_at': self.timestamp.isoformat(),
            'processing_steps': len(self.processing_history),
            'handlers_involved': [step['handler'] for step in self.processing_history],
            'final_result': self.processing_history[-1]['result'] if self.processing_history else None
        }


class Handler(ABC):
    """Abstract handler in the chain of responsibility."""
    
    def __init__(self, name: str):
        self.name = name
        self._next_handler: Optional['Handler'] = None
        self.processed_count = 0
        self.processing_time = 0.0
        
    def set_next(self, handler: 'Handler') -> 'Handler':
        """Set the next handler in the chain."""
        self._next_handler = handler
        return handler
    
    def handle(self, request: Request) -> Optional[Any]:
        """Handle the request or pass it to the next handler."""
        start_time = time.time()
        
        try:
            # Check if this handler can process the request
            if self.can_handle(request):
                request.add_processing_step(self.name, "processing")
                result = self.process_request(request)
                
                self.processed_count += 1
                self.processing_time += time.time() - start_time
                
                request.add_processing_step(self.name, "completed", result)
                return result
            else:
                # Pass to next handler if available
                request.add_processing_step(self.name, "skipped")
                if self._next_handler:
                    return self._next_handler.handle(request)
                else:
                    request.add_processing_step(self.name, "unhandled")
                    return None
                    
        except Exception as e:
            self.processing_time += time.time() - start_time
            request.add_processing_step(self.name, "error", str(e))
            
            # Try next handler on error
            if self._next_handler:
                return self._next_handler.handle(request)
            else:
                raise e
    
    @abstractmethod
    def can_handle(self, request: Request) -> bool:
        """Check if this handler can process the request."""
        pass
    
    @abstractmethod
    def process_request(self, request: Request) -> Any:
        """Process the request - must be implemented by subclasses."""
        pass
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            'name': self.name,
            'processed_count': self.processed_count,
            'total_processing_time': self.processing_time,
            'average_processing_time': self.processing_time / self.processed_count if self.processed_count > 0 else 0,
            'has_next_handler': self._next_handler is not None,
            'next_handler': self._next_handler.name if self._next_handler else None
        }


# ============================================================================
# AUTHENTICATION AND AUTHORIZATION CHAIN
# ============================================================================

class AuthenticationRequest(Request):
    """Authentication request with user credentials."""
    
    def __init__(self, username: str, password: str, auth_method: str = "password"):
        super().__init__("authentication", {
            'username': username,
            'password': password,
            'auth_method': auth_method,
            'ip_address': '192.168.1.100',  # Simulated
            'user_agent': 'Mozilla/5.0...'  # Simulated
        })


class RateLimitHandler(Handler):
    """Handler to check rate limiting."""
    
    def __init__(self):
        super().__init__("RateLimitHandler")
        self.request_counts = {}  # username -> count
        self.time_windows = {}    # username -> timestamp
        self.max_requests = 5
        self.time_window = 60  # seconds
    
    def can_handle(self, request: Request) -> bool:
        """Always check rate limits for authentication requests."""
        return request.request_type == "authentication"
    
    def process_request(self, request: Request) -> Any:
        """Check rate limiting for user."""
        username = request.data.get('username', '')
        current_time = time.time()
        
        # Reset count if time window has passed
        if username in self.time_windows:
            if current_time - self.time_windows[username] > self.time_window:
                self.request_counts[username] = 0
                self.time_windows[username] = current_time
        else:
            self.time_windows[username] = current_time
            self.request_counts[username] = 0
        
        # Check rate limit
        self.request_counts[username] = self.request_counts.get(username, 0) + 1
        
        if self.request_counts[username] > self.max_requests:
            raise Exception(f"Rate limit exceeded for user {username}. "
                          f"Max {self.max_requests} requests per {self.time_window} seconds.")
        
        print(f"Rate limit check passed for {username} "
              f"({self.request_counts[username]}/{self.max_requests})")
        
        return f"Rate limit OK ({self.request_counts[username]}/{self.max_requests})"


class InputValidationHandler(Handler):
    """Handler to validate input data."""
    
    def __init__(self):
        super().__init__("InputValidationHandler")
        self.username_pattern = re.compile(r'^[a-zA-Z0-9_]{3,20}$')
        self.password_min_length = 6
    
    def can_handle(self, request: Request) -> bool:
        """Validate all authentication requests."""
        return request.request_type == "authentication"
    
    def process_request(self, request: Request) -> Any:
        """Validate input data."""
        username = request.data.get('username', '')
        password = request.data.get('password', '')
        
        errors = []
        
        # Validate username
        if not username:
            errors.append("Username is required")
        elif not self.username_pattern.match(username):
            errors.append("Username must be 3-20 characters, alphanumeric and underscore only")
        
        # Validate password
        if not password:
            errors.append("Password is required")
        elif len(password) < self.password_min_length:
            errors.append(f"Password must be at least {self.password_min_length} characters")
        
        if errors:
            raise Exception(f"Validation errors: {'; '.join(errors)}")
        
        print(f"Input validation passed for user {username}")
        return "Input validation successful"


class AuthenticationHandler(Handler):
    """Handler to authenticate user credentials."""
    
    def __init__(self):
        super().__init__("AuthenticationHandler")
        # Simulated user database
        self.users = {
            'alice': {'password': 'password123', 'active': True, 'role': 'admin'},
            'bob': {'password': 'secret456', 'active': True, 'role': 'user'},
            'charlie': {'password': 'mypass789', 'active': False, 'role': 'user'}
        }
    
    def can_handle(self, request: Request) -> bool:
        """Handle authentication requests."""
        return request.request_type == "authentication"
    
    def process_request(self, request: Request) -> Any:
        """Authenticate user credentials."""
        username = request.data.get('username', '')
        password = request.data.get('password', '')
        
        # Check if user exists
        if username not in self.users:
            raise Exception(f"User {username} not found")
        
        user = self.users[username]
        
        # Check if user is active
        if not user['active']:
            raise Exception(f"User {username} is deactivated")
        
        # Check password
        if user['password'] != password:
            raise Exception("Invalid password")
        
        # Authentication successful
        auth_result = {
            'username': username,
            'role': user['role'],
            'authenticated': True,
            'auth_time': datetime.now().isoformat()
        }
        
        print(f"Authentication successful for user {username} with role {user['role']}")
        return auth_result


class AuthorizationHandler(Handler):
    """Handler to check user authorization."""
    
    def __init__(self, required_role: str = None):
        super().__init__("AuthorizationHandler")
        self.required_role = required_role
        self.role_hierarchy = {
            'admin': 3,
            'moderator': 2,
            'user': 1,
            'guest': 0
        }
    
    def can_handle(self, request: Request) -> bool:
        """Handle requests that need authorization."""
        return request.request_type == "authentication" and self.required_role is not None
    
    def process_request(self, request: Request) -> Any:
        """Check user authorization."""
        # Get authentication result from request metadata
        auth_result = request.metadata.get('auth_result')
        
        if not auth_result or not auth_result.get('authenticated'):
            raise Exception("User must be authenticated first")
        
        user_role = auth_result.get('role', 'guest')
        
        # Check role hierarchy
        user_level = self.role_hierarchy.get(user_role, 0)
        required_level = self.role_hierarchy.get(self.required_role, 0)
        
        if user_level < required_level:
            raise Exception(f"Insufficient privileges. Required: {self.required_role}, "
                          f"User has: {user_role}")
        
        print(f"Authorization successful for user {auth_result['username']} "
              f"(role: {user_role}, required: {self.required_role})")
        
        return f"Authorization granted for role {user_role}"


class AuditLogHandler(Handler):
    """Handler to log authentication attempts."""
    
    def __init__(self):
        super().__init__("AuditLogHandler")
        self.audit_logs: List[Dict[str, Any]] = []
    
    def can_handle(self, request: Request) -> bool:
        """Log all authentication requests."""
        return request.request_type == "authentication"
    
    def process_request(self, request: Request) -> Any:
        """Log authentication attempt."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'username': request.data.get('username', 'unknown'),
            'ip_address': request.data.get('ip_address', 'unknown'),
            'user_agent': request.data.get('user_agent', 'unknown'),
            'auth_method': request.data.get('auth_method', 'password'),
            'success': True,  # If we reach here, authentication was successful
            'processing_steps': len(request.processing_history)
        }
        
        self.audit_logs.append(log_entry)
        
        print(f"Audit log created for user {log_entry['username']}")
        return f"Audit log entry #{len(self.audit_logs)} created"
    
    def get_audit_logs(self, username: str = None) -> List[Dict[str, Any]]:
        """Get audit logs, optionally filtered by username."""
        if username:
            return [log for log in self.audit_logs if log['username'] == username]
        return self.audit_logs.copy()


# ============================================================================
# SUPPORT TICKET PROCESSING CHAIN
# ============================================================================

class SupportTicket(Request):
    """Support ticket request."""
    
    def __init__(self, ticket_id: str, customer_id: str, issue_type: str, 
                 description: str, priority: int = 1):
        super().__init__("support_ticket", {
            'ticket_id': ticket_id,
            'customer_id': customer_id,
            'issue_type': issue_type,
            'description': description,
            'customer_tier': 'standard'  # Will be determined by handler
        }, priority)


class CustomerTierHandler(Handler):
    """Handler to determine customer tier and adjust priority."""
    
    def __init__(self):
        super().__init__("CustomerTierHandler")
        # Simulated customer database
        self.customers = {
            'CUST001': {'tier': 'premium', 'name': 'Acme Corp'},
            'CUST002': {'tier': 'standard', 'name': 'Tech Solutions'},
            'CUST003': {'tier': 'basic', 'name': 'Small Business'},
            'CUST004': {'tier': 'premium', 'name': 'Enterprise Inc'}
        }
        self.tier_priority_boost = {
            'premium': 2,
            'standard': 1,
            'basic': 0
        }
    
    def can_handle(self, request: Request) -> bool:
        """Handle all support tickets."""
        return request.request_type == "support_ticket"
    
    def process_request(self, request: Request) -> Any:
        """Determine customer tier and adjust priority."""
        customer_id = request.data.get('customer_id', '')
        
        if customer_id in self.customers:
            customer = self.customers[customer_id]
            tier = customer['tier']
            
            # Update request data
            request.data['customer_tier'] = tier
            request.data['customer_name'] = customer['name']
            
            # Boost priority based on tier
            priority_boost = self.tier_priority_boost.get(tier, 0)
            request.priority += priority_boost
            
            result = f"Customer tier: {tier}, Priority boosted to: {request.priority}"
            print(f"Customer {customer_id} ({customer['name']}) - {result}")
            
            return result
        else:
            # Unknown customer - treat as basic
            request.data['customer_tier'] = 'basic'
            request.data['customer_name'] = 'Unknown Customer'
            
            result = "Unknown customer, assigned basic tier"
            print(result)
            return result


class AutoResponseHandler(Handler):
    """Handler to provide automatic responses for common issues."""
    
    def __init__(self):
        super().__init__("AutoResponseHandler")
        self.auto_responses = {
            'password_reset': {
                'response': 'Please visit our password reset page at example.com/reset',
                'resolution_time': 0,
                'requires_human': False
            },
            'billing_inquiry': {
                'response': 'Your billing information can be found in your account dashboard',
                'resolution_time': 0,
                'requires_human': False
            },
            'account_locked': {
                'response': 'Your account has been unlocked. Please try logging in again.',
                'resolution_time': 5,  # 5 minutes to process
                'requires_human': False
            }
        }
    
    def can_handle(self, request: Request) -> bool:
        """Handle tickets that have automatic responses."""
        issue_type = request.data.get('issue_type', '')
        return (request.request_type == "support_ticket" and 
                issue_type in self.auto_responses)
    
    def process_request(self, request: Request) -> Any:
        """Provide automatic response."""
        issue_type = request.data.get('issue_type', '')
        auto_response = self.auto_responses[issue_type]
        
        # Simulate processing time
        if auto_response['resolution_time'] > 0:
            time.sleep(0.1)  # Reduced for demo
        
        result = {
            'response_type': 'automatic',
            'response': auto_response['response'],
            'resolution_time_minutes': auto_response['resolution_time'],
            'ticket_status': 'resolved' if not auto_response['requires_human'] else 'pending'
        }
        
        print(f"Auto-response provided for {issue_type}: {auto_response['response']}")
        return result


class Level1SupportHandler(Handler):
    """Handler for Level 1 support (basic issues)."""
    
    def __init__(self):
        super().__init__("Level1SupportHandler")
        self.handled_issue_types = [
            'general_inquiry', 'feature_request', 'documentation',
            'basic_troubleshooting'
        ]
        self.max_priority = 3
    
    def can_handle(self, request: Request) -> bool:
        """Handle basic support tickets."""
        issue_type = request.data.get('issue_type', '')
        return (request.request_type == "support_ticket" and
                issue_type in self.handled_issue_types and
                request.priority <= self.max_priority)
    
    def process_request(self, request: Request) -> Any:
        """Process Level 1 support ticket."""
        issue_type = request.data.get('issue_type', '')
        description = request.data.get('description', '')
        
        # Simulate processing time based on issue complexity
        processing_time = 15 + len(description) // 10  # 15+ minutes
        time.sleep(0.05)  # Reduced for demo
        
        result = {
            'response_type': 'level1_support',
            'assigned_agent': 'Agent_L1_001',
            'estimated_resolution_minutes': processing_time,
            'ticket_status': 'in_progress',
            'next_update_hours': 4
        }
        
        print(f"Level 1 support assigned for {issue_type} "
              f"(Est. {processing_time} min resolution)")
        
        return result


class Level2SupportHandler(Handler):
    """Handler for Level 2 support (technical issues)."""
    
    def __init__(self):
        super().__init__("Level2SupportHandler")
        self.handled_issue_types = [
            'technical_issue', 'integration_problem', 'performance_issue',
            'advanced_troubleshooting'
        ]
        self.max_priority = 5
    
    def can_handle(self, request: Request) -> bool:
        """Handle technical support tickets."""
        issue_type = request.data.get('issue_type', '')
        return (request.request_type == "support_ticket" and
                issue_type in self.handled_issue_types and
                request.priority <= self.max_priority)
    
    def process_request(self, request: Request) -> Any:
        """Process Level 2 support ticket."""
        issue_type = request.data.get('issue_type', '')
        customer_tier = request.data.get('customer_tier', 'basic')
        
        # Priority customers get faster response
        base_time = 60 if customer_tier == 'premium' else 120  # minutes
        
        result = {
            'response_type': 'level2_support',
            'assigned_agent': 'Agent_L2_003',
            'estimated_resolution_minutes': base_time,
            'ticket_status': 'escalated_to_technical',
            'next_update_hours': 2,
            'requires_specialist': issue_type == 'integration_problem'
        }
        
        print(f"Level 2 support assigned for {issue_type} "
              f"(Est. {base_time} min resolution)")
        
        return result


class EscalationHandler(Handler):
    """Handler for high-priority escalations."""
    
    def __init__(self):
        super().__init__("EscalationHandler")
        self.escalation_threshold = 5
    
    def can_handle(self, request: Request) -> bool:
        """Handle high-priority tickets that need escalation."""
        return (request.request_type == "support_ticket" and
                request.priority >= self.escalation_threshold)
    
    def process_request(self, request: Request) -> Any:
        """Escalate high-priority ticket."""
        customer_name = request.data.get('customer_name', 'Unknown')
        issue_type = request.data.get('issue_type', 'unknown')
        
        result = {
            'response_type': 'escalation',
            'escalated_to': 'Senior Support Manager',
            'escalation_reason': f'High priority ({request.priority}) ticket',
            'immediate_response_required': True,
            'ticket_status': 'escalated',
            'next_update_minutes': 30,
            'manager_notified': True
        }
        
        print(f"ESCALATION: {issue_type} for {customer_name} "
              f"(Priority: {request.priority})")
        
        return result


# ============================================================================
# HTTP REQUEST PROCESSING CHAIN
# ============================================================================

class HTTPRequest(Request):
    """HTTP request for web processing chain."""
    
    def __init__(self, method: str, path: str, headers: Dict[str, str], 
                 body: str = "", query_params: Dict[str, str] = None):
        super().__init__("http_request", {
            'method': method,
            'path': path,
            'headers': headers,
            'body': body,
            'query_params': query_params or {},
            'client_ip': '192.168.1.100'  # Simulated
        })


class CORSHandler(Handler):
    """Handler for CORS (Cross-Origin Resource Sharing) processing."""
    
    def __init__(self, allowed_origins: List[str] = None):
        super().__init__("CORSHandler")
        self.allowed_origins = allowed_origins or ['*']
        self.allowed_methods = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
        self.allowed_headers = ['Content-Type', 'Authorization', 'X-Requested-With']
    
    def can_handle(self, request: Request) -> bool:
        """Handle all HTTP requests for CORS."""
        return request.request_type == "http_request"
    
    def process_request(self, request: Request) -> Any:
        """Process CORS headers."""
        headers = request.data.get('headers', {})
        origin = headers.get('Origin', '')
        
        cors_headers = {}
        
        # Check if origin is allowed
        if '*' in self.allowed_origins or origin in self.allowed_origins:
            cors_headers['Access-Control-Allow-Origin'] = origin or '*'
            cors_headers['Access-Control-Allow-Methods'] = ', '.join(self.allowed_methods)
            cors_headers['Access-Control-Allow-Headers'] = ', '.join(self.allowed_headers)
            cors_headers['Access-Control-Max-Age'] = '86400'  # 24 hours
            
            # Store CORS headers in request metadata
            request.metadata['cors_headers'] = cors_headers
            
            print(f"CORS headers added for origin: {origin or 'any'}")
            return f"CORS processed for origin: {origin or 'any'}"
        else:
            raise Exception(f"Origin {origin} not allowed by CORS policy")


class AuthenticationMiddleware(Handler):
    """Handler for HTTP authentication."""
    
    def __init__(self, protected_paths: List[str] = None):
        super().__init__("AuthenticationMiddleware")
        self.protected_paths = protected_paths or ['/api/', '/admin/']
        self.valid_tokens = {
            'token123': {'user': 'alice', 'role': 'admin'},
            'token456': {'user': 'bob', 'role': 'user'}
        }
    
    def can_handle(self, request: Request) -> bool:
        """Handle requests to protected paths."""
        if request.request_type != "http_request":
            return False
        
        path = request.data.get('path', '')
        return any(path.startswith(protected) for protected in self.protected_paths)
    
    def process_request(self, request: Request) -> Any:
        """Authenticate HTTP request."""
        headers = request.data.get('headers', {})
        auth_header = headers.get('Authorization', '')
        
        if not auth_header.startswith('Bearer '):
            raise Exception("Missing or invalid Authorization header")
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        if token not in self.valid_tokens:
            raise Exception("Invalid authentication token")
        
        user_info = self.valid_tokens[token]
        request.metadata['authenticated_user'] = user_info
        
        print(f"HTTP authentication successful for user: {user_info['user']}")
        return f"Authenticated as {user_info['user']} ({user_info['role']})"


class RateLimitMiddleware(Handler):
    """Handler for HTTP rate limiting."""
    
    def __init__(self, requests_per_minute: int = 60):
        super().__init__("RateLimitMiddleware")
        self.requests_per_minute = requests_per_minute
        self.client_requests = {}  # client_ip -> [timestamps]
    
    def can_handle(self, request: Request) -> bool:
        """Handle all HTTP requests for rate limiting."""
        return request.request_type == "http_request"
    
    def process_request(self, request: Request) -> Any:
        """Check rate limits for client."""
        client_ip = request.data.get('client_ip', 'unknown')
        current_time = time.time()
        
        # Initialize client tracking
        if client_ip not in self.client_requests:
            self.client_requests[client_ip] = []
        
        # Remove old requests (older than 1 minute)
        self.client_requests[client_ip] = [
            timestamp for timestamp in self.client_requests[client_ip]
            if current_time - timestamp < 60
        ]
        
        # Check rate limit
        if len(self.client_requests[client_ip]) >= self.requests_per_minute:
            raise Exception(f"Rate limit exceeded for IP {client_ip}. "
                          f"Max {self.requests_per_minute} requests per minute.")
        
        # Add current request
        self.client_requests[client_ip].append(current_time)
        
        remaining = self.requests_per_minute - len(self.client_requests[client_ip])
        
        print(f"Rate limit check passed for {client_ip} ({remaining} requests remaining)")
        return f"Rate limit OK ({remaining} remaining)"


class RequestLoggingHandler(Handler):
    """Handler for HTTP request logging."""
    
    def __init__(self):
        super().__init__("RequestLoggingHandler")
        self.request_logs: List[Dict[str, Any]] = []
    
    def can_handle(self, request: Request) -> bool:
        """Log all HTTP requests."""
        return request.request_type == "http_request"
    
    def process_request(self, request: Request) -> Any:
        """Log HTTP request details."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'method': request.data.get('method', 'UNKNOWN'),
            'path': request.data.get('path', '/'),
            'client_ip': request.data.get('client_ip', 'unknown'),
            'user_agent': request.data.get('headers', {}).get('User-Agent', 'unknown'),
            'authenticated_user': request.metadata.get('authenticated_user', {}).get('user'),
            'processing_time_ms': sum(
                step.get('processing_time', 0) for step in request.processing_history
            ) * 1000
        }
        
        self.request_logs.append(log_entry)
        
        print(f"HTTP request logged: {log_entry['method']} {log_entry['path']} "
              f"from {log_entry['client_ip']}")
        
        return f"Request logged (#{len(self.request_logs)})"
    
    def get_request_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent request logs."""
        return self.request_logs[-limit:] if self.request_logs else []


# ============================================================================
# CHAIN BUILDER AND MANAGER
# ============================================================================

class ChainBuilder:
    """Builder for creating handler chains."""
    
    def __init__(self):
        self.handlers: List[Handler] = []
    
    def add_handler(self, handler: Handler) -> 'ChainBuilder':
        """Add handler to the chain."""
        self.handlers.append(handler)
        return self
    
    def build(self) -> Optional[Handler]:
        """Build the handler chain."""
        if not self.handlers:
            return None
        
        # Link handlers together
        for i in range(len(self.handlers) - 1):
            self.handlers[i].set_next(self.handlers[i + 1])
        
        return self.handlers[0]  # Return first handler
    
    def clear(self) -> 'ChainBuilder':
        """Clear all handlers."""
        self.handlers = []
        return self


class ChainManager:
    """Manager for multiple handler chains."""
    
    def __init__(self):
        self.chains: Dict[str, Handler] = {}
        self.chain_stats: Dict[str, Dict[str, Any]] = {}
    
    def register_chain(self, name: str, chain: Handler) -> None:
        """Register a handler chain."""
        self.chains[name] = chain
        self.chain_stats[name] = {
            'requests_processed': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0
        }
        print(f"Registered chain: {name}")
    
    def process_request(self, chain_name: str, request: Request) -> Optional[Any]:
        """Process request using specified chain."""
        if chain_name not in self.chains:
            raise ValueError(f"Unknown chain: {chain_name}")
        
        chain = self.chains[chain_name]
        stats = self.chain_stats[chain_name]
        
        start_time = time.time()
        
        try:
            result = chain.handle(request)
            
            # Update statistics
            stats['requests_processed'] += 1
            stats['successful_requests'] += 1
            stats['total_processing_time'] += time.time() - start_time
            
            return result
            
        except Exception as e:
            stats['requests_processed'] += 1
            stats['failed_requests'] += 1
            stats['total_processing_time'] += time.time() - start_time
            
            print(f"Chain {chain_name} failed to process request: {e}")
            raise e
    
    def get_chain_statistics(self, chain_name: str = None) -> Dict[str, Any]:
        """Get statistics for chain(s)."""
        if chain_name:
            if chain_name not in self.chain_stats:
                return {}
            
            stats = self.chain_stats[chain_name].copy()
            if stats['requests_processed'] > 0:
                stats['success_rate'] = (stats['successful_requests'] / 
                                       stats['requests_processed']) * 100
                stats['average_processing_time'] = (stats['total_processing_time'] / 
                                                  stats['requests_processed'])
            
            return stats
        else:
            return {name: self.get_chain_statistics(name) 
                   for name in self.chain_stats.keys()}
    
    def get_handler_statistics(self, chain_name: str) -> List[Dict[str, Any]]:
        """Get statistics for all handlers in a chain."""
        if chain_name not in self.chains:
            return []
        
        handler_stats = []
        current_handler = self.chains[chain_name]
        
        while current_handler:
            handler_stats.append(current_handler.get_handler_stats())
            current_handler = current_handler._next_handler
        
        return handler_stats


def demonstrate_chain_of_responsibility_pattern():
    """
    Demonstrate Chain of Responsibility pattern implementations.
    """
    print("=== CHAIN OF RESPONSIBILITY PATTERN DEMONSTRATION ===\n")
    
    # 1. Authentication and Authorization Chain
    print("1. AUTHENTICATION AND AUTHORIZATION CHAIN:")
    
    # Build authentication chain
    auth_chain = (ChainBuilder()
                  .add_handler(RateLimitHandler())
                  .add_handler(InputValidationHandler())
                  .add_handler(AuthenticationHandler())
                  .add_handler(AuthorizationHandler('user'))
                  .add_handler(AuditLogHandler())
                  .build())
    
    # Test authentication requests
    auth_requests = [
        AuthenticationRequest('alice', 'password123'),
        AuthenticationRequest('bob', 'secret456'),
        AuthenticationRequest('charlie', 'mypass789'),  # Inactive user
        AuthenticationRequest('invalid_user', 'wrongpass'),  # Non-existent user
        AuthenticationRequest('alice', 'wrongpass'),  # Wrong password
    ]
    
    print("   Testing authentication chain:")
    
    for i, request in enumerate(auth_requests, 1):
        print(f"\n   Request {i}: {request.data['username']}")
        print("   " + "-" * 40)
        
        try:
            result = auth_chain.handle(request)
            print(f"   ✓ Authentication successful: {result}")
            
            # Show processing summary
            summary = request.get_processing_summary()
            print(f"   Handlers involved: {summary['handlers_involved']}")
            
        except Exception as e:
            print(f"   ✗ Authentication failed: {e}")
            
            # Show processing history even for failures
            if request.processing_history:
                handlers = [step['handler'] for step in request.processing_history]
                print(f"   Handlers processed: {handlers}")
    
    print()
    
    # 2. Support Ticket Processing Chain
    print("2. SUPPORT TICKET PROCESSING CHAIN:")
    
    # Build support ticket chain
    support_chain = (ChainBuilder()
                     .add_handler(CustomerTierHandler())
                     .add_handler(AutoResponseHandler())
                     .add_handler(Level1SupportHandler())
                     .add_handler(Level2SupportHandler())
                     .add_handler(EscalationHandler())
                     .build())
    
    # Test support tickets
    tickets = [
        SupportTicket('TKT001', 'CUST001', 'password_reset', 
                     'I forgot my password and cannot log in', 1),
        SupportTicket('TKT002', 'CUST002', 'general_inquiry', 
                     'How do I use the new feature?', 2),
        SupportTicket('TKT003', 'CUST003', 'technical_issue', 
                     'API integration is not working properly', 3),
        SupportTicket('TKT004', 'CUST004', 'critical_bug', 
                     'System is completely down for all users', 6),  # High priority
        SupportTicket('TKT005', 'CUST999', 'billing_inquiry', 
                     'Question about my invoice', 1),  # Unknown customer
    ]
    
    print("   Testing support ticket chain:")
    
    for ticket in tickets:
        print(f"\n   Ticket {ticket.data['ticket_id']}: {ticket.data['issue_type']}")
        print(f"   Customer: {ticket.data['customer_id']}, Priority: {ticket.priority}")
        print("   " + "-" * 50)
        
        try:
            result = support_chain.handle(ticket)
            print(f"   ✓ Ticket processed: {result}")
            
            # Show processing summary
            summary = ticket.get_processing_summary()
            print(f"   Final priority: {ticket.priority}")
            print(f"   Handlers: {' → '.join(summary['handlers_involved'])}")
            
        except Exception as e:
            print(f"   ✗ Ticket processing failed: {e}")
    
    print()
    
    # 3. HTTP Request Processing Chain
    print("3. HTTP REQUEST PROCESSING CHAIN:")
    
    # Build HTTP processing chain
    http_chain = (ChainBuilder()
                  .add_handler(CORSHandler(['https://example.com', 'https://app.example.com']))
                  .add_handler(RateLimitMiddleware(10))  # 10 requests per minute
                  .add_handler(AuthenticationMiddleware(['/api/', '/admin/']))
                  .add_handler(RequestLoggingHandler())
                  .build())
    
    # Test HTTP requests
    http_requests = [
        HTTPRequest('GET', '/public/info', {'Origin': 'https://example.com'}),
        HTTPRequest('GET', '/api/users', {
            'Authorization': 'Bearer token123',
            'Origin': 'https://app.example.com'
        }),
        HTTPRequest('POST', '/api/data', {
            'Authorization': 'Bearer invalid_token',
            'Content-Type': 'application/json'
        }),
        HTTPRequest('GET', '/admin/settings', {
            'Authorization': 'Bearer token456'
        }),
        HTTPRequest('GET', '/api/public', {})  # Missing auth for protected path
    ]
    
    print("   Testing HTTP request chain:")
    
    for i, request in enumerate(http_requests, 1):
        print(f"\n   Request {i}: {request.data['method']} {request.data['path']}")
        print("   " + "-" * 40)
        
        try:
            result = http_chain.handle(request)
            print(f"   ✓ Request processed: {result}")
            
            # Show CORS headers if added
            if 'cors_headers' in request.metadata:
                print(f"   CORS headers added: {len(request.metadata['cors_headers'])}")
            
            # Show authenticated user if present
            if 'authenticated_user' in request.metadata:
                user = request.metadata['authenticated_user']
                print(f"   Authenticated as: {user['user']} ({user['role']})")
            
        except Exception as e:
            print(f"   ✗ Request failed: {e}")
    
    print()
    
    # 4. Chain Manager and Statistics
    print("4. CHAIN MANAGER AND STATISTICS:")
    
    # Create chain manager
    chain_manager = ChainManager()
    
    # Register chains
    chain_manager.register_chain('authentication', auth_chain)
    chain_manager.register_chain('support', support_chain)
    chain_manager.register_chain('http', http_chain)
    
    # Process some requests through manager
    print("   Processing requests through chain manager:")
    
    test_requests = [
        ('authentication', AuthenticationRequest('alice', 'password123')),
        ('support', SupportTicket('TKT999', 'CUST001', 'general_inquiry', 'Test ticket', 1)),
        ('http', HTTPRequest('GET', '/public/test', {}))
    ]
    
    for chain_name, request in test_requests:
        try:
            result = chain_manager.process_request(chain_name, request)
            print(f"   ✓ {chain_name} chain: {result}")
        except Exception as e:
            print(f"   ✗ {chain_name} chain failed: {e}")
    
    # Show chain statistics
    print(f"\n   Chain Statistics:")
    all_stats = chain_manager.get_chain_statistics()
    
    for chain_name, stats in all_stats.items():
        print(f"   {chain_name.title()} Chain:")
        print(f"     Requests processed: {stats['requests_processed']}")
        print(f"     Success rate: {stats.get('success_rate', 0):.1f}%")
        print(f"     Avg processing time: {stats.get('average_processing_time', 0):.4f}s")
    
    print()
    
    # Show handler statistics for authentication chain
    print("   Handler Statistics (Authentication Chain):")
    handler_stats = chain_manager.get_handler_statistics('authentication')
    
    for stats in handler_stats:
        print(f"   {stats['name']}:")
        print(f"     Processed: {stats['processed_count']} requests")
        print(f"     Avg time: {stats['average_processing_time']:.4f}s")
        print(f"     Next handler: {stats['next_handler'] or 'None'}")
    
    print()
    
    # 5. Dynamic Chain Modification
    print("5. DYNAMIC CHAIN MODIFICATION:")
    
    # Create a simple chain
    simple_chain = (ChainBuilder()
                    .add_handler(InputValidationHandler())
                    .add_handler(AuthenticationHandler())
                    .build())
    
    print("   Original chain: InputValidation → Authentication")
    
    # Test with original chain
    test_request = AuthenticationRequest('alice', 'password123')
    
    try:
        result = simple_chain.handle(test_request)
        print(f"   ✓ Original chain result: {result}")
    except Exception as e:
        print(f"   ✗ Original chain failed: {e}")
    
    # Add rate limiting to the beginning
    rate_limiter = RateLimitHandler()
    rate_limiter.set_next(simple_chain)
    
    print(f"\n   Modified chain: RateLimit → InputValidation → Authentication")
    
    # Test with modified chain
    test_request2 = AuthenticationRequest('bob', 'secret456')
    
    try:
        result = rate_limiter.handle(test_request2)
        print(f"   ✓ Modified chain result: {result}")
    except Exception as e:
        print(f"   ✗ Modified chain failed: {e}")
    
    print()
    
    # 6. Chain of Responsibility Pattern Benefits
    print("6. CHAIN OF RESPONSIBILITY PATTERN BENEFITS:")
    print("   ✓ Decoupling: Sender doesn't need to know which handler processes the request")
    print("   ✓ Flexibility: Handlers can be added, removed, or reordered dynamically")
    print("   ✓ Single Responsibility: Each handler has a specific responsibility")
    print("   ✓ Chain Composition: Complex processing pipelines can be built easily")
    print("   ✓ Extensibility: New handlers can be added without modifying existing code")
    print("   ✓ Fallback Handling: Requests can fall through to default handlers")
    print("   ✓ Conditional Processing: Handlers can choose whether to process requests")
    print("   ✓ Request Enrichment: Handlers can add metadata to requests")
    print("   ✓ Error Handling: Failed handlers can pass requests to next handler")
    print("   ✓ Monitoring: Processing history provides audit trail")
    print()
    
    print("=== CHAIN OF RESPONSIBILITY PATTERN DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_chain_of_responsibility_pattern()
