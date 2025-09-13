"""
RATE LIMITER DESIGN - Complete System Design
===========================================

Problem Statement:
Design a comprehensive rate limiting system that handles:
- Multiple rate limiting algorithms (Token Bucket, Leaky Bucket, Fixed Window, Sliding Window)
- Different rate limiting scopes (per user, per IP, per API key, global)
- Distributed rate limiting across multiple servers
- Rate limit configuration and policy management
- Rate limit monitoring and analytics
- Graceful degradation and fallback mechanisms
- Rate limit bypass for privileged users/operations
- Dynamic rate limit adjustment based on system load
- Rate limit violation handling and penalties
- Integration with API gateways and load balancers

Requirements:
- Support multiple rate limiting algorithms with configurable parameters
- Implement distributed rate limiting with consistency guarantees
- Provide flexible rate limit policies based on various criteria
- Handle high-throughput scenarios with minimal latency
- Support dynamic rate limit updates without service restart
- Provide comprehensive monitoring and alerting
- Handle rate limit violations with appropriate responses
- Support hierarchical rate limiting (global -> user -> operation)
- Implement fair queuing and priority-based rate limiting
- Provide rate limit testing and simulation capabilities

Design Patterns Used:
- Strategy: Different rate limiting algorithms
- Factory: Rate limiter creation based on type
- Observer: Rate limit event monitoring
- Decorator: Rate limit policy application
- Chain of Responsibility: Hierarchical rate limiting
- State: Rate limiter state management
- Command: Rate limit operations
- Proxy: Distributed rate limiter proxy
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import json
import hashlib
import redis
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import math
import uuid


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class RateLimitAlgorithm(Enum):
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW_LOG = "sliding_window_log"
    SLIDING_WINDOW_COUNTER = "sliding_window_counter"


class RateLimitScope(Enum):
    GLOBAL = "global"
    USER = "user"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    TENANT = "tenant"


class RateLimitResult(Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    QUEUED = "queued"


class RateLimitAction(Enum):
    ALLOW = "allow"
    DENY = "deny"
    THROTTLE = "throttle"
    QUEUE = "queue"


@dataclass
class RateLimitRequest:
    """Rate limit request information."""
    key: str
    scope: RateLimitScope
    timestamp: datetime = field(default_factory=datetime.now)
    cost: int = 1
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class RateLimitResponse:
    """Rate limit response information."""
    result: RateLimitResult
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[timedelta] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitPolicy:
    """Rate limit policy configuration."""
    name: str
    algorithm: RateLimitAlgorithm
    scope: RateLimitScope
    limit: int
    window: timedelta
    burst_limit: Optional[int] = None
    priority: int = 0
    active: bool = True
    
    # Advanced settings
    fair_queuing: bool = False
    queue_size: int = 100
    bypass_keys: List[str] = field(default_factory=list)
    penalty_multiplier: float = 1.0
    
    # Dynamic adjustment
    auto_scale: bool = False
    min_limit: int = 1
    max_limit: int = 10000
    scale_factor: float = 0.1


@dataclass
class RateLimitStats:
    """Rate limit statistics."""
    total_requests: int = 0
    allowed_requests: int = 0
    denied_requests: int = 0
    queued_requests: int = 0
    average_response_time_ms: float = 0.0
    current_usage: int = 0
    peak_usage: int = 0
    violations: int = 0
    last_violation: Optional[datetime] = None


# ============================================================================
# RATE LIMITING ALGORITHMS
# ============================================================================

class RateLimiter(ABC):
    """Abstract rate limiter."""
    
    @abstractmethod
    def is_allowed(self, request: RateLimitRequest) -> RateLimitResponse:
        """Check if request is allowed."""
        pass
    
    @abstractmethod
    def get_stats(self) -> RateLimitStats:
        """Get rate limiter statistics."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset rate limiter state."""
        pass


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter implementation."""
    
    def __init__(self, policy: RateLimitPolicy):
        self.policy = policy
        self.bucket_size = policy.burst_limit or policy.limit
        self.refill_rate = policy.limit / policy.window.total_seconds()
        
        self.tokens = float(self.bucket_size)
        self.last_refill = time.time()
        
        self.stats = RateLimitStats()
        self._lock = threading.Lock()
    
    def is_allowed(self, request: RateLimitRequest) -> RateLimitResponse:
        """Check if request is allowed using token bucket."""
        with self._lock:
            now = time.time()
            self._refill_tokens(now)
            
            self.stats.total_requests += 1
            self.stats.current_usage = int(self.bucket_size - self.tokens)
            
            if self.stats.current_usage > self.stats.peak_usage:
                self.stats.peak_usage = self.stats.current_usage
            
            if self.tokens >= request.cost:
                # Allow request
                self.tokens -= request.cost
                self.stats.allowed_requests += 1
                
                remaining = int(self.tokens)
                reset_time = datetime.now() + timedelta(
                    seconds=(self.bucket_size - self.tokens) / self.refill_rate
                )
                
                return RateLimitResponse(
                    result=RateLimitResult.ALLOWED,
                    allowed=True,
                    remaining=remaining,
                    reset_time=reset_time,
                    message="Request allowed"
                )
            else:
                # Deny request
                self.stats.denied_requests += 1
                self.stats.violations += 1
                self.stats.last_violation = datetime.now()
                
                retry_after = timedelta(
                    seconds=(request.cost - self.tokens) / self.refill_rate
                )
                
                return RateLimitResponse(
                    result=RateLimitResult.DENIED,
                    allowed=False,
                    remaining=0,
                    reset_time=datetime.now() + retry_after,
                    retry_after=retry_after,
                    message="Rate limit exceeded"
                )
    
    def _refill_tokens(self, now: float) -> None:
        """Refill tokens based on elapsed time."""
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.bucket_size, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_stats(self) -> RateLimitStats:
        """Get token bucket statistics."""
        return self.stats
    
    def reset(self) -> None:
        """Reset token bucket state."""
        with self._lock:
            self.tokens = float(self.bucket_size)
            self.last_refill = time.time()
            self.stats = RateLimitStats()


class LeakyBucketRateLimiter(RateLimiter):
    """Leaky bucket rate limiter implementation."""
    
    def __init__(self, policy: RateLimitPolicy):
        self.policy = policy
        self.bucket_size = policy.burst_limit or policy.limit
        self.leak_rate = policy.limit / policy.window.total_seconds()
        
        self.queue: deque = deque()
        self.last_leak = time.time()
        
        self.stats = RateLimitStats()
        self._lock = threading.Lock()
    
    def is_allowed(self, request: RateLimitRequest) -> RateLimitResponse:
        """Check if request is allowed using leaky bucket."""
        with self._lock:
            now = time.time()
            self._leak_requests(now)
            
            self.stats.total_requests += 1
            self.stats.current_usage = len(self.queue)
            
            if self.stats.current_usage > self.stats.peak_usage:
                self.stats.peak_usage = self.stats.current_usage
            
            if len(self.queue) < self.bucket_size:
                # Add request to queue
                self.queue.append(request)
                
                if self.policy.fair_queuing:
                    self.stats.queued_requests += 1
                    return RateLimitResponse(
                        result=RateLimitResult.QUEUED,
                        allowed=False,
                        remaining=self.bucket_size - len(self.queue),
                        reset_time=datetime.now() + timedelta(seconds=len(self.queue) / self.leak_rate),
                        message="Request queued"
                    )
                else:
                    self.stats.allowed_requests += 1
                    return RateLimitResponse(
                        result=RateLimitResult.ALLOWED,
                        allowed=True,
                        remaining=self.bucket_size - len(self.queue),
                        reset_time=datetime.now() + timedelta(seconds=len(self.queue) / self.leak_rate),
                        message="Request allowed"
                    )
            else:
                # Bucket is full
                self.stats.denied_requests += 1
                self.stats.violations += 1
                self.stats.last_violation = datetime.now()
                
                return RateLimitResponse(
                    result=RateLimitResult.DENIED,
                    allowed=False,
                    remaining=0,
                    reset_time=datetime.now() + timedelta(seconds=len(self.queue) / self.leak_rate),
                    retry_after=timedelta(seconds=1 / self.leak_rate),
                    message="Bucket is full"
                )
    
    def _leak_requests(self, now: float) -> None:
        """Leak requests from bucket based on elapsed time."""
        elapsed = now - self.last_leak
        requests_to_leak = int(elapsed * self.leak_rate)
        
        for _ in range(min(requests_to_leak, len(self.queue))):
            self.queue.popleft()
        
        self.last_leak = now
    
    def get_stats(self) -> RateLimitStats:
        """Get leaky bucket statistics."""
        return self.stats
    
    def reset(self) -> None:
        """Reset leaky bucket state."""
        with self._lock:
            self.queue.clear()
            self.last_leak = time.time()
            self.stats = RateLimitStats()


class FixedWindowRateLimiter(RateLimiter):
    """Fixed window rate limiter implementation."""
    
    def __init__(self, policy: RateLimitPolicy):
        self.policy = policy
        self.window_size = policy.window.total_seconds()
        
        self.request_count = 0
        self.window_start = time.time()
        
        self.stats = RateLimitStats()
        self._lock = threading.Lock()
    
    def is_allowed(self, request: RateLimitRequest) -> RateLimitResponse:
        """Check if request is allowed using fixed window."""
        with self._lock:
            now = time.time()
            
            # Check if we need to reset the window
            if now - self.window_start >= self.window_size:
                self.request_count = 0
                self.window_start = now
            
            self.stats.total_requests += 1
            self.stats.current_usage = self.request_count
            
            if self.stats.current_usage > self.stats.peak_usage:
                self.stats.peak_usage = self.stats.current_usage
            
            if self.request_count < self.policy.limit:
                # Allow request
                self.request_count += request.cost
                self.stats.allowed_requests += 1
                
                remaining = self.policy.limit - self.request_count
                reset_time = datetime.fromtimestamp(self.window_start + self.window_size)
                
                return RateLimitResponse(
                    result=RateLimitResult.ALLOWED,
                    allowed=True,
                    remaining=remaining,
                    reset_time=reset_time,
                    message="Request allowed"
                )
            else:
                # Deny request
                self.stats.denied_requests += 1
                self.stats.violations += 1
                self.stats.last_violation = datetime.now()
                
                reset_time = datetime.fromtimestamp(self.window_start + self.window_size)
                retry_after = reset_time - datetime.now()
                
                return RateLimitResponse(
                    result=RateLimitResult.DENIED,
                    allowed=False,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=retry_after,
                    message="Fixed window limit exceeded"
                )
    
    def get_stats(self) -> RateLimitStats:
        """Get fixed window statistics."""
        return self.stats
    
    def reset(self) -> None:
        """Reset fixed window state."""
        with self._lock:
            self.request_count = 0
            self.window_start = time.time()
            self.stats = RateLimitStats()


class SlidingWindowLogRateLimiter(RateLimiter):
    """Sliding window log rate limiter implementation."""
    
    def __init__(self, policy: RateLimitPolicy):
        self.policy = policy
        self.window_size = policy.window.total_seconds()
        
        self.request_log: List[float] = []
        
        self.stats = RateLimitStats()
        self._lock = threading.Lock()
    
    def is_allowed(self, request: RateLimitRequest) -> RateLimitResponse:
        """Check if request is allowed using sliding window log."""
        with self._lock:
            now = time.time()
            
            # Remove old requests outside the window
            cutoff_time = now - self.window_size
            self.request_log = [timestamp for timestamp in self.request_log 
                              if timestamp > cutoff_time]
            
            self.stats.total_requests += 1
            self.stats.current_usage = len(self.request_log)
            
            if self.stats.current_usage > self.stats.peak_usage:
                self.stats.peak_usage = self.stats.current_usage
            
            if len(self.request_log) < self.policy.limit:
                # Allow request
                for _ in range(request.cost):
                    self.request_log.append(now)
                
                self.stats.allowed_requests += 1
                
                remaining = self.policy.limit - len(self.request_log)
                reset_time = datetime.fromtimestamp(
                    min(self.request_log) + self.window_size if self.request_log else now + self.window_size
                )
                
                return RateLimitResponse(
                    result=RateLimitResult.ALLOWED,
                    allowed=True,
                    remaining=remaining,
                    reset_time=reset_time,
                    message="Request allowed"
                )
            else:
                # Deny request
                self.stats.denied_requests += 1
                self.stats.violations += 1
                self.stats.last_violation = datetime.now()
                
                # Calculate when the oldest request will expire
                oldest_request = min(self.request_log) if self.request_log else now
                reset_time = datetime.fromtimestamp(oldest_request + self.window_size)
                retry_after = reset_time - datetime.now()
                
                return RateLimitResponse(
                    result=RateLimitResult.DENIED,
                    allowed=False,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=retry_after,
                    message="Sliding window limit exceeded"
                )
    
    def get_stats(self) -> RateLimitStats:
        """Get sliding window log statistics."""
        return self.stats
    
    def reset(self) -> None:
        """Reset sliding window log state."""
        with self._lock:
            self.request_log.clear()
            self.stats = RateLimitStats()


class SlidingWindowCounterRateLimiter(RateLimiter):
    """Sliding window counter rate limiter implementation."""
    
    def __init__(self, policy: RateLimitPolicy, num_buckets: int = 10):
        self.policy = policy
        self.window_size = policy.window.total_seconds()
        self.num_buckets = num_buckets
        self.bucket_size = self.window_size / num_buckets
        
        self.buckets = [0] * num_buckets
        self.bucket_timestamps = [0.0] * num_buckets
        
        self.stats = RateLimitStats()
        self._lock = threading.Lock()
    
    def is_allowed(self, request: RateLimitRequest) -> RateLimitResponse:
        """Check if request is allowed using sliding window counter."""
        with self._lock:
            now = time.time()
            
            # Update buckets
            self._update_buckets(now)
            
            # Calculate current usage
            current_usage = sum(self.buckets)
            
            self.stats.total_requests += 1
            self.stats.current_usage = current_usage
            
            if self.stats.current_usage > self.stats.peak_usage:
                self.stats.peak_usage = self.stats.current_usage
            
            if current_usage < self.policy.limit:
                # Allow request
                bucket_index = int((now % self.window_size) / self.bucket_size)
                self.buckets[bucket_index] += request.cost
                
                self.stats.allowed_requests += 1
                
                remaining = self.policy.limit - (current_usage + request.cost)
                reset_time = datetime.fromtimestamp(now + self.bucket_size)
                
                return RateLimitResponse(
                    result=RateLimitResult.ALLOWED,
                    allowed=True,
                    remaining=remaining,
                    reset_time=reset_time,
                    message="Request allowed"
                )
            else:
                # Deny request
                self.stats.denied_requests += 1
                self.stats.violations += 1
                self.stats.last_violation = datetime.now()
                
                retry_after = timedelta(seconds=self.bucket_size)
                reset_time = datetime.now() + retry_after
                
                return RateLimitResponse(
                    result=RateLimitResult.DENIED,
                    allowed=False,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=retry_after,
                    message="Sliding window counter limit exceeded"
                )
    
    def _update_buckets(self, now: float) -> None:
        """Update bucket counters based on current time."""
        for i in range(self.num_buckets):
            bucket_start = (now // self.bucket_size) * self.bucket_size - (i * self.bucket_size)
            
            # Reset buckets that are outside the current window
            if now - bucket_start >= self.window_size:
                self.buckets[i] = 0
                self.bucket_timestamps[i] = bucket_start
    
    def get_stats(self) -> RateLimitStats:
        """Get sliding window counter statistics."""
        return self.stats
    
    def reset(self) -> None:
        """Reset sliding window counter state."""
        with self._lock:
            self.buckets = [0] * self.num_buckets
            self.bucket_timestamps = [0.0] * self.num_buckets
            self.stats = RateLimitStats()


# ============================================================================
# RATE LIMITER FACTORY
# ============================================================================

class RateLimiterFactory:
    """Factory for creating rate limiters."""
    
    @staticmethod
    def create_rate_limiter(policy: RateLimitPolicy) -> RateLimiter:
        """Create rate limiter based on policy algorithm."""
        if policy.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return TokenBucketRateLimiter(policy)
        elif policy.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            return LeakyBucketRateLimiter(policy)
        elif policy.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return FixedWindowRateLimiter(policy)
        elif policy.algorithm == RateLimitAlgorithm.SLIDING_WINDOW_LOG:
            return SlidingWindowLogRateLimiter(policy)
        elif policy.algorithm == RateLimitAlgorithm.SLIDING_WINDOW_COUNTER:
            return SlidingWindowCounterRateLimiter(policy)
        else:
            raise ValueError(f"Unsupported rate limit algorithm: {policy.algorithm}")


# ============================================================================
# DISTRIBUTED RATE LIMITER
# ============================================================================

class DistributedRateLimiter:
    """Distributed rate limiter using Redis."""
    
    def __init__(self, policy: RateLimitPolicy, redis_client=None):
        self.policy = policy
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        self.stats = RateLimitStats()
        
        # Lua scripts for atomic operations
        self.token_bucket_script = self._create_token_bucket_script()
        self.fixed_window_script = self._create_fixed_window_script()
    
    def _create_token_bucket_script(self) -> str:
        """Create Lua script for atomic token bucket operations."""
        return """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local tokens = tonumber(ARGV[2])
        local interval = tonumber(ARGV[3])
        local cost = tonumber(ARGV[4])
        local now = tonumber(ARGV[5])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local current_tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now
        
        -- Refill tokens
        local elapsed = now - last_refill
        local tokens_to_add = elapsed * tokens / interval
        current_tokens = math.min(capacity, current_tokens + tokens_to_add)
        
        local allowed = 0
        local remaining = current_tokens
        
        if current_tokens >= cost then
            current_tokens = current_tokens - cost
            allowed = 1
            remaining = current_tokens
        end
        
        -- Update bucket state
        redis.call('HMSET', key, 'tokens', current_tokens, 'last_refill', now)
        redis.call('EXPIRE', key, interval * 2)
        
        return {allowed, remaining, current_tokens}
        """
    
    def _create_fixed_window_script(self) -> str:
        """Create Lua script for atomic fixed window operations."""
        return """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local cost = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        local window_start = math.floor(now / window) * window
        local window_key = key .. ':' .. window_start
        
        local current = tonumber(redis.call('GET', window_key)) or 0
        local allowed = 0
        local remaining = limit - current
        
        if current + cost <= limit then
            current = current + cost
            redis.call('SET', window_key, current)
            redis.call('EXPIRE', window_key, window * 2)
            allowed = 1
            remaining = limit - current
        end
        
        return {allowed, remaining, current}
        """
    
    def is_allowed(self, request: RateLimitRequest) -> RateLimitResponse:
        """Check if request is allowed using distributed rate limiter."""
        key = self._generate_key(request)
        now = time.time()
        
        try:
            if self.policy.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                result = self._check_token_bucket(key, request, now)
            elif self.policy.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                result = self._check_fixed_window(key, request, now)
            else:
                # Fallback to local rate limiter
                local_limiter = RateLimiterFactory.create_rate_limiter(self.policy)
                result = local_limiter.is_allowed(request)
            
            # Update statistics
            self.stats.total_requests += 1
            if result.allowed:
                self.stats.allowed_requests += 1
            else:
                self.stats.denied_requests += 1
                self.stats.violations += 1
                self.stats.last_violation = datetime.now()
            
            return result
            
        except Exception as e:
            # Fallback to allow on Redis failure
            print(f"Distributed rate limiter error: {e}")
            self.stats.total_requests += 1
            self.stats.allowed_requests += 1
            
            return RateLimitResponse(
                result=RateLimitResult.ALLOWED,
                allowed=True,
                remaining=self.policy.limit,
                reset_time=datetime.now() + self.policy.window,
                message="Fallback: Redis error"
            )
    
    def _generate_key(self, request: RateLimitRequest) -> str:
        """Generate Redis key for rate limiting."""
        return f"rate_limit:{self.policy.name}:{request.scope.value}:{request.key}"
    
    def _check_token_bucket(self, key: str, request: RateLimitRequest, now: float) -> RateLimitResponse:
        """Check token bucket using Redis Lua script."""
        capacity = self.policy.burst_limit or self.policy.limit
        tokens_per_second = self.policy.limit / self.policy.window.total_seconds()
        interval = self.policy.window.total_seconds()
        
        result = self.redis_client.eval(
            self.token_bucket_script,
            1,
            key,
            capacity,
            tokens_per_second,
            interval,
            request.cost,
            now
        )
        
        allowed, remaining, current_tokens = result
        
        return RateLimitResponse(
            result=RateLimitResult.ALLOWED if allowed else RateLimitResult.DENIED,
            allowed=bool(allowed),
            remaining=int(remaining),
            reset_time=datetime.now() + timedelta(
                seconds=(capacity - current_tokens) / tokens_per_second
            ),
            message="Token bucket check"
        )
    
    def _check_fixed_window(self, key: str, request: RateLimitRequest, now: float) -> RateLimitResponse:
        """Check fixed window using Redis Lua script."""
        window_seconds = self.policy.window.total_seconds()
        
        result = self.redis_client.eval(
            self.fixed_window_script,
            1,
            key,
            self.policy.limit,
            window_seconds,
            request.cost,
            now
        )
        
        allowed, remaining, current = result
        
        window_start = math.floor(now / window_seconds) * window_seconds
        reset_time = datetime.fromtimestamp(window_start + window_seconds)
        
        return RateLimitResponse(
            result=RateLimitResult.ALLOWED if allowed else RateLimitResult.DENIED,
            allowed=bool(allowed),
            remaining=int(remaining),
            reset_time=reset_time,
            retry_after=reset_time - datetime.now() if not allowed else None,
            message="Fixed window check"
        )
    
    def get_stats(self) -> RateLimitStats:
        """Get distributed rate limiter statistics."""
        return self.stats
    
    def reset(self) -> None:
        """Reset distributed rate limiter state."""
        self.stats = RateLimitStats()


# ============================================================================
# HIERARCHICAL RATE LIMITER
# ============================================================================

class HierarchicalRateLimiter:
    """Hierarchical rate limiter with multiple levels."""
    
    def __init__(self):
        self.limiters: List[Tuple[RateLimiter, RateLimitPolicy]] = []
        self.stats = RateLimitStats()
    
    def add_limiter(self, limiter: RateLimiter, policy: RateLimitPolicy) -> None:
        """Add rate limiter to hierarchy."""
        self.limiters.append((limiter, policy))
        # Sort by priority (higher priority first)
        self.limiters.sort(key=lambda x: x[1].priority, reverse=True)
    
    def is_allowed(self, request: RateLimitRequest) -> RateLimitResponse:
        """Check request against all rate limiters in hierarchy."""
        self.stats.total_requests += 1
        
        # Check bypass keys
        for limiter, policy in self.limiters:
            if request.key in policy.bypass_keys:
                continue
        
        # Apply rate limiters in priority order
        for limiter, policy in self.limiters:
            if not policy.active:
                continue
            
            # Check if this limiter applies to the request scope
            if policy.scope != RateLimitScope.GLOBAL and policy.scope != request.scope:
                continue
            
            response = limiter.is_allowed(request)
            
            if not response.allowed:
                # Request denied by this limiter
                self.stats.denied_requests += 1
                self.stats.violations += 1
                self.stats.last_violation = datetime.now()
                
                # Apply penalty multiplier
                if policy.penalty_multiplier > 1.0:
                    if response.retry_after:
                        response.retry_after = timedelta(
                            seconds=response.retry_after.total_seconds() * policy.penalty_multiplier
                        )
                
                response.metadata["denied_by"] = policy.name
                return response
        
        # All limiters allowed the request
        self.stats.allowed_requests += 1
        
        # Return the most restrictive remaining count
        min_remaining = float('inf')
        earliest_reset = None
        
        for limiter, policy in self.limiters:
            if policy.active:
                limiter_response = limiter.is_allowed(
                    RateLimitRequest(request.key, request.scope, request.timestamp, 0)
                )
                if limiter_response.remaining < min_remaining:
                    min_remaining = limiter_response.remaining
                if earliest_reset is None or limiter_response.reset_time < earliest_reset:
                    earliest_reset = limiter_response.reset_time
        
        return RateLimitResponse(
            result=RateLimitResult.ALLOWED,
            allowed=True,
            remaining=int(min_remaining) if min_remaining != float('inf') else 0,
            reset_time=earliest_reset or datetime.now(),
            message="Request allowed by all limiters"
        )
    
    def get_stats(self) -> RateLimitStats:
        """Get hierarchical rate limiter statistics."""
        return self.stats
    
    def reset(self) -> None:
        """Reset all rate limiters in hierarchy."""
        for limiter, _ in self.limiters:
            limiter.reset()
        self.stats = RateLimitStats()


# ============================================================================
# RATE LIMIT MANAGER
# ============================================================================

class RateLimitManager:
    """Central rate limit management system."""
    
    def __init__(self, use_distributed: bool = False, redis_client=None):
        self.use_distributed = use_distributed
        self.redis_client = redis_client
        
        self.policies: Dict[str, RateLimitPolicy] = {}
        self.limiters: Dict[str, RateLimiter] = {}
        self.hierarchical_limiters: Dict[str, HierarchicalRateLimiter] = {}
        
        self.global_stats = RateLimitStats()
        self._lock = threading.Lock()
        
        print(f"🚦 Rate Limit Manager initialized (distributed={use_distributed})")
    
    def add_policy(self, policy: RateLimitPolicy) -> None:
        """Add rate limit policy."""
        with self._lock:
            self.policies[policy.name] = policy
            
            # Create appropriate rate limiter
            if self.use_distributed:
                limiter = DistributedRateLimiter(policy, self.redis_client)
            else:
                limiter = RateLimiterFactory.create_rate_limiter(policy)
            
            self.limiters[policy.name] = limiter
    
    def create_hierarchical_limiter(self, name: str, policy_names: List[str]) -> HierarchicalRateLimiter:
        """Create hierarchical rate limiter from multiple policies."""
        hierarchical = HierarchicalRateLimiter()
        
        for policy_name in policy_names:
            if policy_name in self.policies and policy_name in self.limiters:
                policy = self.policies[policy_name]
                limiter = self.limiters[policy_name]
                hierarchical.add_limiter(limiter, policy)
        
        self.hierarchical_limiters[name] = hierarchical
        return hierarchical
    
    def check_rate_limit(self, request: RateLimitRequest, 
                        limiter_name: str = None) -> RateLimitResponse:
        """Check rate limit for request."""
        with self._lock:
            self.global_stats.total_requests += 1
            
            if limiter_name:
                # Use specific limiter
                if limiter_name in self.limiters:
                    response = self.limiters[limiter_name].is_allowed(request)
                elif limiter_name in self.hierarchical_limiters:
                    response = self.hierarchical_limiters[limiter_name].is_allowed(request)
                else:
                    response = RateLimitResponse(
                        result=RateLimitResult.ALLOWED,
                        allowed=True,
                        remaining=1000,
                        reset_time=datetime.now() + timedelta(hours=1),
                        message="Limiter not found, allowing request"
                    )
            else:
                # Find applicable limiter based on scope
                applicable_limiters = []
                
                for name, policy in self.policies.items():
                    if policy.active and (policy.scope == request.scope or 
                                        policy.scope == RateLimitScope.GLOBAL):
                        applicable_limiters.append((self.limiters[name], policy))
                
                if applicable_limiters:
                    # Use hierarchical approach
                    temp_hierarchical = HierarchicalRateLimiter()
                    for limiter, policy in applicable_limiters:
                        temp_hierarchical.add_limiter(limiter, policy)
                    
                    response = temp_hierarchical.is_allowed(request)
                else:
                    # No applicable limiters, allow request
                    response = RateLimitResponse(
                        result=RateLimitResult.ALLOWED,
                        allowed=True,
                        remaining=1000,
                        reset_time=datetime.now() + timedelta(hours=1),
                        message="No applicable rate limiters"
                    )
            
            # Update global statistics
            if response.allowed:
                self.global_stats.allowed_requests += 1
            else:
                self.global_stats.denied_requests += 1
                self.global_stats.violations += 1
                self.global_stats.last_violation = datetime.now()
            
            return response
    
    def update_policy(self, policy_name: str, updates: Dict[str, Any]) -> bool:
        """Update existing rate limit policy."""
        with self._lock:
            if policy_name not in self.policies:
                return False
            
            policy = self.policies[policy_name]
            
            # Update policy attributes
            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            
            # Recreate limiter with updated policy
            if self.use_distributed:
                limiter = DistributedRateLimiter(policy, self.redis_client)
            else:
                limiter = RateLimiterFactory.create_rate_limiter(policy)
            
            self.limiters[policy_name] = limiter
            return True
    
    def get_policy_stats(self, policy_name: str) -> Optional[RateLimitStats]:
        """Get statistics for specific policy."""
        if policy_name in self.limiters:
            return self.limiters[policy_name].get_stats()
        return None
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all rate limiting statistics."""
        policy_stats = {}
        for name, limiter in self.limiters.items():
            policy_stats[name] = limiter.get_stats()
        
        hierarchical_stats = {}
        for name, limiter in self.hierarchical_limiters.items():
            hierarchical_stats[name] = limiter.get_stats()
        
        return {
            "global": self.global_stats,
            "policies": policy_stats,
            "hierarchical": hierarchical_stats,
            "policy_count": len(self.policies),
            "active_policies": sum(1 for p in self.policies.values() if p.active)
        }
    
    def reset_all(self) -> None:
        """Reset all rate limiters."""
        with self._lock:
            for limiter in self.limiters.values():
                limiter.reset()
            
            for limiter in self.hierarchical_limiters.values():
                limiter.reset()
            
            self.global_stats = RateLimitStats()


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_rate_limiter():
    """Demonstrate the rate limiter system."""
    print("=== RATE LIMITER SYSTEM DEMONSTRATION ===\n")
    
    # Create rate limit manager
    print("1. RATE LIMIT MANAGER SETUP:")
    
    manager = RateLimitManager(use_distributed=False)
    print("   ✓ Rate limit manager initialized")
    print()
    
    # Create different rate limit policies
    print("2. RATE LIMIT POLICY CREATION:")
    
    # API rate limit policy
    api_policy = RateLimitPolicy(
        name="api_requests",
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        scope=RateLimitScope.API_KEY,
        limit=100,
        window=timedelta(minutes=1),
        burst_limit=150,
        priority=10
    )
    manager.add_policy(api_policy)
    print("   ✓ Created API rate limit policy (100 req/min, burst 150)")
    
    # User rate limit policy
    user_policy = RateLimitPolicy(
        name="user_requests",
        algorithm=RateLimitAlgorithm.SLIDING_WINDOW_LOG,
        scope=RateLimitScope.USER,
        limit=50,
        window=timedelta(minutes=1),
        priority=20
    )
    manager.add_policy(user_policy)
    print("   ✓ Created user rate limit policy (50 req/min)")
    
    # IP rate limit policy
    ip_policy = RateLimitPolicy(
        name="ip_requests",
        algorithm=RateLimitAlgorithm.FIXED_WINDOW,
        scope=RateLimitScope.IP_ADDRESS,
        limit=200,
        window=timedelta(minutes=1),
        priority=5
    )
    manager.add_policy(ip_policy)
    print("   ✓ Created IP rate limit policy (200 req/min)")
    
    # Global rate limit policy
    global_policy = RateLimitPolicy(
        name="global_requests",
        algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
        scope=RateLimitScope.GLOBAL,
        limit=1000,
        window=timedelta(minutes=1),
        priority=1
    )
    manager.add_policy(global_policy)
    print("   ✓ Created global rate limit policy (1000 req/min)")
    
    print()
    
    # Test individual rate limiters
    print("3. INDIVIDUAL RATE LIMITER TESTING:")
    
    # Test API rate limiter
    print("   API Rate Limiter Test:")
    api_requests = []
    for i in range(10):
        request = RateLimitRequest(
            key="api_key_123",
            scope=RateLimitScope.API_KEY
        )
        response = manager.check_rate_limit(request, "api_requests")
        api_requests.append(response.allowed)
        
        if i < 3:  # Show first few results
            print(f"     Request {i+1}: {'✓ Allowed' if response.allowed else '✗ Denied'} "
                  f"(remaining: {response.remaining})")
    
    print(f"     Total allowed: {sum(api_requests)}/10")
    
    # Test user rate limiter
    print("\n   User Rate Limiter Test:")
    user_requests = []
    for i in range(8):
        request = RateLimitRequest(
            key="user_456",
            scope=RateLimitScope.USER
        )
        response = manager.check_rate_limit(request, "user_requests")
        user_requests.append(response.allowed)
    
    print(f"     Total allowed: {sum(user_requests)}/8")
    
    print()
    
    # Test hierarchical rate limiting
    print("4. HIERARCHICAL RATE LIMITING TEST:")
    
    hierarchical = manager.create_hierarchical_limiter(
        "api_hierarchy", 
        ["global_requests", "api_requests", "user_requests"]
    )
    
    print("   Created hierarchical limiter (global -> api -> user)")
    
    # Test hierarchical requests
    hierarchical_results = []
    for i in range(15):
        request = RateLimitRequest(
            key="hierarchical_test",
            scope=RateLimitScope.API_KEY
        )
        response = manager.check_rate_limit(request, "api_hierarchy")
        hierarchical_results.append(response.allowed)
        
        if i < 5:  # Show first few results
            print(f"   Request {i+1}: {'✓ Allowed' if response.allowed else '✗ Denied'}")
    
    print(f"   Total allowed: {sum(hierarchical_results)}/15")
    print()
    
    # Test different algorithms
    print("5. ALGORITHM COMPARISON TEST:")
    
    algorithms = [
        (RateLimitAlgorithm.TOKEN_BUCKET, "Token Bucket"),
        (RateLimitAlgorithm.LEAKY_BUCKET, "Leaky Bucket"),
        (RateLimitAlgorithm.FIXED_WINDOW, "Fixed Window"),
        (RateLimitAlgorithm.SLIDING_WINDOW_LOG, "Sliding Window Log"),
        (RateLimitAlgorithm.SLIDING_WINDOW_COUNTER, "Sliding Window Counter")
    ]
    
    for algorithm, name in algorithms:
        # Create test policy
        test_policy = RateLimitPolicy(
            name=f"test_{algorithm.value}",
            algorithm=algorithm,
            scope=RateLimitScope.USER,
            limit=5,
            window=timedelta(seconds=10)
        )
        
        # Create rate limiter
        limiter = RateLimiterFactory.create_rate_limiter(test_policy)
        
        # Test burst of requests
        allowed_count = 0
        for i in range(8):
            request = RateLimitRequest(key="test_user", scope=RateLimitScope.USER)
            response = limiter.is_allowed(request)
            if response.allowed:
                allowed_count += 1
        
        print(f"   {name}: {allowed_count}/8 requests allowed")
    
    print()
    
    # Test rate limit updates
    print("6. DYNAMIC POLICY UPDATE TEST:")
    
    # Update API policy
    updates = {
        "limit": 150,
        "burst_limit": 200
    }
    success = manager.update_policy("api_requests", updates)
    print(f"   Updated API policy: {'✓ Success' if success else '✗ Failed'}")
    
    # Test with updated policy
    request = RateLimitRequest(key="api_key_updated", scope=RateLimitScope.API_KEY)
    response = manager.check_rate_limit(request, "api_requests")
    print(f"   Request with updated policy: {'✓ Allowed' if response.allowed else '✗ Denied'}")
    
    print()
    
    # Test burst handling
    print("7. BURST HANDLING TEST:")
    
    burst_policy = RateLimitPolicy(
        name="burst_test",
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        scope=RateLimitScope.USER,
        limit=10,
        window=timedelta(seconds=10),
        burst_limit=20
    )
    manager.add_policy(burst_policy)
    
    # Send burst of requests
    burst_results = []
    for i in range(25):
        request = RateLimitRequest(key="burst_user", scope=RateLimitScope.USER)
        response = manager.check_rate_limit(request, "burst_test")
        burst_results.append(response.allowed)
    
    print(f"   Burst test: {sum(burst_results)}/25 requests allowed")
    print(f"   Expected: ~20 (burst limit)")
    
    print()
    
    # Test rate limiting with different costs
    print("8. WEIGHTED REQUEST TEST:")
    
    weighted_results = []
    costs = [1, 1, 5, 1, 1, 10, 1, 1]  # Different request costs
    
    for i, cost in enumerate(costs):
        request = RateLimitRequest(
            key="weighted_user", 
            scope=RateLimitScope.USER, 
            cost=cost
        )
        response = manager.check_rate_limit(request, "burst_test")
        weighted_results.append(response.allowed)
        
        print(f"   Request {i+1} (cost {cost}): {'✓ Allowed' if response.allowed else '✗ Denied'}")
    
    print()
    
    # Performance test
    print("9. PERFORMANCE TEST:")
    
    import time
    
    perf_policy = RateLimitPolicy(
        name="performance_test",
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        scope=RateLimitScope.USER,
        limit=10000,
        window=timedelta(minutes=1)
    )
    manager.add_policy(perf_policy)
    
    # Test throughput
    start_time = time.time()
    request_count = 1000
    
    for i in range(request_count):
        request = RateLimitRequest(key=f"perf_user_{i % 100}", scope=RateLimitScope.USER)
        manager.check_rate_limit(request, "performance_test")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"   Processed {request_count} requests in {duration:.3f} seconds")
    print(f"   Throughput: {request_count/duration:.0f} requests/second")
    
    print()
    
    # Show comprehensive statistics
    print("10. COMPREHENSIVE STATISTICS:")
    
    all_stats = manager.get_all_stats()
    
    print(f"   Global Statistics:")
    global_stats = all_stats["global"]
    print(f"     Total requests: {global_stats.total_requests}")
    print(f"     Allowed: {global_stats.allowed_requests}")
    print(f"     Denied: {global_stats.denied_requests}")
    print(f"     Violations: {global_stats.violations}")
    
    print(f"\n   Policy Statistics:")
    for policy_name, stats in all_stats["policies"].items():
        print(f"     {policy_name}:")
        print(f"       Requests: {stats.total_requests}")
        print(f"       Allowed: {stats.allowed_requests}")
        print(f"       Denied: {stats.denied_requests}")
        print(f"       Violations: {stats.violations}")
    
    print(f"\n   System Information:")
    print(f"     Total policies: {all_stats['policy_count']}")
    print(f"     Active policies: {all_stats['active_policies']}")
    print(f"     Hierarchical limiters: {len(all_stats['hierarchical'])}")
    
    print()
    
    # Test rate limit bypass
    print("11. RATE LIMIT BYPASS TEST:")
    
    # Add bypass key to policy
    bypass_policy = RateLimitPolicy(
        name="bypass_test",
        algorithm=RateLimitAlgorithm.FIXED_WINDOW,
        scope=RateLimitScope.USER,
        limit=3,
        window=timedelta(seconds=10),
        bypass_keys=["privileged_user"]
    )
    manager.add_policy(bypass_policy)
    
    # Test regular user (should be limited)
    regular_allowed = 0
    for i in range(5):
        request = RateLimitRequest(key="regular_user", scope=RateLimitScope.USER)
        response = manager.check_rate_limit(request, "bypass_test")
        if response.allowed:
            regular_allowed += 1
    
    # Test privileged user (should bypass)
    privileged_allowed = 0
    for i in range(5):
        request = RateLimitRequest(key="privileged_user", scope=RateLimitScope.USER)
        response = manager.check_rate_limit(request, "bypass_test")
        if response.allowed:
            privileged_allowed += 1
    
    print(f"   Regular user: {regular_allowed}/5 requests allowed")
    print(f"   Privileged user: {privileged_allowed}/5 requests allowed")
    
    print()
    
    # Test cleanup
    print("12. CLEANUP TEST:")
    
    manager.reset_all()
    
    final_stats = manager.get_all_stats()
    total_requests = sum(stats.total_requests for stats in final_stats["policies"].values())
    
    print(f"   Reset all rate limiters")
    print(f"   Total requests after reset: {total_requests}")
    
    print()
    print("=== RATE LIMITER DEMONSTRATION COMPLETE ===")


if __name__ == "__main__":
    demonstrate_rate_limiter()
