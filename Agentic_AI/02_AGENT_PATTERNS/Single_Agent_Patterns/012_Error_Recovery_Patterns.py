#!/usr/bin/env python3
"""
Error Recovery Patterns: Graceful Handling of Failures and Errors
================================================================

WHAT IS THE PROBLEM?
==================
Most systems break completely when something goes wrong, providing poor user experience and losing data.

Example: Online Banking Without Error Recovery
- Network connection drops during money transfer
- System crashes and user doesn't know if transfer completed
- User tries again, potentially transferring money twice
- No way to recover or check transaction status
- Customer loses trust and money

REAL WORLD EXAMPLE:
=================
How does Amazon's payment system handle errors?

ERROR RECOVERY PATTERNS:

1. RETRY WITH BACKOFF:
- Network timeout? Retry after 1 second
- Still failing? Retry after 2 seconds
- Still failing? Retry after 4 seconds
- Give up after reasonable attempts

2. CIRCUIT BREAKER:
- Service failing frequently? Stop calling it temporarily
- Let it recover instead of overwhelming it
- Periodically test if it's back online

3. GRACEFUL DEGRADATION:
- Recommendation service down? Show default products
- Search service slow? Show cached results
- Payment service error? Allow order without immediate payment

4. TRANSACTION ROLLBACK:
- Transfer fails halfway? Undo all changes
- Order processing error? Restore inventory
- Keep system in consistent state

THE ALGORITHM:
=============
1. TRY: Attempt the operation
2. DETECT: Monitor for errors and failures
3. CLASSIFY: Determine error type and severity
4. RECOVER: Apply appropriate recovery strategy
5. FALLBACK: Use alternative approach if recovery fails
6. LEARN: Update error handling based on patterns

PSEUDO CODE:
===========
def resilient_operation(operation, data):
    max_retries = 3
    circuit_breaker = CircuitBreaker()
    
    for attempt in range(max_retries):
        try:
            # Check if service is available
            if circuit_breaker.is_open():
                return fallback_operation(data)
            
            # Attempt operation
            result = operation(data)
            
            # Success - reset error counters
            circuit_breaker.record_success()
            return result
            
        except RetryableError as e:
            # Temporary error - retry with backoff
            if attempt < max_retries - 1:
                backoff_time = (2 ** attempt) * random.uniform(0.5, 1.5)
                time.sleep(backoff_time)
                continue
            else:
                circuit_breaker.record_failure()
                return fallback_operation(data)
                
        except FatalError as e:
            # Permanent error - don't retry
            circuit_breaker.record_failure()
            return error_response(e)
    
    return fallback_operation(data)

WHY IS THIS CRITICAL?
===================
- Prevents cascading failures that bring down entire systems
- Provides better user experience during outages
- Maintains data consistency even when things go wrong
- Reduces manual intervention and support costs
- Builds user trust through reliable operation
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import traceback

class ErrorType(Enum):
    TRANSIENT = "transient"        # Temporary errors that might resolve
    PERSISTENT = "persistent"      # Errors that are likely to continue
    FATAL = "fatal"               # Errors that require manual intervention
    RESOURCE = "resource"         # Resource exhaustion or unavailability
    TIMEOUT = "timeout"           # Operation took too long
    VALIDATION = "validation"     # Input data problems
    EXTERNAL = "external"         # Third-party service failures

class RecoveryStrategy(Enum):
    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    TRANSACTION_ROLLBACK = "transaction_rollback"
    ASYNC_RETRY = "async_retry"

@dataclass
class ErrorEvent:
    """Record of an error occurrence"""
    error_id: str
    error_type: ErrorType
    error_message: str
    operation_name: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    retry_count: int = 0

@dataclass
class RecoveryResult:
    """Result of error recovery attempt"""
    success: bool
    strategy_used: RecoveryStrategy
    final_result: Any
    recovery_time: float
    error_details: Optional[str] = None
    fallback_used: bool = False

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        current_time = time.time()
        
        if self.state == "closed":
            return True
        elif self.state == "open":
            # Check if timeout has passed
            if current_time - self.last_failure_time >= self.timeout:
                self.state = "half_open"
                self.success_count = 0
                return True
            return False
        elif self.state == "half_open":
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record successful operation"""
        if self.state == "half_open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "closed"
                self.failure_count = 0
        elif self.state == "closed":
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self) -> None:
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }

class ErrorRecoveryAgent:
    """
    An agent that implements robust error recovery patterns
    
    EXAMPLE USAGE:
    =============
    # Create error recovery agent
    agent = ErrorRecoveryAgent("payment_processor")
    
    # Register operations with recovery strategies
    agent.register_operation("process_payment", PaymentOperation())
    agent.register_fallback("process_payment", process_payment_offline)
    
    # Execute with automatic error recovery
    result = await agent.execute("process_payment", payment_data)
    """
    
    def __init__(self, agent_id: str, max_retries: int = 3, base_backoff: float = 1.0):
        self.agent_id = agent_id
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        
        # Operation registry
        self.operations: Dict[str, Callable] = {}
        self.fallback_operations: Dict[str, Callable] = {}
        self.recovery_strategies: Dict[str, List[RecoveryStrategy]] = {}
        
        # Error tracking
        self.error_history: List[ErrorEvent] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
        
        # Recovery configuration
        self.strategy_config = {
            RecoveryStrategy.RETRY: {"max_retries": max_retries, "backoff_multiplier": 2.0},
            RecoveryStrategy.CIRCUIT_BREAKER: {"failure_threshold": 5, "timeout": 60.0},
            RecoveryStrategy.FALLBACK: {"timeout": 30.0},
            RecoveryStrategy.GRACEFUL_DEGRADATION: {"quality_threshold": 0.5},
        }
    
    def register_operation(self, operation_name: str, operation_func: Callable,
                          recovery_strategies: List[RecoveryStrategy] = None) -> None:
        """Register an operation with its recovery strategies"""
        self.operations[operation_name] = operation_func
        
        # Default recovery strategies if none provided
        if recovery_strategies is None:
            recovery_strategies = [RecoveryStrategy.RETRY, RecoveryStrategy.CIRCUIT_BREAKER, RecoveryStrategy.FALLBACK]
        
        self.recovery_strategies[operation_name] = recovery_strategies
        
        # Initialize circuit breaker if needed
        if RecoveryStrategy.CIRCUIT_BREAKER in recovery_strategies:
            self.circuit_breakers[operation_name] = CircuitBreaker()
        
        # Initialize operation statistics
        self.operation_stats[operation_name] = {
            "total_executions": 0,
            "successful_executions": 0,
            "error_count": 0,
            "recovery_count": 0,
            "average_execution_time": 0.0
        }
        
        print(f"Registered operation: {operation_name} with strategies: {[s.value for s in recovery_strategies]}")
    
    def register_fallback(self, operation_name: str, fallback_func: Callable) -> None:
        """Register a fallback function for an operation"""
        self.fallback_operations[operation_name] = fallback_func
        print(f"Registered fallback for: {operation_name}")
    
    async def execute(self, operation_name: str, *args, **kwargs) -> RecoveryResult:
        """
        Execute operation with comprehensive error recovery
        
        Args:
            operation_name: Name of the operation to execute
            *args, **kwargs: Arguments for the operation
            
        Returns:
            RecoveryResult with operation outcome and recovery details
        """
        if operation_name not in self.operations:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.FALLBACK,
                final_result=None,
                recovery_time=0.0,
                error_details=f"Operation {operation_name} not registered"
            )
        
        print(f"\nEXECUTING: {operation_name}")
        print("-" * 40)
        
        start_time = time.time()
        operation_func = self.operations[operation_name]
        strategies = self.recovery_strategies[operation_name]
        
        # Update statistics
        self.operation_stats[operation_name]["total_executions"] += 1
        
        # Try each recovery strategy in sequence
        last_error = None
        
        for strategy in strategies:
            try:
                print(f"Attempting with strategy: {strategy.value}")
                
                if strategy == RecoveryStrategy.RETRY:
                    result = await self._execute_with_retry(operation_name, operation_func, args, kwargs)
                elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    result = await self._execute_with_circuit_breaker(operation_name, operation_func, args, kwargs)
                elif strategy == RecoveryStrategy.FALLBACK:
                    result = await self._execute_with_fallback(operation_name, args, kwargs)
                elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                    result = await self._execute_with_degradation(operation_name, operation_func, args, kwargs)
                else:
                    # Direct execution for other strategies
                    result = await operation_func(*args, **kwargs)
                
                # Success!
                execution_time = time.time() - start_time
                self._update_success_stats(operation_name, execution_time)
                
                print(f"✓ SUCCESS with {strategy.value}")
                
                return RecoveryResult(
                    success=True,
                    strategy_used=strategy,
                    final_result=result,
                    recovery_time=execution_time,
                    fallback_used=(strategy == RecoveryStrategy.FALLBACK)
                )
                
            except Exception as e:
                last_error = e
                error_event = self._record_error(operation_name, e, strategy)
                print(f"✗ FAILED with {strategy.value}: {str(e)}")
                
                # Continue to next strategy
                continue
        
        # All strategies failed
        execution_time = time.time() - start_time
        self._update_failure_stats(operation_name)
        
        print("✗ ALL RECOVERY STRATEGIES FAILED")
        
        return RecoveryResult(
            success=False,
            strategy_used=strategies[-1] if strategies else RecoveryStrategy.FALLBACK,
            final_result=None,
            recovery_time=execution_time,
            error_details=str(last_error)
        )
    
    async def _execute_with_retry(self, operation_name: str, operation_func: Callable,
                                args: tuple, kwargs: dict) -> Any:
        """Execute operation with exponential backoff retry"""
        config = self.strategy_config[RecoveryStrategy.RETRY]
        max_retries = config["max_retries"]
        backoff_multiplier = config["backoff_multiplier"]
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"  Retry attempt {attempt}/{max_retries}")
                
                result = await operation_func(*args, **kwargs)
                
                if attempt > 0:
                    print(f"  ✓ Succeeded on retry {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if error is retryable
                error_type = self._classify_error(e)
                if error_type in [ErrorType.FATAL, ErrorType.VALIDATION]:
                    print(f"  Non-retryable error: {error_type.value}")
                    raise e
                
                if attempt < max_retries:
                    # Calculate backoff time with jitter
                    backoff_time = self.base_backoff * (backoff_multiplier ** attempt)
                    jitter = random.uniform(0.5, 1.5)
                    wait_time = backoff_time * jitter
                    
                    print(f"  Waiting {wait_time:.2f}s before retry")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  Max retries exceeded")
        
        raise last_exception
    
    async def _execute_with_circuit_breaker(self, operation_name: str, operation_func: Callable,
                                          args: tuple, kwargs: dict) -> Any:
        """Execute operation with circuit breaker protection"""
        if operation_name not in self.circuit_breakers:
            return await operation_func(*args, **kwargs)
        
        circuit_breaker = self.circuit_breakers[operation_name]
        
        if not circuit_breaker.can_execute():
            state = circuit_breaker.get_state()
            print(f"  Circuit breaker {state['state']} - operation blocked")
            raise Exception(f"Circuit breaker open for {operation_name}")
        
        try:
            result = await operation_func(*args, **kwargs)
            circuit_breaker.record_success()
            print(f"  Circuit breaker: operation succeeded")
            return result
            
        except Exception as e:
            circuit_breaker.record_failure()
            state = circuit_breaker.get_state()
            print(f"  Circuit breaker: failure recorded (state: {state['state']})")
            raise e
    
    async def _execute_with_fallback(self, operation_name: str, args: tuple, kwargs: dict) -> Any:
        """Execute fallback operation"""
        if operation_name not in self.fallback_operations:
            raise Exception(f"No fallback available for {operation_name}")
        
        fallback_func = self.fallback_operations[operation_name]
        print(f"  Executing fallback operation")
        
        result = await fallback_func(*args, **kwargs)
        print(f"  ✓ Fallback succeeded")
        return result
    
    async def _execute_with_degradation(self, operation_name: str, operation_func: Callable,
                                      args: tuple, kwargs: dict) -> Any:
        """Execute with graceful degradation"""
        try:
            # Try normal operation with timeout
            config = self.strategy_config[RecoveryStrategy.GRACEFUL_DEGRADATION]
            timeout = config["timeout"]
            
            result = await asyncio.wait_for(operation_func(*args, **kwargs), timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            print(f"  Operation timed out, providing degraded service")
            # Return degraded result
            return self._generate_degraded_result(operation_name, args, kwargs)
        except Exception as e:
            print(f"  Operation failed, providing degraded service")
            return self._generate_degraded_result(operation_name, args, kwargs)
    
    def _generate_degraded_result(self, operation_name: str, args: tuple, kwargs: dict) -> Any:
        """Generate a degraded but functional result"""
        # This would be customized per operation type
        return {
            "status": "degraded",
            "message": f"Service temporarily degraded for {operation_name}",
            "data": None,
            "degradation_reason": "error_recovery"
        }
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify error type for appropriate recovery strategy"""
        error_str = str(error).lower()
        
        if any(keyword in error_str for keyword in ["timeout", "connection", "network"]):
            return ErrorType.TRANSIENT
        elif any(keyword in error_str for keyword in ["validation", "invalid", "bad request"]):
            return ErrorType.VALIDATION
        elif any(keyword in error_str for keyword in ["resource", "memory", "disk"]):
            return ErrorType.RESOURCE
        elif any(keyword in error_str for keyword in ["permission", "unauthorized", "forbidden"]):
            return ErrorType.FATAL
        else:
            return ErrorType.PERSISTENT
    
    def _record_error(self, operation_name: str, error: Exception, 
                     recovery_strategy: RecoveryStrategy) -> ErrorEvent:
        """Record error event for analysis and learning"""
        error_event = ErrorEvent(
            error_id=f"error_{len(self.error_history) + 1}_{int(time.time())}",
            error_type=self._classify_error(error),
            error_message=str(error),
            operation_name=operation_name,
            recovery_attempted=recovery_strategy
        )
        
        self.error_history.append(error_event)
        return error_event
    
    def _update_success_stats(self, operation_name: str, execution_time: float) -> None:
        """Update statistics for successful operation"""
        stats = self.operation_stats[operation_name]
        stats["successful_executions"] += 1
        
        # Update average execution time
        current_avg = stats["average_execution_time"]
        total_successes = stats["successful_executions"]
        stats["average_execution_time"] = (current_avg * (total_successes - 1) + execution_time) / total_successes
    
    def _update_failure_stats(self, operation_name: str) -> None:
        """Update statistics for failed operation"""
        stats = self.operation_stats[operation_name]
        stats["error_count"] += 1
    
    def get_error_recovery_status(self) -> Dict[str, Any]:
        """Get comprehensive error recovery status and metrics"""
        
        # Calculate overall success rates
        operation_success_rates = {}
        for op_name, stats in self.operation_stats.items():
            total = stats["total_executions"]
            successful = stats["successful_executions"]
            operation_success_rates[op_name] = successful / total if total > 0 else 0.0
        
        # Analyze error patterns
        error_patterns = {}
        for error in self.error_history[-20:]:  # Last 20 errors
            error_type = error.error_type.value
            if error_type not in error_patterns:
                error_patterns[error_type] = 0
            error_patterns[error_type] += 1
        
        # Get circuit breaker states
        circuit_breaker_states = {}
        for op_name, cb in self.circuit_breakers.items():
            circuit_breaker_states[op_name] = cb.get_state()
        
        return {
            "agent_id": self.agent_id,
            "registered_operations": len(self.operations),
            "fallback_operations": len(self.fallback_operations),
            "total_errors": len(self.error_history),
            "recent_error_patterns": error_patterns,
            "operation_success_rates": operation_success_rates,
            "operation_statistics": self.operation_stats.copy(),
            "circuit_breaker_states": circuit_breaker_states,
            "recovery_strategies_available": list(set(
                strategy.value for strategies in self.recovery_strategies.values() 
                for strategy in strategies
            ))
        }

# EXAMPLE OPERATIONS WITH DIFFERENT FAILURE MODES
# ===============================================

async def reliable_database_operation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulates a database operation that occasionally fails"""
    await asyncio.sleep(0.1)
    
    # Simulate different failure scenarios
    failure_chance = random.random()
    
    if failure_chance < 0.1:  # 10% transient network errors
        raise Exception("Connection timeout to database")
    elif failure_chance < 0.15:  # 5% resource errors
        raise Exception("Database connection pool exhausted")
    elif failure_chance < 0.18:  # 3% validation errors
        raise Exception("Invalid data format for database")
    
    # Success case
    return {
        "status": "success",
        "data_id": f"db_{int(time.time())}",
        "records_affected": 1
    }

async def external_api_call(api_endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Simulates external API call that can have various failures"""
    await asyncio.sleep(0.2)
    
    failure_chance = random.random()
    
    if failure_chance < 0.2:  # 20% external service errors
        raise Exception("External service unavailable")
    elif failure_chance < 0.25:  # 5% timeout errors
        raise Exception("Request timeout")
    
    return {
        "status": "success",
        "api_response": f"Response from {api_endpoint}",
        "timestamp": time.time()
    }

async def payment_processing(payment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulates payment processing with occasional failures"""
    await asyncio.sleep(0.3)
    
    amount = payment_data.get("amount", 0)
    
    # Higher failure rate for large amounts (risk management)
    failure_chance = 0.1 + (amount / 10000) * 0.2
    
    if random.random() < failure_chance:
        raise Exception("Payment gateway error")
    
    return {
        "status": "success", 
        "transaction_id": f"txn_{int(time.time())}",
        "amount_processed": amount
    }

# FALLBACK OPERATIONS
# ==================

async def database_fallback(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback: Cache or queue operation for later"""
    return {
        "status": "queued",
        "message": "Operation queued for retry when database is available",
        "queue_id": f"queue_{int(time.time())}"
    }

async def api_fallback(api_endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback: Return cached data or default response"""
    return {
        "status": "cached",
        "message": "Returning cached data due to service unavailability",
        "cached_response": "Default cached data"
    }

async def payment_fallback(payment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback: Mark payment as pending for manual processing"""
    return {
        "status": "pending",
        "message": "Payment marked for manual processing",
        "manual_review_id": f"manual_{int(time.time())}"
    }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_database_resilience():
    """Demo: Database operations with error recovery"""
    print("\nDEMO 1: DATABASE OPERATIONS WITH ERROR RECOVERY")
    print("=" * 60)
    
    # Create error recovery agent
    agent = ErrorRecoveryAgent("database_service", max_retries=2)
    
    # Register database operation with recovery strategies
    agent.register_operation("save_data", reliable_database_operation, 
                           [RecoveryStrategy.RETRY, RecoveryStrategy.CIRCUIT_BREAKER, RecoveryStrategy.FALLBACK])
    agent.register_fallback("save_data", database_fallback)
    
    # Simulate multiple database operations
    test_data = [
        {"user_id": "user1", "data": "important_data_1"},
        {"user_id": "user2", "data": "important_data_2"},
        {"user_id": "user3", "data": "important_data_3"},
        {"user_id": "user4", "data": "important_data_4"},
        {"user_id": "user5", "data": "important_data_5"}
    ]
    
    results = []
    for i, data in enumerate(test_data):
        print(f"\n--- Database Operation {i+1} ---")
        result = await agent.execute("save_data", data)
        results.append(result)
        
        print(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
        if result.fallback_used:
            print("Note: Fallback operation used")
    
    # Show statistics
    status = agent.get_error_recovery_status()
    success_rate = status["operation_success_rates"].get("save_data", 0.0)
    print(f"\nDatabase Operations Summary:")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Total errors encountered: {status['total_errors']}")

async def demo_payment_system_resilience():
    """Demo: Payment system with comprehensive error recovery"""
    print("\nDEMO 2: PAYMENT SYSTEM WITH COMPREHENSIVE ERROR RECOVERY")
    print("=" * 60)
    
    # Create payment processing agent
    agent = ErrorRecoveryAgent("payment_system", max_retries=3)
    
    # Register payment operation with all recovery strategies
    agent.register_operation("process_payment", payment_processing,
                           [RecoveryStrategy.RETRY, RecoveryStrategy.CIRCUIT_BREAKER, 
                            RecoveryStrategy.GRACEFUL_DEGRADATION, RecoveryStrategy.FALLBACK])
    agent.register_fallback("process_payment", payment_fallback)
    
    # Register external API operation
    agent.register_operation("verify_payment", external_api_call,
                           [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK])
    agent.register_fallback("verify_payment", api_fallback)
    
    # Simulate various payment scenarios
    payment_scenarios = [
        {"amount": 50.00, "currency": "USD", "method": "credit_card"},
        {"amount": 1500.00, "currency": "USD", "method": "bank_transfer"},
        {"amount": 25.00, "currency": "EUR", "method": "paypal"},
        {"amount": 5000.00, "currency": "USD", "method": "wire_transfer"},
        {"amount": 100.00, "currency": "GBP", "method": "debit_card"}
    ]
    
    successful_payments = 0
    total_payment_value = 0
    
    for i, payment in enumerate(payment_scenarios):
        print(f"\n--- Payment {i+1}: ${payment['amount']:.2f} via {payment['method']} ---")
        
        # Process payment
        result = await agent.execute("process_payment", payment)
        
        if result.success and not result.fallback_used:
            successful_payments += 1
            total_payment_value += payment["amount"]
            print(f"✓ Payment processed successfully")
        elif result.success and result.fallback_used:
            print(f"⚠ Payment queued for manual processing")
        else:
            print(f"✗ Payment failed completely")
        
        # Verify payment (simulate additional API call)
        if result.success:
            verify_result = await agent.execute("verify_payment", "payment_gateway", 
                                              {"transaction_id": "test_txn"})
            if verify_result.success:
                print(f"✓ Payment verification completed")
    
    # Final summary
    print(f"\nPayment System Summary:")
    print(f"Successful payments: {successful_payments}/{len(payment_scenarios)}")
    print(f"Total value processed: ${total_payment_value:.2f}")
    
    # Show detailed error recovery status
    status = agent.get_error_recovery_status()
    print(f"Total recovery operations: {sum(stats['recovery_count'] for stats in status['operation_statistics'].values())}")
    print(f"Error patterns: {status['recent_error_patterns']}")

async def demo_circuit_breaker_protection():
    """Demo: Circuit breaker protecting against cascading failures"""
    print("\nDEMO 3: CIRCUIT BREAKER PROTECTING AGAINST CASCADING FAILURES")
    print("=" * 60)
    
    # Create agent with fast-failing circuit breaker
    agent = ErrorRecoveryAgent("service_gateway")
    
    # Register operation that will fail frequently to trigger circuit breaker
    async def unreliable_service(data: str) -> str:
        # High failure rate to trigger circuit breaker quickly
        if random.random() < 0.7:  # 70% failure rate
            raise Exception("Service overloaded")
        return f"Processed: {data}"
    
    async def service_fallback(data: str) -> str:
        return f"Cached result for: {data}"
    
    # Configure with low failure threshold for demo
    cb_config = agent.strategy_config[RecoveryStrategy.CIRCUIT_BREAKER]
    cb_config["failure_threshold"] = 3  # Open after 3 failures
    cb_config["timeout"] = 10.0  # 10 second timeout
    
    agent.register_operation("call_service", unreliable_service,
                           [RecoveryStrategy.CIRCUIT_BREAKER, RecoveryStrategy.FALLBACK])
    agent.register_fallback("call_service", service_fallback)
    
    # Make multiple calls to trigger circuit breaker
    for i in range(10):
        print(f"\n--- Service Call {i+1} ---")
        result = await agent.execute("call_service", f"request_{i+1}")
        
        if result.success:
            print(f"✓ Service call succeeded")
            if result.fallback_used:
                print("  (Used fallback due to circuit breaker)")
        else:
            print(f"✗ Service call failed")
        
        # Show circuit breaker state
        cb_state = agent.circuit_breakers["call_service"].get_state()
        print(f"Circuit breaker state: {cb_state['state']} (failures: {cb_state['failure_count']})")
        
        # Brief delay between calls
        await asyncio.sleep(0.5)

async def main():
    """
    Demonstrate Error Recovery Patterns for resilient system operation
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement retry logic with exponential backoff
    2. How circuit breakers prevent cascading failures
    3. How fallback operations maintain system availability
    4. How to classify errors for appropriate recovery strategies
    5. How error recovery patterns improve system reliability
    
    REAL WORLD APPLICATIONS:
    =======================
    - E-commerce platforms handling payment failures
    - Microservices with network communication errors
    - Database systems with connection pool exhaustion
    - External API integrations with service outages
    - Cloud applications dealing with infrastructure failures
    - Financial systems requiring transaction consistency
    """
    
    print("ERROR RECOVERY PATTERNS DEMONSTRATION")
    print("This shows how to build resilient systems that gracefully handle failures!")
    
    await demo_database_resilience()
    await demo_payment_system_resilience()
    await demo_circuit_breaker_protection()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Retry patterns handle transient failures effectively")
    print("✓ Circuit breakers prevent system overload during outages")
    print("✓ Fallback operations maintain service availability")
    print("✓ Error classification enables appropriate recovery strategies")
    print("✓ Comprehensive error recovery improves user experience and system reliability")
    print("\nTRY IT YOURSELF:")
    print("- Implement custom recovery strategies for your specific use cases")
    print("- Add transaction rollback for data consistency")
    print("- Create monitoring and alerting for error patterns")
    print("- Implement async retry queues for delayed processing")

if __name__ == "__main__":
    asyncio.run(main())
