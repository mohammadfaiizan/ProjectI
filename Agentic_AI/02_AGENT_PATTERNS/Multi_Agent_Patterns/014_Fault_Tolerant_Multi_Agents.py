#!/usr/bin/env python3
"""
Fault Tolerant Multi-Agents: Building Resilient Systems That Never Fail
========================================================================

WHAT IS THE PROBLEM?
==================
Individual agents fail, networks go down, servers crash. Without fault tolerance, the entire system fails when any component fails.

Example: Banking System Nightmare
BAD APPROACH (Single Point of Failure):
- All transactions go through one server
- Server crashes during peak hours
- Entire banking system goes offline
- Customers can't access money, make payments
- Bank loses millions in downtime costs
- Reputation severely damaged

REAL WORLD EXAMPLE:
=================
How does Netflix ensure you can always watch movies?

NETFLIX FAULT TOLERANCE:
1. Multiple copies of every movie on different servers
2. If one server fails, traffic automatically switches to backup
3. Data centers in multiple geographic locations
4. Chaos engineering - intentionally break things to test resilience
5. Circuit breakers stop cascading failures
6. Graceful degradation - system works even with reduced functionality

FAULT TOLERANCE PRINCIPLES:
- Redundancy: Multiple copies of critical components
- Failover: Automatic switching to backup systems
- Recovery: Quick restoration of failed components
- Isolation: Failures don't spread to other parts
- Monitoring: Early detection of problems

THE ALGORITHM:
=============
1. REPLICATE: Create multiple copies of critical agents
2. MONITOR: Continuously check agent health and availability
3. DETECT: Quickly identify when agents fail
4. ISOLATE: Prevent failures from spreading
5. FAILOVER: Automatically switch to healthy agents
6. RECOVER: Restore failed agents when possible
7. LEARN: Improve system based on failure patterns

WHY IS THIS ESSENTIAL?
====================
- Ensures system availability despite component failures
- Prevents cascading failures that bring down everything
- Maintains service quality during partial failures
- Reduces downtime and service interruptions
- Builds user trust through reliable service
- Enables systems to self-heal automatically
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid

class AgentState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    OFFLINE = "offline"

class FailureType(Enum):
    CRASH = "crash"
    NETWORK = "network"
    OVERLOAD = "overload"
    CORRUPTION = "corruption"
    TIMEOUT = "timeout"

class RecoveryStrategy(Enum):
    RESTART = "restart"
    REPLICATE = "replicate"
    MIGRATE = "migrate"
    DEGRADE = "degrade"

@dataclass
class HealthCheck:
    """Health check result for an agent"""
    agent_id: str
    timestamp: float
    is_healthy: bool
    response_time: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FailureEvent:
    """Record of a failure event"""
    agent_id: str
    failure_type: FailureType
    timestamp: float
    description: str
    affected_operations: List[str] = field(default_factory=list)
    recovery_time: Optional[float] = None

@dataclass
class BackupAgent:
    """Backup agent configuration"""
    primary_agent_id: str
    backup_agent_id: str
    sync_frequency: float = 60.0  # seconds
    last_sync: float = 0.0
    is_active: bool = False

class FaultTolerantAgent:
    """Agent with built-in fault tolerance capabilities"""
    
    def __init__(self, agent_id: str, agent_type: str, 
                 failure_probability: float = 0.01):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.failure_probability = failure_probability
        
        # Agent state
        self.state = AgentState.HEALTHY
        self.last_heartbeat = time.time()
        self.operation_count = 0
        self.error_count = 0
        
        # Fault tolerance
        self.backup_agents: List[str] = []
        self.synchronized_state: Dict[str, Any] = {}
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        # Performance tracking
        self.response_times: List[float] = []
        self.failure_history: List[FailureEvent] = []
        
        # Circuit breaker pattern
        self.circuit_breaker_threshold = 5  # failures
        self.circuit_breaker_timeout = 30.0  # seconds
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0.0
        self.circuit_breaker_open = False
    
    async def execute_operation(self, operation: str, data: Any = None) -> Dict[str, Any]:
        """Execute operation with fault tolerance"""
        
        start_time = time.time()
        
        try:
            # Check if agent is healthy enough to operate
            if not self.can_execute_operation():
                raise Exception("Agent not available for operations")
            
            # Simulate operation execution
            result = await self.perform_operation(operation, data)
            
            # Record successful operation
            execution_time = time.time() - start_time
            self.operation_count += 1
            self.response_times.append(execution_time)
            self.last_heartbeat = time.time()
            
            # Reset circuit breaker on success
            if self.circuit_breaker_failures > 0:
                self.circuit_breaker_failures = max(0, self.circuit_breaker_failures - 1)
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            # Handle operation failure
            execution_time = time.time() - start_time
            await self.handle_operation_failure(operation, str(e), execution_time)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "agent_id": self.agent_id
            }
    
    async def perform_operation(self, operation: str, data: Any) -> Any:
        """Perform the actual operation (with potential failures)"""
        
        # Simulate operation processing time
        processing_time = random.uniform(0.1, 0.5)
        await asyncio.sleep(processing_time)
        
        # Simulate random failures
        if random.random() < self.failure_probability:
            failure_types = [
                FailureType.CRASH,
                FailureType.NETWORK,
                FailureType.OVERLOAD,
                FailureType.TIMEOUT
            ]
            failure_type = random.choice(failure_types)
            
            await self.simulate_failure(failure_type)
        
        # Simulate successful operation
        return {
            "operation": operation,
            "data_processed": data,
            "timestamp": time.time(),
            "agent_type": self.agent_type
        }
    
    async def simulate_failure(self, failure_type: FailureType) -> None:
        """Simulate different types of failures"""
        
        if failure_type == FailureType.CRASH:
            self.state = AgentState.FAILED
            raise Exception("Agent crashed during operation")
        
        elif failure_type == FailureType.NETWORK:
            raise Exception("Network connection lost")
        
        elif failure_type == FailureType.OVERLOAD:
            self.state = AgentState.DEGRADED
            raise Exception("Agent overloaded - too many requests")
        
        elif failure_type == FailureType.TIMEOUT:
            await asyncio.sleep(2.0)  # Simulate timeout
            raise Exception("Operation timed out")
        
        elif failure_type == FailureType.CORRUPTION:
            raise Exception("Data corruption detected")
    
    async def handle_operation_failure(self, operation: str, error: str, execution_time: float) -> None:
        """Handle operation failure"""
        
        self.error_count += 1
        self.response_times.append(execution_time)
        
        # Record failure event
        failure_event = FailureEvent(
            agent_id=self.agent_id,
            failure_type=self.classify_failure(error),
            timestamp=time.time(),
            description=error,
            affected_operations=[operation]
        )
        self.failure_history.append(failure_event)
        
        # Update circuit breaker
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            print(f"  Circuit breaker OPEN for {self.agent_id}")
        
        print(f"  {self.agent_id} operation failed: {error}")
    
    def classify_failure(self, error: str) -> FailureType:
        """Classify failure type from error message"""
        error_lower = error.lower()
        
        if "crash" in error_lower:
            return FailureType.CRASH
        elif "network" in error_lower:
            return FailureType.NETWORK
        elif "overload" in error_lower:
            return FailureType.OVERLOAD
        elif "timeout" in error_lower:
            return FailureType.TIMEOUT
        elif "corruption" in error_lower:
            return FailureType.CORRUPTION
        else:
            return FailureType.CRASH  # Default
    
    def can_execute_operation(self) -> bool:
        """Check if agent can execute operations"""
        
        # Check agent state
        if self.state == AgentState.FAILED or self.state == AgentState.OFFLINE:
            return False
        
        # Check circuit breaker
        if self.circuit_breaker_open:
            # Check if timeout period has passed
            if time.time() - self.circuit_breaker_last_failure > self.circuit_breaker_timeout:
                self.circuit_breaker_open = False
                self.circuit_breaker_failures = 0
                print(f"  Circuit breaker CLOSED for {self.agent_id}")
                return True
            else:
                return False
        
        return True
    
    async def perform_health_check(self) -> HealthCheck:
        """Perform health check on agent"""
        
        start_time = time.time()
        
        try:
            # Simulate health check
            await asyncio.sleep(0.05)  # Health check time
            
            # Calculate health metrics
            error_rate = self.error_count / max(1, self.operation_count)
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            
            # Determine health status
            is_healthy = (
                self.state != AgentState.FAILED and
                self.state != AgentState.OFFLINE and
                error_rate < 0.1 and  # Less than 10% error rate
                avg_response_time < 2.0 and  # Response time under 2 seconds
                not self.circuit_breaker_open
            )
            
            response_time = time.time() - start_time
            
            return HealthCheck(
                agent_id=self.agent_id,
                timestamp=time.time(),
                is_healthy=is_healthy,
                response_time=response_time,
                metrics={
                    "state": self.state.value,
                    "error_rate": error_rate,
                    "avg_response_time": avg_response_time,
                    "operation_count": self.operation_count,
                    "circuit_breaker_open": self.circuit_breaker_open
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            
            return HealthCheck(
                agent_id=self.agent_id,
                timestamp=time.time(),
                is_healthy=False,
                response_time=response_time,
                error_message=str(e)
            )
    
    async def recover(self, recovery_strategy: RecoveryStrategy) -> bool:
        """Attempt to recover from failure"""
        
        if self.recovery_attempts >= self.max_recovery_attempts:
            print(f"  {self.agent_id} exceeded max recovery attempts")
            return False
        
        self.recovery_attempts += 1
        print(f"  Attempting recovery for {self.agent_id} (attempt {self.recovery_attempts})")
        
        try:
            if recovery_strategy == RecoveryStrategy.RESTART:
                await self.restart_agent()
            elif recovery_strategy == RecoveryStrategy.REPLICATE:
                await self.create_replica()
            elif recovery_strategy == RecoveryStrategy.MIGRATE:
                await self.migrate_state()
            elif recovery_strategy == RecoveryStrategy.DEGRADE:
                await self.degrade_gracefully()
            
            self.state = AgentState.HEALTHY
            self.circuit_breaker_open = False
            self.circuit_breaker_failures = 0
            print(f"  {self.agent_id} recovered successfully")
            return True
            
        except Exception as e:
            print(f"  Recovery failed for {self.agent_id}: {e}")
            return False
    
    async def restart_agent(self) -> None:
        """Restart the agent"""
        self.state = AgentState.RECOVERING
        await asyncio.sleep(1.0)  # Restart time
        
        # Reset error counters
        self.error_count = 0
        self.response_times = []
        self.recovery_attempts = 0
    
    async def create_replica(self) -> None:
        """Create a replica of the agent"""
        self.state = AgentState.RECOVERING
        await asyncio.sleep(0.5)  # Replication time
    
    async def migrate_state(self) -> None:
        """Migrate agent state to new instance"""
        self.state = AgentState.RECOVERING
        await asyncio.sleep(0.8)  # Migration time
    
    async def degrade_gracefully(self) -> None:
        """Degrade to limited functionality"""
        self.state = AgentState.DEGRADED
        await asyncio.sleep(0.2)  # Degradation time
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get failure and recovery summary"""
        
        total_failures = len(self.failure_history)
        failure_types = {}
        for failure in self.failure_history:
            failure_type = failure.failure_type.value
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        
        return {
            "agent_id": self.agent_id,
            "current_state": self.state.value,
            "total_operations": self.operation_count,
            "total_failures": total_failures,
            "error_rate": self.error_count / max(1, self.operation_count),
            "failure_types": failure_types,
            "recovery_attempts": self.recovery_attempts,
            "circuit_breaker_open": self.circuit_breaker_open
        }

class FaultTolerantSystem:
    """
    System providing fault tolerance for multiple agents
    
    EXAMPLE USAGE:
    =============
    # Create fault tolerant system
    system = FaultTolerantSystem("banking_system")
    
    # Add redundant agents
    for i in range(5):
        agent = FaultTolerantAgent(f"server_{i}", "transaction_processor")
        system.add_agent(agent)
    
    # Configure redundancy
    system.setup_redundancy_groups()
    
    # Execute operations with fault tolerance
    result = await system.execute_with_fault_tolerance("process_payment", {"amount": 100})
    """
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.agents: Dict[str, FaultTolerantAgent] = {}
        self.redundancy_groups: Dict[str, List[str]] = {}
        self.backup_configurations: List[BackupAgent] = []
        
        # Health monitoring
        self.health_check_interval = 5.0  # seconds
        self.last_health_check = time.time()
        self.health_history: List[HealthCheck] = []
        
        # Failure tracking
        self.system_failures: List[FailureEvent] = []
        self.recovery_statistics: Dict[str, int] = {}
        
        # System metrics
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.failover_events = 0
    
    def add_agent(self, agent: FaultTolerantAgent) -> None:
        """Add agent to fault tolerant system"""
        self.agents[agent.agent_id] = agent
        print(f"Added fault tolerant agent: {agent.agent_id}")
    
    def setup_redundancy_groups(self, group_size: int = 3) -> None:
        """Setup redundancy groups for fault tolerance"""
        
        agent_list = list(self.agents.keys())
        
        # Create groups of agents for redundancy
        for i in range(0, len(agent_list), group_size):
            group_id = f"group_{i // group_size}"
            self.redundancy_groups[group_id] = agent_list[i:i + group_size]
            
            print(f"Created redundancy group {group_id}: {self.redundancy_groups[group_id]}")
        
        # Setup backup relationships
        for group_agents in self.redundancy_groups.values():
            if len(group_agents) >= 2:
                primary = group_agents[0]
                for backup in group_agents[1:]:
                    backup_config = BackupAgent(
                        primary_agent_id=primary,
                        backup_agent_id=backup
                    )
                    self.backup_configurations.append(backup_config)
    
    async def execute_with_fault_tolerance(self, operation: str, data: Any = None) -> Dict[str, Any]:
        """Execute operation with automatic fault tolerance"""
        
        self.total_operations += 1
        
        # Find healthy agents
        healthy_agents = await self.get_healthy_agents()
        
        if not healthy_agents:
            self.failed_operations += 1
            return {
                "success": False,
                "error": "No healthy agents available",
                "failover_attempted": True
            }
        
        # Try primary agent first
        primary_agent = healthy_agents[0]
        
        try:
            result = await primary_agent.execute_operation(operation, data)
            
            if result["success"]:
                self.successful_operations += 1
                return result
            else:
                # Primary failed, try failover
                return await self.attempt_failover(operation, data, healthy_agents[1:])
                
        except Exception as e:
            # Primary agent exception, try failover
            return await self.attempt_failover(operation, data, healthy_agents[1:])
    
    async def attempt_failover(self, operation: str, data: Any, 
                             backup_agents: List[FaultTolerantAgent]) -> Dict[str, Any]:
        """Attempt failover to backup agents"""
        
        self.failover_events += 1
        print(f"  Attempting failover for operation: {operation}")
        
        for backup_agent in backup_agents:
            try:
                result = await backup_agent.execute_operation(operation, data)
                
                if result["success"]:
                    self.successful_operations += 1
                    result["failover_successful"] = True
                    result["backup_agent"] = backup_agent.agent_id
                    print(f"  Failover successful to {backup_agent.agent_id}")
                    return result
                    
            except Exception as e:
                print(f"  Backup agent {backup_agent.agent_id} also failed: {e}")
                continue
        
        # All agents failed
        self.failed_operations += 1
        return {
            "success": False,
            "error": "All agents failed, operation could not be completed",
            "failover_attempted": True,
            "failover_successful": False
        }
    
    async def get_healthy_agents(self) -> List[FaultTolerantAgent]:
        """Get list of currently healthy agents"""
        
        healthy_agents = []
        
        for agent in self.agents.values():
            health_check = await agent.perform_health_check()
            
            if health_check.is_healthy:
                healthy_agents.append(agent)
        
        return healthy_agents
    
    async def perform_system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        
        print(f"\nPerforming system health check...")
        
        health_results = {}
        healthy_count = 0
        degraded_count = 0
        failed_count = 0
        
        for agent_id, agent in self.agents.items():
            health_check = await agent.perform_health_check()
            health_results[agent_id] = health_check
            self.health_history.append(health_check)
            
            if health_check.is_healthy:
                healthy_count += 1
            elif agent.state == AgentState.DEGRADED:
                degraded_count += 1
            else:
                failed_count += 1
        
        # Trigger recovery for failed agents
        await self.recover_failed_agents()
        
        system_health = {
            "system_id": self.system_id,
            "timestamp": time.time(),
            "total_agents": len(self.agents),
            "healthy_agents": healthy_count,
            "degraded_agents": degraded_count,
            "failed_agents": failed_count,
            "system_availability": healthy_count / len(self.agents) if self.agents else 0,
            "agent_health": {agent_id: check.is_healthy for agent_id, check in health_results.items()}
        }
        
        print(f"System health: {healthy_count}/{len(self.agents)} agents healthy")
        
        return system_health
    
    async def recover_failed_agents(self) -> None:
        """Attempt to recover failed agents"""
        
        failed_agents = [agent for agent in self.agents.values() 
                        if agent.state == AgentState.FAILED]
        
        recovery_tasks = []
        for agent in failed_agents:
            # Choose recovery strategy based on failure type
            strategy = self.choose_recovery_strategy(agent)
            recovery_task = agent.recover(strategy)
            recovery_tasks.append(recovery_task)
        
        if recovery_tasks:
            print(f"Attempting recovery for {len(recovery_tasks)} failed agents...")
            recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
            
            successful_recoveries = sum(1 for result in recovery_results if result is True)
            print(f"Successfully recovered {successful_recoveries}/{len(recovery_tasks)} agents")
    
    def choose_recovery_strategy(self, agent: FaultTolerantAgent) -> RecoveryStrategy:
        """Choose appropriate recovery strategy for agent"""
        
        if not agent.failure_history:
            return RecoveryStrategy.RESTART
        
        # Analyze recent failures
        recent_failures = [f for f in agent.failure_history 
                          if time.time() - f.timestamp < 300]  # Last 5 minutes
        
        if len(recent_failures) > 3:
            return RecoveryStrategy.REPLICATE  # Too many failures, create new instance
        
        latest_failure = agent.failure_history[-1]
        
        if latest_failure.failure_type == FailureType.CRASH:
            return RecoveryStrategy.RESTART
        elif latest_failure.failure_type == FailureType.CORRUPTION:
            return RecoveryStrategy.REPLICATE
        elif latest_failure.failure_type == FailureType.OVERLOAD:
            return RecoveryStrategy.DEGRADE
        else:
            return RecoveryStrategy.RESTART
    
    async def run_fault_tolerance_test(self, duration: float = 30.0, 
                                     operation_rate: float = 5.0) -> Dict[str, Any]:
        """Run comprehensive fault tolerance test"""
        
        print(f"\nRUNNING FAULT TOLERANCE TEST")
        print(f"Duration: {duration}s, Operation rate: {operation_rate}/s")
        print("-" * 50)
        
        start_time = time.time()
        test_results = []
        
        # Setup background health monitoring
        health_monitoring_task = asyncio.create_task(
            self.continuous_health_monitoring(duration)
        )
        
        # Generate continuous load
        while time.time() - start_time < duration:
            operation_id = f"op_{len(test_results)}"
            
            # Execute operation with fault tolerance
            result = await self.execute_with_fault_tolerance(
                "test_operation", 
                {"operation_id": operation_id}
            )
            
            test_results.append(result)
            
            # Control operation rate
            await asyncio.sleep(1.0 / operation_rate)
        
        # Stop health monitoring
        health_monitoring_task.cancel()
        
        # Generate test summary
        test_duration = time.time() - start_time
        
        successful_ops = len([r for r in test_results if r.get("success", False)])
        failed_ops = len([r for r in test_results if not r.get("success", False)])
        failover_ops = len([r for r in test_results if r.get("failover_successful", False)])
        
        fault_tolerance_summary = {
            "test_duration": test_duration,
            "total_operations": len(test_results),
            "successful_operations": successful_ops,
            "failed_operations": failed_ops,
            "success_rate": successful_ops / len(test_results) if test_results else 0,
            "failover_operations": failover_ops,
            "failover_success_rate": failover_ops / failed_ops if failed_ops > 0 else 0,
            "system_resilience": successful_ops / len(test_results) if test_results else 0,
            "agent_failure_summary": {agent_id: agent.get_failure_summary() 
                                    for agent_id, agent in self.agents.items()}
        }
        
        print(f"\nFault Tolerance Test Results:")
        print(f"- Operations completed: {len(test_results)}")
        print(f"- Success rate: {fault_tolerance_summary['success_rate']:.1%}")
        print(f"- Failover success rate: {fault_tolerance_summary['failover_success_rate']:.1%}")
        print(f"- System resilience: {fault_tolerance_summary['system_resilience']:.1%}")
        
        return fault_tolerance_summary
    
    async def continuous_health_monitoring(self, duration: float) -> None:
        """Continuous health monitoring during test"""
        
        try:
            end_time = time.time() + duration
            
            while time.time() < end_time:
                await self.perform_system_health_check()
                await asyncio.sleep(self.health_check_interval)
                
        except asyncio.CancelledError:
            pass
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        return {
            "system_id": self.system_id,
            "total_agents": len(self.agents),
            "redundancy_groups": len(self.redundancy_groups),
            "backup_configurations": len(self.backup_configurations),
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "system_success_rate": self.successful_operations / max(1, self.total_operations),
            "failover_events": self.failover_events,
            "agent_statistics": {agent_id: agent.get_failure_summary() 
                               for agent_id, agent in self.agents.items()}
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_banking_system_fault_tolerance():
    """Demo: Banking system with fault tolerance"""
    print("\nDEMO 1: BANKING SYSTEM FAULT TOLERANCE")
    print("=" * 50)
    
    # Create fault tolerant banking system
    banking_system = FaultTolerantSystem("secure_banking")
    
    # Add transaction processing servers
    servers = [
        ("primary_server", 0.02),      # 2% failure rate
        ("backup_server_1", 0.03),     # 3% failure rate
        ("backup_server_2", 0.01),     # 1% failure rate
        ("dr_server", 0.04),           # 4% failure rate (disaster recovery)
        ("cloud_server", 0.02)         # 2% failure rate
    ]
    
    for server_id, failure_rate in servers:
        agent = FaultTolerantAgent(server_id, "transaction_processor", failure_rate)
        banking_system.add_agent(agent)
    
    # Setup redundancy
    banking_system.setup_redundancy_groups(group_size=2)
    
    # Run fault tolerance test
    result = await banking_system.run_fault_tolerance_test(
        duration=20.0,
        operation_rate=8.0  # 8 transactions per second
    )
    
    print(f"\nBanking System Results:")
    print(f"- System resilience: {result['system_resilience']:.1%}")
    print(f"- Failover success: {result['failover_success_rate']:.1%}")

async def demo_microservice_resilience():
    """Demo: Microservice architecture with fault tolerance"""
    print("\nDEMO 2: MICROSERVICE RESILIENCE")
    print("=" * 50)
    
    # Create microservice system
    microservice_system = FaultTolerantSystem("ecommerce_platform")
    
    # Add different microservices
    services = [
        ("user_service_1", 0.03),
        ("user_service_2", 0.02),
        ("payment_service_1", 0.01),  # Critical service - lower failure rate
        ("payment_service_2", 0.01),
        ("inventory_service_1", 0.04),
        ("inventory_service_2", 0.03),
        ("notification_service", 0.05)  # Non-critical - higher failure rate
    ]
    
    for service_id, failure_rate in services:
        agent = FaultTolerantAgent(service_id, "microservice", failure_rate)
        microservice_system.add_agent(agent)
    
    # Setup service redundancy
    microservice_system.setup_redundancy_groups(group_size=2)
    
    # Test microservice resilience
    result = await microservice_system.run_fault_tolerance_test(
        duration=15.0,
        operation_rate=12.0  # 12 operations per second
    )
    
    print(f"\nMicroservice Results:")
    print(f"- Platform availability: {result['system_resilience']:.1%}")
    print(f"- Service failover rate: {result['failover_success_rate']:.1%}")

async def demo_distributed_database_cluster():
    """Demo: Distributed database with fault tolerance"""
    print("\nDEMO 3: DISTRIBUTED DATABASE CLUSTER")
    print("=" * 50)
    
    # Create database cluster
    db_cluster = FaultTolerantSystem("distributed_database")
    
    # Add database nodes
    db_nodes = [
        ("master_db", 0.01),      # Master node - very reliable
        ("replica_db_1", 0.02),   # Read replica
        ("replica_db_2", 0.02),   # Read replica
        ("replica_db_3", 0.03),   # Read replica
        ("backup_db", 0.02)       # Backup node
    ]
    
    for node_id, failure_rate in db_nodes:
        agent = FaultTolerantAgent(node_id, "database_node", failure_rate)
        db_cluster.add_agent(agent)
    
    # Setup database redundancy
    db_cluster.setup_redundancy_groups(group_size=3)
    
    # Test database cluster resilience
    result = await db_cluster.run_fault_tolerance_test(
        duration=18.0,
        operation_rate=6.0  # 6 database operations per second
    )
    
    print(f"\nDatabase Cluster Results:")
    print(f"- Data availability: {result['system_resilience']:.1%}")
    print(f"- Replica failover: {result['failover_success_rate']:.1%}")
    
    # Show individual node performance
    stats = db_cluster.get_system_statistics()
    print(f"\nNode Performance:")
    for node_id, node_stats in stats['agent_statistics'].items():
        print(f"  {node_id}: {node_stats['error_rate']:.1%} error rate")

async def main():
    """
    Demonstrate Fault Tolerant Multi-Agents for building resilient systems
    
    WHAT YOU'LL LEARN:
    ================
    1. How to design systems that continue working despite component failures
    2. How to implement automatic failover and recovery mechanisms
    3. How to use redundancy and replication for high availability
    4. How to monitor system health and detect failures early
    5. How fault tolerance enables building robust, production-ready systems
    
    REAL WORLD APPLICATIONS:
    =======================
    - Banking and financial transaction systems
    - E-commerce platforms and payment processing
    - Distributed databases and data storage systems
    - Microservice architectures and API gateways
    - Cloud computing platforms and infrastructure
    - Mission-critical control systems (aerospace, medical)
    """
    
    print("FAULT TOLERANT MULTI-AGENTS DEMONSTRATION")
    print("This shows how to build systems that never fail!")
    
    await demo_banking_system_fault_tolerance()
    await demo_microservice_resilience()
    await demo_distributed_database_cluster()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Redundancy and replication prevent single points of failure")
    print("✓ Automatic failover maintains service availability")
    print("✓ Health monitoring enables proactive failure detection")
    print("✓ Circuit breakers prevent cascading failures")
    print("✓ Fault tolerance is essential for production systems")

if __name__ == "__main__":
    asyncio.run(main())
