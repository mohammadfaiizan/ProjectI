#!/usr/bin/env python3
"""
Agent Load Balancing: Optimal Resource Distribution and Performance
==================================================================

WHAT IS THE PROBLEM?
==================
When work isn't distributed evenly, some agents get overwhelmed while others sit idle. This wastes resources and creates bottlenecks.

Example: Restaurant Without Load Balancing
BAD APPROACH:
- Customers all line up at one cashier
- Other cashiers stand around doing nothing
- Long wait times frustrate customers
- One overworked cashier makes mistakes
- Restaurant looks inefficient and unprofessional

REAL WORLD EXAMPLE:
=================
How does Netflix handle millions of users?

NETFLIX LOAD BALANCING:
1. User requests come to load balancer
2. Load balancer checks server health and current load
3. Routes request to least busy available server
4. Monitors response times continuously
5. Automatically removes failing servers from rotation
6. Scales servers up/down based on demand

LOAD BALANCING STRATEGIES:
- Round Robin: Distribute requests equally
- Least Connections: Send to server with fewest active connections
- Weighted: Distribute based on server capacity
- Geographic: Route to nearest server
- Health-based: Only use healthy servers

THE ALGORITHM:
=============
1. MONITOR: Track agent load, performance, and health
2. ANALYZE: Identify overloaded and underutilized agents
3. DISTRIBUTE: Route new tasks to optimal agents
4. BALANCE: Redistribute existing work if needed
5. SCALE: Add/remove agents based on demand
6. OPTIMIZE: Continuously improve distribution strategy

WHY IS THIS CRITICAL?
===================
- Maximizes system throughput and performance
- Prevents bottlenecks and single points of failure
- Ensures fair resource utilization
- Improves user experience through faster response
- Enables elastic scaling with demand
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    GEOGRAPHIC = "geographic"

class AgentHealth(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"

@dataclass
class LoadMetrics:
    """Agent load and performance metrics"""
    agent_id: str
    cpu_usage: float  # 0.0 to 1.0
    memory_usage: float  # 0.0 to 1.0
    active_connections: int
    requests_per_second: float
    average_response_time: float
    error_rate: float  # 0.0 to 1.0
    health_status: AgentHealth
    last_updated: float = field(default_factory=time.time)

@dataclass
class WorkloadRequest:
    """Request to be load balanced"""
    id: str
    request_type: str
    payload_size: int
    priority: int = 1
    geographic_region: str = "global"
    estimated_duration: float = 1.0
    timestamp: float = field(default_factory=time.time)

class LoadBalancedAgent:
    """Agent that participates in load balancing"""
    
    def __init__(self, agent_id: str, capacity: float = 1.0, 
                 geographic_region: str = "global", weight: float = 1.0):
        self.agent_id = agent_id
        self.capacity = capacity  # Processing capacity multiplier
        self.geographic_region = geographic_region
        self.weight = weight  # Load balancing weight
        
        # Current state
        self.active_connections = 0
        self.max_connections = int(capacity * 100)
        self.current_load = 0.0
        
        # Performance tracking
        self.response_times: List[float] = []
        self.requests_handled = 0
        self.errors_count = 0
        self.total_processing_time = 0.0
        
        # Health monitoring
        self.health_status = AgentHealth.HEALTHY
        self.last_health_check = time.time()
        
        # Load simulation
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
    
    async def handle_request(self, request: WorkloadRequest) -> Dict[str, Any]:
        """Handle an incoming request"""
        
        start_time = time.time()
        self.active_connections += 1
        
        try:
            # Simulate request processing
            processing_time = await self.process_request(request)
            
            # Update metrics
            self.requests_handled += 1
            self.response_times.append(processing_time)
            self.total_processing_time += processing_time
            
            # Keep only recent response times
            if len(self.response_times) > 100:
                self.response_times.pop(0)
            
            print(f"  {self.agent_id} handled {request.id} in {processing_time:.2f}s")
            
            return {
                "request_id": request.id,
                "agent_id": self.agent_id,
                "processing_time": processing_time,
                "success": True
            }
            
        except Exception as e:
            # Handle errors
            self.errors_count += 1
            processing_time = time.time() - start_time
            
            print(f"  {self.agent_id} failed {request.id}: {str(e)}")
            
            return {
                "request_id": request.id,
                "agent_id": self.agent_id,
                "processing_time": processing_time,
                "success": False,
                "error": str(e)
            }
            
        finally:
            self.active_connections -= 1
            self.update_load_metrics()
    
    async def process_request(self, request: WorkloadRequest) -> float:
        """Process the actual request"""
        
        # Base processing time adjusted by capacity
        base_time = request.estimated_duration / self.capacity
        
        # Add load-based delay
        load_factor = 1.0 + (self.current_load * 0.5)
        processing_time = base_time * load_factor
        
        # Add some randomness
        processing_time *= random.uniform(0.8, 1.2)
        
        # Simulate request processing
        await asyncio.sleep(processing_time)
        
        # Simulate occasional failures
        if random.random() < 0.02:  # 2% failure rate
            raise Exception("Simulated processing error")
        
        return processing_time
    
    def update_load_metrics(self) -> None:
        """Update current load metrics"""
        
        # Update CPU usage based on active connections
        self.cpu_usage = min(1.0, self.active_connections / self.max_connections * 1.2)
        
        # Update memory usage (simulated)
        self.memory_usage = min(1.0, self.cpu_usage * 0.8 + random.uniform(0, 0.2))
        
        # Update current load
        self.current_load = (self.cpu_usage + self.memory_usage) / 2.0
        
        # Update health status
        self.update_health_status()
    
    def update_health_status(self) -> None:
        """Update agent health status"""
        
        error_rate = self.get_error_rate()
        avg_response_time = self.get_average_response_time()
        
        if (self.current_load > 0.9 or 
            error_rate > 0.1 or 
            avg_response_time > 5.0):
            self.health_status = AgentHealth.CRITICAL
        elif (self.current_load > 0.7 or 
              error_rate > 0.05 or 
              avg_response_time > 3.0):
            self.health_status = AgentHealth.WARNING
        else:
            self.health_status = AgentHealth.HEALTHY
        
        self.last_health_check = time.time()
    
    def get_load_metrics(self) -> LoadMetrics:
        """Get current load metrics"""
        
        return LoadMetrics(
            agent_id=self.agent_id,
            cpu_usage=self.cpu_usage,
            memory_usage=self.memory_usage,
            active_connections=self.active_connections,
            requests_per_second=self.get_requests_per_second(),
            average_response_time=self.get_average_response_time(),
            error_rate=self.get_error_rate(),
            health_status=self.health_status
        )
    
    def get_requests_per_second(self) -> float:
        """Calculate requests per second"""
        if self.total_processing_time > 0:
            return self.requests_handled / self.total_processing_time
        return 0.0
    
    def get_average_response_time(self) -> float:
        """Calculate average response time"""
        if self.response_times:
            return statistics.mean(self.response_times)
        return 0.0
    
    def get_error_rate(self) -> float:
        """Calculate error rate"""
        total_requests = self.requests_handled + self.errors_count
        if total_requests > 0:
            return self.errors_count / total_requests
        return 0.0
    
    def is_available(self) -> bool:
        """Check if agent is available for new requests"""
        return (self.health_status != AgentHealth.FAILED and 
                self.active_connections < self.max_connections)
    
    def get_load_score(self, strategy: LoadBalancingStrategy) -> float:
        """Get load score for load balancing decision"""
        
        if strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self.active_connections
        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self.get_average_response_time()
        elif strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self.current_load
        else:
            return self.current_load  # Default

class LoadBalancer:
    """
    Load balancer that distributes work across multiple agents
    
    EXAMPLE USAGE:
    =============
    # Create load balancer
    balancer = LoadBalancer("web_servers", LoadBalancingStrategy.LEAST_CONNECTIONS)
    
    # Add server agents
    for i in range(5):
        server = LoadBalancedAgent(f"server_{i}", capacity=random.uniform(0.8, 1.5))
        balancer.add_agent(server)
    
    # Handle incoming requests
    result = await balancer.handle_request_load(1000, duration=30)
    """
    
    def __init__(self, balancer_id: str, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.balancer_id = balancer_id
        self.strategy = strategy
        self.agents: Dict[str, LoadBalancedAgent] = {}
        
        # Load balancing state
        self.round_robin_index = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Performance tracking
        self.request_history: List[Dict[str, Any]] = []
        self.agent_performance: Dict[str, List[float]] = {}
        
        # Health monitoring
        self.health_check_interval = 5.0  # seconds
        self.last_health_check = time.time()
    
    def add_agent(self, agent: LoadBalancedAgent) -> None:
        """Add agent to load balancer"""
        self.agents[agent.agent_id] = agent
        self.agent_performance[agent.agent_id] = []
        print(f"Added agent: {agent.agent_id} (capacity: {agent.capacity:.1f}x)")
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove agent from load balancer"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.agent_performance[agent_id]
            print(f"Removed agent: {agent_id}")
    
    async def route_request(self, request: WorkloadRequest) -> Dict[str, Any]:
        """Route request to optimal agent"""
        
        # Get available agents
        available_agents = [agent for agent in self.agents.values() if agent.is_available()]
        
        if not available_agents:
            return {
                "request_id": request.id,
                "success": False,
                "error": "No available agents"
            }
        
        # Select agent based on strategy
        selected_agent = self.select_agent(available_agents, request)
        
        if not selected_agent:
            return {
                "request_id": request.id,
                "success": False,
                "error": "No suitable agent found"
            }
        
        # Route request to selected agent
        result = await selected_agent.handle_request(request)
        
        # Update statistics
        self.total_requests += 1
        if result.get("success", False):
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Record performance
        self.request_history.append(result)
        if len(self.request_history) > 1000:  # Keep last 1000 requests
            self.request_history.pop(0)
        
        return result
    
    def select_agent(self, available_agents: List[LoadBalancedAgent], 
                    request: WorkloadRequest) -> Optional[LoadBalancedAgent]:
        """Select best agent based on load balancing strategy"""
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self.round_robin_selection(available_agents)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self.weighted_round_robin_selection(available_agents)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(available_agents, key=lambda a: a.active_connections)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return min(available_agents, key=lambda a: a.get_average_response_time())
        
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return min(available_agents, key=lambda a: a.current_load)
        
        elif self.strategy == LoadBalancingStrategy.GEOGRAPHIC:
            return self.geographic_selection(available_agents, request)
        
        else:
            return available_agents[0]  # Fallback
    
    def round_robin_selection(self, available_agents: List[LoadBalancedAgent]) -> LoadBalancedAgent:
        """Simple round-robin selection"""
        agent = available_agents[self.round_robin_index % len(available_agents)]
        self.round_robin_index += 1
        return agent
    
    def weighted_round_robin_selection(self, available_agents: List[LoadBalancedAgent]) -> LoadBalancedAgent:
        """Weighted round-robin based on agent capacity"""
        weights = [agent.weight * agent.capacity for agent in available_agents]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return available_agents[0]
        
        # Select based on weight
        selection_point = random.uniform(0, total_weight)
        current_weight = 0.0
        
        for i, weight in enumerate(weights):
            current_weight += weight
            if current_weight >= selection_point:
                return available_agents[i]
        
        return available_agents[-1]  # Fallback
    
    def geographic_selection(self, available_agents: List[LoadBalancedAgent], 
                           request: WorkloadRequest) -> LoadBalancedAgent:
        """Select agent based on geographic proximity"""
        
        # Try to find agent in same region
        regional_agents = [agent for agent in available_agents 
                          if agent.geographic_region == request.geographic_region]
        
        if regional_agents:
            # Use least connections among regional agents
            return min(regional_agents, key=lambda a: a.active_connections)
        else:
            # Fallback to least connections globally
            return min(available_agents, key=lambda a: a.active_connections)
    
    async def handle_request_load(self, request_count: int, 
                                duration: float = 30.0, 
                                request_rate: float = 10.0) -> Dict[str, Any]:
        """Handle a sustained load of requests"""
        
        print(f"\nHANDLING REQUEST LOAD: {request_count} requests over {duration}s")
        print(f"Strategy: {self.strategy.value}")
        print("-" * 60)
        
        start_time = time.time()
        requests_sent = 0
        results = []
        
        # Calculate request interval
        request_interval = 1.0 / request_rate if request_rate > 0 else 0.1
        
        while (requests_sent < request_count and 
               time.time() - start_time < duration):
            
            # Create request
            request = WorkloadRequest(
                id=f"req_{requests_sent}",
                request_type="web_request",
                payload_size=random.randint(1024, 10240),  # 1-10KB
                priority=random.randint(1, 3),
                geographic_region=random.choice(["us_east", "us_west", "europe", "asia"]),
                estimated_duration=random.uniform(0.1, 0.5)
            )
            
            # Route request (don't wait for completion)
            asyncio.create_task(self.route_and_record_request(request, results))
            
            requests_sent += 1
            
            # Periodic health checks
            if time.time() - self.last_health_check > self.health_check_interval:
                await self.perform_health_checks()
                self.last_health_check = time.time()
            
            # Control request rate
            await asyncio.sleep(request_interval)
        
        # Wait a bit for remaining requests to complete
        await asyncio.sleep(2.0)
        
        execution_time = time.time() - start_time
        
        # Generate load test results
        load_results = self.generate_load_test_results(execution_time, results)
        
        print(f"\nLoad test completed in {execution_time:.2f}s")
        print(f"Requests processed: {len(results)}")
        print(f"Success rate: {load_results['success_rate']:.1%}")
        
        return load_results
    
    async def route_and_record_request(self, request: WorkloadRequest, results: List[Dict[str, Any]]) -> None:
        """Route request and record result"""
        try:
            result = await self.route_request(request)
            results.append(result)
        except Exception as e:
            results.append({
                "request_id": request.id,
                "success": False,
                "error": str(e)
            })
    
    async def perform_health_checks(self) -> None:
        """Perform health checks on all agents"""
        
        unhealthy_agents = []
        
        for agent in self.agents.values():
            agent.update_load_metrics()
            
            if agent.health_status == AgentHealth.FAILED:
                unhealthy_agents.append(agent.agent_id)
        
        # Remove failed agents
        for agent_id in unhealthy_agents:
            print(f"Removing failed agent: {agent_id}")
            self.remove_agent(agent_id)
    
    def generate_load_test_results(self, execution_time: float, 
                                 results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive load test results"""
        
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        # Calculate performance metrics
        if successful_results:
            response_times = [r["processing_time"] for r in successful_results]
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times)
        else:
            avg_response_time = 0.0
            p95_response_time = 0.0
        
        # Agent distribution
        agent_distribution = {}
        for result in successful_results:
            agent_id = result.get("agent_id", "unknown")
            agent_distribution[agent_id] = agent_distribution.get(agent_id, 0) + 1
        
        return {
            "execution_time": execution_time,
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "requests_per_second": len(results) / execution_time if execution_time > 0 else 0,
            "average_response_time": avg_response_time,
            "p95_response_time": p95_response_time,
            "agent_distribution": agent_distribution,
            "load_balancing_strategy": self.strategy.value,
            "agent_health": {agent_id: agent.health_status.value for agent_id, agent in self.agents.items()}
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        agent_metrics = {}
        for agent_id, agent in self.agents.items():
            metrics = agent.get_load_metrics()
            agent_metrics[agent_id] = {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "active_connections": metrics.active_connections,
                "requests_per_second": metrics.requests_per_second,
                "average_response_time": metrics.average_response_time,
                "error_rate": metrics.error_rate,
                "health_status": metrics.health_status.value
            }
        
        return {
            "balancer_id": self.balancer_id,
            "strategy": self.strategy.value,
            "total_agents": len(self.agents),
            "healthy_agents": len([a for a in self.agents.values() if a.health_status == AgentHealth.HEALTHY]),
            "total_requests_processed": self.total_requests,
            "overall_success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            "agent_metrics": agent_metrics
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_web_server_load_balancing():
    """Demo: Web server load balancing like Netflix"""
    print("\nDEMO 1: WEB SERVER LOAD BALANCING")
    print("=" * 50)
    
    # Create load balancer
    web_balancer = LoadBalancer("web_load_balancer", LoadBalancingStrategy.LEAST_CONNECTIONS)
    
    # Add web servers with different capacities
    servers = [
        ("server_1", 1.5, "us_east"),
        ("server_2", 1.2, "us_east"),
        ("server_3", 1.8, "us_west"),
        ("server_4", 1.0, "us_west"),
        ("server_5", 1.3, "europe")
    ]
    
    for server_id, capacity, region in servers:
        server = LoadBalancedAgent(server_id, capacity, region)
        web_balancer.add_agent(server)
    
    # Simulate web traffic
    result = await web_balancer.handle_request_load(
        request_count=500,
        duration=20.0,
        request_rate=30.0  # 30 requests per second
    )
    
    print(f"\nWeb Server Results:")
    print(f"- Requests/second: {result['requests_per_second']:.1f}")
    print(f"- Average response time: {result['average_response_time']:.3f}s")
    print(f"- P95 response time: {result['p95_response_time']:.3f}s")
    
    # Show load distribution
    distribution = result['agent_distribution']
    print(f"\nLoad Distribution:")
    for agent_id, count in sorted(distribution.items()):
        percentage = (count / result['successful_requests']) * 100
        print(f"  {agent_id}: {count} requests ({percentage:.1f}%)")

async def demo_microservice_load_balancing():
    """Demo: Microservice load balancing with different strategies"""
    print("\nDEMO 2: MICROSERVICE LOAD BALANCING COMPARISON")
    print("=" * 50)
    
    strategies = [
        LoadBalancingStrategy.ROUND_ROBIN,
        LoadBalancingStrategy.LEAST_CONNECTIONS,
        LoadBalancingStrategy.RESOURCE_BASED
    ]
    
    results_comparison = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy.value} strategy:")
        
        # Create load balancer with strategy
        balancer = LoadBalancer(f"microservice_{strategy.value}", strategy)
        
        # Add microservice instances
        for i in range(4):
            capacity = random.uniform(0.8, 1.5)
            instance = LoadBalancedAgent(f"service_{i}", capacity)
            balancer.add_agent(instance)
        
        # Test load
        result = await balancer.handle_request_load(
            request_count=200,
            duration=10.0,
            request_rate=25.0
        )
        
        results_comparison[strategy.value] = {
            "success_rate": result['success_rate'],
            "avg_response_time": result['average_response_time'],
            "requests_per_second": result['requests_per_second']
        }
        
        print(f"  Success rate: {result['success_rate']:.1%}")
        print(f"  Avg response time: {result['average_response_time']:.3f}s")
    
    # Compare strategies
    print(f"\nSTRATEGY COMPARISON:")
    for strategy, metrics in results_comparison.items():
        print(f"  {strategy}:")
        print(f"    Success: {metrics['success_rate']:.1%}")
        print(f"    Response: {metrics['avg_response_time']:.3f}s")
        print(f"    Throughput: {metrics['requests_per_second']:.1f} req/s")

async def demo_geographic_load_balancing():
    """Demo: Geographic load balancing for global services"""
    print("\nDEMO 3: GEOGRAPHIC LOAD BALANCING")
    print("=" * 50)
    
    # Create geographic load balancer
    geo_balancer = LoadBalancer("global_cdn", LoadBalancingStrategy.GEOGRAPHIC)
    
    # Add servers in different regions
    regions = [
        ("us_east_1", 1.4, "us_east"),
        ("us_east_2", 1.2, "us_east"),
        ("us_west_1", 1.6, "us_west"),
        ("europe_1", 1.3, "europe"),
        ("europe_2", 1.1, "europe"),
        ("asia_1", 1.5, "asia")
    ]
    
    for server_id, capacity, region in regions:
        server = LoadBalancedAgent(server_id, capacity, region)
        geo_balancer.add_agent(server)
    
    # Simulate global traffic
    result = await geo_balancer.handle_request_load(
        request_count=300,
        duration=15.0,
        request_rate=25.0
    )
    
    print(f"\nGeographic Load Balancing Results:")
    print(f"- Global success rate: {result['success_rate']:.1%}")
    print(f"- Average response time: {result['average_response_time']:.3f}s")
    
    # Show regional distribution
    distribution = result['agent_distribution']
    regional_stats = {}
    
    for agent_id, count in distribution.items():
        # Extract region from agent metrics
        agent = geo_balancer.agents[agent_id]
        region = agent.geographic_region
        regional_stats[region] = regional_stats.get(region, 0) + count
    
    print(f"\nRegional Request Distribution:")
    for region, count in sorted(regional_stats.items()):
        percentage = (count / result['successful_requests']) * 100
        print(f"  {region}: {count} requests ({percentage:.1f}%)")

async def main():
    """
    Demonstrate Agent Load Balancing for optimal resource distribution
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement different load balancing strategies
    2. How to monitor agent health and performance metrics
    3. How to distribute load optimally across multiple agents
    4. How to handle geographic and capacity-based routing
    5. How load balancing improves system performance and reliability
    
    REAL WORLD APPLICATIONS:
    =======================
    - Web server load balancing (Netflix, Google, Amazon)
    - Microservice architecture and API gateways
    - Content delivery networks (CDNs)
    - Database query distribution and read replicas
    - Cloud computing resource allocation
    - Game server matchmaking and distribution
    """
    
    print("AGENT LOAD BALANCING DEMONSTRATION")
    print("This shows how to optimally distribute work across multiple agents!")
    
    await demo_web_server_load_balancing()
    await demo_microservice_load_balancing()
    await demo_geographic_load_balancing()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Load balancing maximizes system throughput and performance")
    print("✓ Different strategies suit different use cases and requirements")
    print("✓ Health monitoring prevents routing to failed or overloaded agents")
    print("✓ Geographic routing improves user experience through proximity")
    print("✓ Dynamic load distribution adapts to changing demand patterns")

if __name__ == "__main__":
    asyncio.run(main())
