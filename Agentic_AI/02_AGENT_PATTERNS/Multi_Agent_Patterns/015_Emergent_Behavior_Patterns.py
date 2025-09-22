#!/usr/bin/env python3
"""
Emergent Behavior Patterns: Complex Systems from Simple Rules
============================================================

WHAT IS THE PROBLEM?
==================
Complex behaviors seem impossible to design directly. But nature shows us that simple rules can create incredibly sophisticated patterns and intelligence.

Example: Traffic Jam Mystery
CENTRALIZED APPROACH (Impossible):
- Traffic control center tries to direct every car individually
- Needs to track millions of vehicles simultaneously
- System overwhelmed by complexity
- Cannot predict or prevent traffic jams
- Single point of failure brings down entire system

REAL WORLD EXAMPLE:
=================
How do bird flocks create beautiful murmurations?

STARLING MURMURATION RULES:
Each bird follows just 3 simple rules:
1. SEPARATION: Don't get too close to neighbors
2. ALIGNMENT: Fly in same direction as nearby birds  
3. COHESION: Stay close to the group

EMERGENT BEHAVIORS:
- No bird is "in charge" or knows the overall pattern
- Flock moves as unified organism
- Creates beautiful, flowing patterns
- Reacts instantly to predators
- Self-organizes without central control
- Scales to thousands of birds seamlessly

THE ALGORITHM:
=============
1. DEFINE: Simple local rules for individual agents
2. INTERACT: Agents follow rules based on local neighbors
3. OBSERVE: Watch for patterns that emerge naturally
4. EVOLVE: Let system adapt and self-organize
5. SCALE: Add more agents to see larger patterns
6. DISCOVER: Find emergent behaviors you never designed

EMERGENCE EXAMPLES:
- Traffic flow patterns from driving rules
- Market trends from individual trading decisions
- Internet structure from local connection choices
- City growth from individual location decisions
- Economic cycles from personal spending patterns

WHY IS THIS REVOLUTIONARY?
========================
- Solves problems too complex for direct design
- Creates adaptive, self-organizing systems
- Produces intelligent behavior without programming it
- Scales naturally to massive systems
- Robust to individual component failures
- Discovers solutions humans never imagined
"""

import asyncio
import time
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics

class EmergentPattern(Enum):
    FLOCKING = "flocking"
    HERDING = "herding"
    SCHOOLING = "schooling"
    CLUSTERING = "clustering"
    WAVE_PROPAGATION = "wave_propagation"
    SELF_ORGANIZATION = "self_organization"
    CONSENSUS_FORMATION = "consensus_formation"
    NETWORK_FORMATION = "network_formation"

class AgentBehavior(Enum):
    EXPLORER = "explorer"
    FOLLOWER = "follower"
    LEADER = "leader"
    CONNECTOR = "connector"
    SPECIALIST = "specialist"

@dataclass
class Position:
    """2D position with basic operations"""
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def normalize(self) -> 'Position':
        magnitude = math.sqrt(self.x**2 + self.y**2)
        if magnitude > 0:
            return Position(self.x / magnitude, self.y / magnitude)
        return Position(0, 0)
    
    def multiply(self, scalar: float) -> 'Position':
        return Position(self.x * scalar, self.y * scalar)
    
    def add(self, other: 'Position') -> 'Position':
        return Position(self.x + other.x, self.y + other.y)

@dataclass
class EmergentAgent:
    """Agent that participates in emergent behavior"""
    id: str
    position: Position
    velocity: Position
    behavior_type: AgentBehavior
    
    # Behavior parameters
    max_speed: float = 2.0
    perception_radius: float = 5.0
    separation_radius: float = 2.0
    
    # State tracking
    neighbors: List['EmergentAgent'] = field(default_factory=list)
    energy: float = 100.0
    age: int = 0
    connections: List[str] = field(default_factory=list)
    
    # Decision factors
    separation_weight: float = 1.5
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    exploration_tendency: float = 0.1

class EmergentSystem:
    """
    System where complex behaviors emerge from simple agent rules
    
    EXAMPLE USAGE:
    =============
    # Create emergent system
    system = EmergentSystem("bird_flock", EmergentPattern.FLOCKING)
    
    # Add agents with simple rules
    for i in range(50):
        agent = EmergentAgent(f"bird_{i}", random_position(), random_velocity(), AgentBehavior.FOLLOWER)
        system.add_agent(agent)
    
    # Run simulation and observe emergent patterns
    result = await system.run_emergence_simulation(duration=60)
    """
    
    def __init__(self, system_id: str, pattern_type: EmergentPattern):
        self.system_id = system_id
        self.pattern_type = pattern_type
        self.agents: Dict[str, EmergentAgent] = {}
        
        # Environment
        self.world_size = 50.0
        self.obstacles: List[Position] = []
        self.attractors: List[Position] = []
        
        # Emergence tracking
        self.iteration = 0
        self.pattern_history: List[Dict[str, Any]] = []
        self.complexity_metrics: List[float] = []
        
        # System parameters
        self.neighbor_update_frequency = 5  # Update neighbors every N iterations
        self.pattern_analysis_frequency = 10  # Analyze patterns every N iterations
        
    def add_agent(self, agent: EmergentAgent) -> None:
        """Add agent to emergent system"""
        self.agents[agent.id] = agent
    
    def add_obstacle(self, position: Position) -> None:
        """Add obstacle to environment"""
        self.obstacles.append(position)
    
    def add_attractor(self, position: Position) -> None:
        """Add attractor to environment"""
        self.attractors.append(position)
    
    async def run_emergence_simulation(self, duration: float = 60.0, 
                                     iterations_per_second: float = 10.0) -> Dict[str, Any]:
        """Run simulation and observe emergent patterns"""
        
        print(f"\nRUNNING EMERGENT BEHAVIOR SIMULATION")
        print(f"Pattern: {self.pattern_type.value}, Agents: {len(self.agents)}")
        print(f"Duration: {duration}s")
        print("-" * 60)
        
        start_time = time.time()
        iteration_interval = 1.0 / iterations_per_second
        
        while time.time() - start_time < duration:
            # Update system state
            await self.update_system()
            
            # Analyze emergent patterns periodically
            if self.iteration % self.pattern_analysis_frequency == 0:
                await self.analyze_emergent_patterns()
            
            # Progress reporting
            if self.iteration % 50 == 0 and self.iteration > 0:
                await self.report_emergence_progress()
            
            self.iteration += 1
            await asyncio.sleep(iteration_interval)
        
        simulation_time = time.time() - start_time
        
        # Generate emergence results
        results = await self.generate_emergence_results(simulation_time)
        
        print(f"\nEmergence simulation completed in {simulation_time:.2f}s")
        print(f"Patterns observed: {len(self.pattern_history)}")
        
        return results
    
    async def update_system(self) -> None:
        """Update all agents based on emergence rules"""
        
        # Update neighbors periodically (expensive operation)
        if self.iteration % self.neighbor_update_frequency == 0:
            self.update_all_neighbors()
        
        # Update agents based on pattern type
        if self.pattern_type == EmergentPattern.FLOCKING:
            await self.update_flocking_behavior()
        elif self.pattern_type == EmergentPattern.CLUSTERING:
            await self.update_clustering_behavior()
        elif self.pattern_type == EmergentPattern.WAVE_PROPAGATION:
            await self.update_wave_behavior()
        elif self.pattern_type == EmergentPattern.CONSENSUS_FORMATION:
            await self.update_consensus_behavior()
        elif self.pattern_type == EmergentPattern.NETWORK_FORMATION:
            await self.update_network_behavior()
        else:
            await self.update_flocking_behavior()  # Default
        
        # Update agent positions
        self.update_agent_positions()
        
        # Apply boundary conditions
        self.apply_boundary_conditions()
    
    def update_all_neighbors(self) -> None:
        """Update neighbor lists for all agents"""
        
        for agent in self.agents.values():
            agent.neighbors = []
            
            for other_agent in self.agents.values():
                if (other_agent.id != agent.id and 
                    agent.position.distance_to(other_agent.position) <= agent.perception_radius):
                    agent.neighbors.append(other_agent)
    
    async def update_flocking_behavior(self) -> None:
        """Update agents using flocking rules (separation, alignment, cohesion)"""
        
        for agent in self.agents.values():
            if not agent.neighbors:
                continue
            
            # Calculate flocking forces
            separation = self.calculate_separation(agent)
            alignment = self.calculate_alignment(agent)
            cohesion = self.calculate_cohesion(agent)
            
            # Apply behavior-specific modifications
            if agent.behavior_type == AgentBehavior.LEADER:
                # Leaders are less influenced by others
                separation = separation.multiply(0.5)
                alignment = alignment.multiply(0.3)
                cohesion = cohesion.multiply(0.3)
                
                # Add exploration tendency
                exploration = Position(random.uniform(-1, 1), random.uniform(-1, 1))
                exploration = exploration.normalize().multiply(agent.exploration_tendency)
                agent.velocity = agent.velocity.add(exploration)
                
            elif agent.behavior_type == AgentBehavior.FOLLOWER:
                # Followers are more influenced by alignment
                alignment = alignment.multiply(1.5)
                
            elif agent.behavior_type == AgentBehavior.EXPLORER:
                # Explorers have random tendency
                exploration = Position(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
                agent.velocity = agent.velocity.add(exploration)
            
            # Combine forces
            total_force = Position(0, 0)
            total_force = total_force.add(separation.multiply(agent.separation_weight))
            total_force = total_force.add(alignment.multiply(agent.alignment_weight))
            total_force = total_force.add(cohesion.multiply(agent.cohesion_weight))
            
            # Apply environmental forces
            obstacle_avoidance = self.calculate_obstacle_avoidance(agent)
            attractor_force = self.calculate_attractor_force(agent)
            
            total_force = total_force.add(obstacle_avoidance.multiply(2.0))
            total_force = total_force.add(attractor_force.multiply(0.5))
            
            # Update velocity
            agent.velocity = agent.velocity.add(total_force.multiply(0.1))
            
            # Limit speed
            speed = math.sqrt(agent.velocity.x**2 + agent.velocity.y**2)
            if speed > agent.max_speed:
                agent.velocity = agent.velocity.normalize().multiply(agent.max_speed)
    
    def calculate_separation(self, agent: EmergentAgent) -> Position:
        """Calculate separation force to avoid crowding"""
        
        separation_force = Position(0, 0)
        close_neighbors = 0
        
        for neighbor in agent.neighbors:
            distance = agent.position.distance_to(neighbor.position)
            if distance < agent.separation_radius and distance > 0:
                # Force pointing away from neighbor
                diff = Position(
                    agent.position.x - neighbor.position.x,
                    agent.position.y - neighbor.position.y
                )
                diff = diff.normalize().multiply(1.0 / distance)  # Inverse distance
                separation_force = separation_force.add(diff)
                close_neighbors += 1
        
        if close_neighbors > 0:
            separation_force = Position(
                separation_force.x / close_neighbors,
                separation_force.y / close_neighbors
            )
        
        return separation_force
    
    def calculate_alignment(self, agent: EmergentAgent) -> Position:
        """Calculate alignment force to match neighbor velocities"""
        
        if not agent.neighbors:
            return Position(0, 0)
        
        avg_velocity = Position(0, 0)
        for neighbor in agent.neighbors:
            avg_velocity = avg_velocity.add(neighbor.velocity)
        
        avg_velocity = Position(
            avg_velocity.x / len(agent.neighbors),
            avg_velocity.y / len(agent.neighbors)
        )
        
        # Alignment force is difference between desired and current velocity
        alignment_force = Position(
            avg_velocity.x - agent.velocity.x,
            avg_velocity.y - agent.velocity.y
        )
        
        return alignment_force.multiply(0.1)
    
    def calculate_cohesion(self, agent: EmergentAgent) -> Position:
        """Calculate cohesion force to move toward neighbor center"""
        
        if not agent.neighbors:
            return Position(0, 0)
        
        # Calculate center of mass of neighbors
        center = Position(0, 0)
        for neighbor in agent.neighbors:
            center = center.add(neighbor.position)
        
        center = Position(
            center.x / len(agent.neighbors),
            center.y / len(agent.neighbors)
        )
        
        # Cohesion force points toward center
        cohesion_force = Position(
            center.x - agent.position.x,
            center.y - agent.position.y
        )
        
        return cohesion_force.multiply(0.01)
    
    def calculate_obstacle_avoidance(self, agent: EmergentAgent) -> Position:
        """Calculate force to avoid obstacles"""
        
        avoidance_force = Position(0, 0)
        
        for obstacle in self.obstacles:
            distance = agent.position.distance_to(obstacle)
            if distance < agent.perception_radius:
                # Force pointing away from obstacle
                diff = Position(
                    agent.position.x - obstacle.x,
                    agent.position.y - obstacle.y
                )
                if distance > 0:
                    avoidance_strength = 1.0 / (distance * distance)
                    avoidance_force = avoidance_force.add(
                        diff.normalize().multiply(avoidance_strength)
                    )
        
        return avoidance_force
    
    def calculate_attractor_force(self, agent: EmergentAgent) -> Position:
        """Calculate force toward attractors"""
        
        attractor_force = Position(0, 0)
        
        for attractor in self.attractors:
            distance = agent.position.distance_to(attractor)
            if distance > 0:
                # Force pointing toward attractor
                diff = Position(
                    attractor.x - agent.position.x,
                    attractor.y - agent.position.y
                )
                attractor_strength = 1.0 / (1.0 + distance)
                attractor_force = attractor_force.add(
                    diff.normalize().multiply(attractor_strength)
                )
        
        return attractor_force
    
    async def update_clustering_behavior(self) -> None:
        """Update agents for clustering emergence"""
        
        for agent in self.agents.values():
            # Simple clustering: move toward densest area
            if agent.neighbors:
                density_center = self.calculate_local_density_center(agent)
                
                # Move toward density center
                direction = Position(
                    density_center.x - agent.position.x,
                    density_center.y - agent.position.y
                )
                
                agent.velocity = agent.velocity.add(direction.multiply(0.02))
            else:
                # Random walk if no neighbors
                random_force = Position(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
                agent.velocity = agent.velocity.add(random_force)
            
            # Limit speed
            speed = math.sqrt(agent.velocity.x**2 + agent.velocity.y**2)
            if speed > agent.max_speed:
                agent.velocity = agent.velocity.normalize().multiply(agent.max_speed)
    
    def calculate_local_density_center(self, agent: EmergentAgent) -> Position:
        """Calculate center of local density"""
        
        if not agent.neighbors:
            return agent.position
        
        # Weight neighbors by inverse distance
        weighted_center = Position(0, 0)
        total_weight = 0
        
        for neighbor in agent.neighbors:
            distance = agent.position.distance_to(neighbor.position)
            weight = 1.0 / (1.0 + distance)
            
            weighted_center = weighted_center.add(
                neighbor.position.multiply(weight)
            )
            total_weight += weight
        
        if total_weight > 0:
            return Position(
                weighted_center.x / total_weight,
                weighted_center.y / total_weight
            )
        
        return agent.position
    
    async def update_wave_behavior(self) -> None:
        """Update agents for wave propagation patterns"""
        
        # Implement wave-like behavior where activation spreads
        wave_speed = 0.1
        
        for agent in self.agents.values():
            # Simple wave: agents get activated by activated neighbors
            if hasattr(agent, 'activation'):
                activation = agent.activation
            else:
                agent.activation = 0.0
                activation = 0.0
            
            # Check for activated neighbors
            neighbor_activation = 0.0
            if agent.neighbors:
                neighbor_activation = max(
                    getattr(neighbor, 'activation', 0.0) for neighbor in agent.neighbors
                )
            
            # Update activation
            if neighbor_activation > 0.5:
                agent.activation = min(1.0, activation + wave_speed)
            else:
                agent.activation = max(0.0, activation - wave_speed * 0.5)
            
            # Move based on activation
            if agent.activation > 0.5:
                # Activated agents move in wave direction
                wave_direction = Position(1.0, 0.0)  # Wave moving right
                agent.velocity = wave_direction.multiply(agent.activation * agent.max_speed)
    
    async def update_consensus_behavior(self) -> None:
        """Update agents for consensus formation"""
        
        for agent in self.agents.values():
            # Simple consensus: agents adopt majority opinion of neighbors
            if hasattr(agent, 'opinion'):
                opinion = agent.opinion
            else:
                agent.opinion = random.choice([0, 1])  # Binary opinion
                opinion = agent.opinion
            
            if agent.neighbors:
                neighbor_opinions = [getattr(neighbor, 'opinion', random.choice([0, 1])) 
                                   for neighbor in agent.neighbors]
                
                # Adopt majority opinion with some probability
                if len(neighbor_opinions) > 0:
                    majority_opinion = 1 if sum(neighbor_opinions) > len(neighbor_opinions) / 2 else 0
                    
                    # Probability of changing opinion
                    change_probability = 0.1
                    if random.random() < change_probability:
                        agent.opinion = majority_opinion
            
            # Move based on opinion (agents with same opinion cluster)
            opinion_direction = Position(1.0 if agent.opinion == 1 else -1.0, 0.0)
            agent.velocity = agent.velocity.add(opinion_direction.multiply(0.01))
    
    async def update_network_behavior(self) -> None:
        """Update agents for network formation"""
        
        for agent in self.agents.values():
            # Form connections based on proximity and behavior
            if agent.behavior_type == AgentBehavior.CONNECTOR:
                # Connectors try to bridge distant groups
                if len(agent.connections) < 5:  # Max connections
                    for neighbor in agent.neighbors:
                        if (neighbor.id not in agent.connections and 
                            len(neighbor.connections) < 3):
                            agent.connections.append(neighbor.id)
                            neighbor.connections.append(agent.id)
            
            # Move to maintain network structure
            if agent.connections:
                # Move toward connected agents
                connected_center = Position(0, 0)
                connected_count = 0
                
                for connection_id in agent.connections:
                    if connection_id in self.agents:
                        connected_agent = self.agents[connection_id]
                        connected_center = connected_center.add(connected_agent.position)
                        connected_count += 1
                
                if connected_count > 0:
                    connected_center = Position(
                        connected_center.x / connected_count,
                        connected_center.y / connected_count
                    )
                    
                    direction = Position(
                        connected_center.x - agent.position.x,
                        connected_center.y - agent.position.y
                    )
                    
                    agent.velocity = agent.velocity.add(direction.multiply(0.01))
    
    def update_agent_positions(self) -> None:
        """Update all agent positions based on velocities"""
        
        for agent in self.agents.values():
            agent.position = agent.position.add(agent.velocity.multiply(0.1))
            agent.age += 1
    
    def apply_boundary_conditions(self) -> None:
        """Apply boundary conditions to keep agents in world"""
        
        for agent in self.agents.values():
            # Wrap around boundaries (toroidal world)
            agent.position.x = agent.position.x % self.world_size
            agent.position.y = agent.position.y % self.world_size
            
            # Alternative: Bounce off boundaries
            # if agent.position.x < 0 or agent.position.x > self.world_size:
            #     agent.velocity.x *= -1
            # if agent.position.y < 0 or agent.position.y > self.world_size:
            #     agent.velocity.y *= -1
    
    async def analyze_emergent_patterns(self) -> None:
        """Analyze and record emergent patterns"""
        
        analysis = {
            "iteration": self.iteration,
            "timestamp": time.time(),
            "agent_count": len(self.agents),
            "pattern_type": self.pattern_type.value
        }
        
        # Calculate system-wide metrics
        if self.pattern_type == EmergentPattern.FLOCKING:
            analysis.update(self.analyze_flocking_patterns())
        elif self.pattern_type == EmergentPattern.CLUSTERING:
            analysis.update(self.analyze_clustering_patterns())
        elif self.pattern_type == EmergentPattern.CONSENSUS_FORMATION:
            analysis.update(self.analyze_consensus_patterns())
        elif self.pattern_type == EmergentPattern.NETWORK_FORMATION:
            analysis.update(self.analyze_network_patterns())
        
        # Calculate complexity metric
        complexity = self.calculate_system_complexity()
        analysis["complexity"] = complexity
        self.complexity_metrics.append(complexity)
        
        self.pattern_history.append(analysis)
    
    def analyze_flocking_patterns(self) -> Dict[str, Any]:
        """Analyze flocking-specific patterns"""
        
        if not self.agents:
            return {}
        
        # Calculate center of mass
        center_x = sum(agent.position.x for agent in self.agents.values()) / len(self.agents)
        center_y = sum(agent.position.y for agent in self.agents.values()) / len(self.agents)
        center_of_mass = Position(center_x, center_y)
        
        # Calculate spread (how dispersed the flock is)
        distances_from_center = [
            agent.position.distance_to(center_of_mass) 
            for agent in self.agents.values()
        ]
        avg_spread = statistics.mean(distances_from_center) if distances_from_center else 0
        
        # Calculate alignment (how similar velocities are)
        velocities = [(agent.velocity.x, agent.velocity.y) for agent in self.agents.values()]
        if velocities:
            avg_velocity_x = statistics.mean([v[0] for v in velocities])
            avg_velocity_y = statistics.mean([v[1] for v in velocities])
            
            velocity_deviations = [
                math.sqrt((v[0] - avg_velocity_x)**2 + (v[1] - avg_velocity_y)**2)
                for v in velocities
            ]
            alignment = 1.0 - (statistics.mean(velocity_deviations) / 2.0)  # Normalized
        else:
            alignment = 0.0
        
        return {
            "center_of_mass": (center_of_mass.x, center_of_mass.y),
            "average_spread": avg_spread,
            "alignment_score": max(0.0, alignment),
            "flock_cohesion": 1.0 / (1.0 + avg_spread)  # Higher cohesion = lower spread
        }
    
    def analyze_clustering_patterns(self) -> Dict[str, Any]:
        """Analyze clustering patterns"""
        
        # Simple clustering analysis: find density peaks
        grid_size = 5.0
        grid_counts = {}
        
        for agent in self.agents.values():
            grid_x = int(agent.position.x // grid_size)
            grid_y = int(agent.position.y // grid_size)
            grid_key = (grid_x, grid_y)
            grid_counts[grid_key] = grid_counts.get(grid_key, 0) + 1
        
        if grid_counts:
            max_density = max(grid_counts.values())
            num_clusters = len([count for count in grid_counts.values() if count > 1])
        else:
            max_density = 0
            num_clusters = 0
        
        return {
            "max_local_density": max_density,
            "number_of_clusters": num_clusters,
            "clustering_efficiency": num_clusters / len(self.agents) if self.agents else 0
        }
    
    def analyze_consensus_patterns(self) -> Dict[str, Any]:
        """Analyze consensus formation patterns"""
        
        opinions = [getattr(agent, 'opinion', 0) for agent in self.agents.values()]
        
        if opinions:
            opinion_0_count = opinions.count(0)
            opinion_1_count = opinions.count(1)
            
            consensus_level = max(opinion_0_count, opinion_1_count) / len(opinions)
        else:
            consensus_level = 0.0
        
        return {
            "consensus_level": consensus_level,
            "opinion_distribution": {"0": opinion_0_count, "1": opinion_1_count} if opinions else {},
            "consensus_reached": consensus_level > 0.8
        }
    
    def analyze_network_patterns(self) -> Dict[str, Any]:
        """Analyze network formation patterns"""
        
        total_connections = sum(len(agent.connections) for agent in self.agents.values())
        avg_connections = total_connections / len(self.agents) if self.agents else 0
        
        # Calculate network density
        max_possible_connections = len(self.agents) * (len(self.agents) - 1) // 2
        network_density = total_connections / max_possible_connections if max_possible_connections > 0 else 0
        
        return {
            "total_connections": total_connections,
            "average_connections_per_agent": avg_connections,
            "network_density": network_density,
            "network_efficiency": min(1.0, avg_connections / 3.0)  # Normalized to ideal of 3 connections
        }
    
    def calculate_system_complexity(self) -> float:
        """Calculate overall system complexity metric"""
        
        if not self.agents:
            return 0.0
        
        # Complexity based on neighbor distribution variation
        neighbor_counts = [len(agent.neighbors) for agent in self.agents.values()]
        
        if len(neighbor_counts) <= 1:
            return 0.0
        
        # Higher variation in neighbor counts = higher complexity
        avg_neighbors = statistics.mean(neighbor_counts)
        neighbor_variance = statistics.variance(neighbor_counts)
        
        # Normalize complexity metric
        complexity = neighbor_variance / (1.0 + avg_neighbors)
        
        return min(1.0, complexity)
    
    async def report_emergence_progress(self) -> None:
        """Report emergence progress"""
        
        if not self.pattern_history:
            return
        
        latest_analysis = self.pattern_history[-1]
        
        print(f"  Iteration {self.iteration}: ", end="")
        
        if self.pattern_type == EmergentPattern.FLOCKING:
            alignment = latest_analysis.get("alignment_score", 0)
            cohesion = latest_analysis.get("flock_cohesion", 0)
            print(f"Alignment: {alignment:.2f}, Cohesion: {cohesion:.2f}")
        
        elif self.pattern_type == EmergentPattern.CLUSTERING:
            clusters = latest_analysis.get("number_of_clusters", 0)
            max_density = latest_analysis.get("max_local_density", 0)
            print(f"Clusters: {clusters}, Max density: {max_density}")
        
        elif self.pattern_type == EmergentPattern.CONSENSUS_FORMATION:
            consensus = latest_analysis.get("consensus_level", 0)
            reached = latest_analysis.get("consensus_reached", False)
            print(f"Consensus: {consensus:.1%}, Reached: {reached}")
        
        else:
            complexity = latest_analysis.get("complexity", 0)
            print(f"Complexity: {complexity:.2f}")
    
    async def generate_emergence_results(self, simulation_time: float) -> Dict[str, Any]:
        """Generate comprehensive emergence results"""
        
        if not self.pattern_history:
            return {"error": "No pattern data collected"}
        
        # Calculate emergence metrics
        final_analysis = self.pattern_history[-1]
        
        # Complexity evolution
        complexity_trend = "stable"
        if len(self.complexity_metrics) > 10:
            early_complexity = statistics.mean(self.complexity_metrics[:5])
            late_complexity = statistics.mean(self.complexity_metrics[-5:])
            
            if late_complexity > early_complexity * 1.2:
                complexity_trend = "increasing"
            elif late_complexity < early_complexity * 0.8:
                complexity_trend = "decreasing"
        
        # Pattern stability
        pattern_stability = self.calculate_pattern_stability()
        
        results = {
            "system_id": self.system_id,
            "pattern_type": self.pattern_type.value,
            "simulation_time": simulation_time,
            "total_iterations": self.iteration,
            "agent_count": len(self.agents),
            "final_analysis": final_analysis,
            "complexity_trend": complexity_trend,
            "pattern_stability": pattern_stability,
            "emergence_quality": self.assess_emergence_quality(final_analysis),
            "system_behavior_summary": self.generate_behavior_summary()
        }
        
        return results
    
    def calculate_pattern_stability(self) -> float:
        """Calculate how stable the emergent patterns are"""
        
        if len(self.complexity_metrics) < 10:
            return 0.5  # Not enough data
        
        # Stability = low variation in recent complexity metrics
        recent_complexity = self.complexity_metrics[-10:]
        complexity_variance = statistics.variance(recent_complexity)
        
        # Lower variance = higher stability
        stability = 1.0 / (1.0 + complexity_variance * 10)
        
        return stability
    
    def assess_emergence_quality(self, final_analysis: Dict[str, Any]) -> float:
        """Assess the quality of emergent behavior"""
        
        if self.pattern_type == EmergentPattern.FLOCKING:
            alignment = final_analysis.get("alignment_score", 0)
            cohesion = final_analysis.get("flock_cohesion", 0)
            return (alignment + cohesion) / 2.0
        
        elif self.pattern_type == EmergentPattern.CLUSTERING:
            efficiency = final_analysis.get("clustering_efficiency", 0)
            return efficiency
        
        elif self.pattern_type == EmergentPattern.CONSENSUS_FORMATION:
            consensus = final_analysis.get("consensus_level", 0)
            return consensus
        
        elif self.pattern_type == EmergentPattern.NETWORK_FORMATION:
            efficiency = final_analysis.get("network_efficiency", 0)
            return efficiency
        
        else:
            return final_analysis.get("complexity", 0)
    
    def generate_behavior_summary(self) -> str:
        """Generate human-readable behavior summary"""
        
        if self.pattern_type == EmergentPattern.FLOCKING:
            return "Agents formed cohesive flocks with coordinated movement patterns"
        elif self.pattern_type == EmergentPattern.CLUSTERING:
            return "Agents self-organized into distinct clusters and groups"
        elif self.pattern_type == EmergentPattern.CONSENSUS_FORMATION:
            return "Agents converged on shared opinions through local interactions"
        elif self.pattern_type == EmergentPattern.NETWORK_FORMATION:
            return "Agents formed complex network structures through connection preferences"
        else:
            return "Complex emergent behaviors observed from simple agent rules"

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_bird_flock_murmuration():
    """Demo: Bird flock creating murmuration patterns"""
    print("\nDEMO 1: BIRD FLOCK MURMURATION")
    print("=" * 50)
    
    # Create flocking system
    flock = EmergentSystem("starling_murmuration", EmergentPattern.FLOCKING)
    
    # Add birds with different behaviors
    for i in range(30):
        x = random.uniform(10, 40)
        y = random.uniform(10, 40)
        vx = random.uniform(-1, 1)
        vy = random.uniform(-1, 1)
        
        # Mostly followers with some leaders
        behavior = AgentBehavior.LEADER if i < 3 else AgentBehavior.FOLLOWER
        
        bird = EmergentAgent(
            id=f"bird_{i}",
            position=Position(x, y),
            velocity=Position(vx, vy),
            behavior_type=behavior
        )
        
        flock.add_agent(bird)
    
    # Add predator as attractor (birds flee from it)
    flock.add_obstacle(Position(25, 25))
    
    # Run murmuration simulation
    result = await flock.run_emergence_simulation(duration=20.0, iterations_per_second=15)
    
    print(f"\nMurmuration Results:")
    print(f"- Flock alignment: {result['final_analysis'].get('alignment_score', 0):.2f}")
    print(f"- Flock cohesion: {result['final_analysis'].get('flock_cohesion', 0):.2f}")
    print(f"- Emergence quality: {result['emergence_quality']:.2f}")
    print(f"- Pattern stability: {result['pattern_stability']:.2f}")

async def demo_city_formation():
    """Demo: Urban clustering emergence like city formation"""
    print("\nDEMO 2: CITY FORMATION THROUGH CLUSTERING")
    print("=" * 50)
    
    # Create clustering system
    city_system = EmergentSystem("urban_development", EmergentPattern.CLUSTERING)
    
    # Add people looking for places to settle
    for i in range(40):
        x = random.uniform(5, 45)
        y = random.uniform(5, 45)
        
        # Different types of settlers
        behavior_types = [AgentBehavior.EXPLORER, AgentBehavior.FOLLOWER, AgentBehavior.CONNECTOR]
        behavior = random.choice(behavior_types)
        
        settler = EmergentAgent(
            id=f"settler_{i}",
            position=Position(x, y),
            velocity=Position(0, 0),
            behavior_type=behavior
        )
        
        city_system.add_agent(settler)
    
    # Add resource locations as attractors
    city_system.add_attractor(Position(15, 15))  # Water source
    city_system.add_attractor(Position(35, 35))  # Trade route
    
    # Run city formation simulation
    result = await city_system.run_emergence_simulation(duration=15.0, iterations_per_second=10)
    
    print(f"\nCity Formation Results:")
    print(f"- Number of clusters: {result['final_analysis'].get('number_of_clusters', 0)}")
    print(f"- Max local density: {result['final_analysis'].get('max_local_density', 0)}")
    print(f"- Clustering efficiency: {result['final_analysis'].get('clustering_efficiency', 0):.2f}")

async def demo_opinion_dynamics():
    """Demo: Consensus formation like social media opinion spread"""
    print("\nDEMO 3: SOCIAL OPINION CONSENSUS FORMATION")
    print("=" * 50)
    
    # Create consensus system
    social_network = EmergentSystem("opinion_dynamics", EmergentPattern.CONSENSUS_FORMATION)
    
    # Add people with initial random opinions
    for i in range(25):
        x = random.uniform(5, 45)
        y = random.uniform(5, 45)
        
        # Influencers vs regular people
        behavior = AgentBehavior.LEADER if i < 3 else AgentBehavior.FOLLOWER
        
        person = EmergentAgent(
            id=f"person_{i}",
            position=Position(x, y),
            velocity=Position(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)),
            behavior_type=behavior
        )
        
        # Set initial random opinion
        person.opinion = random.choice([0, 1])
        
        social_network.add_agent(person)
    
    # Run opinion consensus simulation
    result = await social_network.run_emergence_simulation(duration=18.0, iterations_per_second=8)
    
    print(f"\nOpinion Consensus Results:")
    print(f"- Final consensus level: {result['final_analysis'].get('consensus_level', 0):.1%}")
    print(f"- Consensus reached: {result['final_analysis'].get('consensus_reached', False)}")
    
    distribution = result['final_analysis'].get('opinion_distribution', {})
    if distribution:
        print(f"- Opinion distribution: {distribution}")

async def demo_network_emergence():
    """Demo: Social network formation from local connections"""
    print("\nDEMO 4: SOCIAL NETWORK EMERGENCE")
    print("=" * 50)
    
    # Create network formation system
    network_system = EmergentSystem("social_network", EmergentPattern.NETWORK_FORMATION)
    
    # Add people with different networking behaviors
    for i in range(20):
        x = random.uniform(10, 40)
        y = random.uniform(10, 40)
        
        # Different networking styles
        behavior_types = [AgentBehavior.CONNECTOR, AgentBehavior.FOLLOWER, AgentBehavior.SPECIALIST]
        behavior = random.choice(behavior_types)
        
        person = EmergentAgent(
            id=f"networker_{i}",
            position=Position(x, y),
            velocity=Position(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3)),
            behavior_type=behavior
        )
        
        person.connections = []  # Initialize connections
        
        network_system.add_agent(person)
    
    # Run network formation simulation
    result = await network_system.run_emergence_simulation(duration=12.0, iterations_per_second=12)
    
    print(f"\nNetwork Formation Results:")
    print(f"- Total connections: {result['final_analysis'].get('total_connections', 0)}")
    print(f"- Average connections per person: {result['final_analysis'].get('average_connections_per_agent', 0):.1f}")
    print(f"- Network density: {result['final_analysis'].get('network_density', 0):.2f}")
    print(f"- Network efficiency: {result['final_analysis'].get('network_efficiency', 0):.2f}")

async def main():
    """
    Demonstrate Emergent Behavior Patterns from simple agent rules
    
    WHAT YOU'LL LEARN:
    ================
    1. How complex behaviors emerge from simple local rules
    2. How to design agent interactions that create desired patterns
    3. How emergence enables self-organization and adaptation
    4. How to measure and analyze emergent system properties
    5. How emergent systems solve problems no individual agent can solve
    
    REAL WORLD APPLICATIONS:
    =======================
    - Traffic flow optimization and smart city planning
    - Social media dynamics and viral content spread
    - Economic market behavior and trading patterns
    - Biological systems modeling (flocks, swarms, ecosystems)
    - Organizational behavior and team dynamics
    - Distributed computing and self-organizing networks
    """
    
    print("EMERGENT BEHAVIOR PATTERNS DEMONSTRATION")
    print("This shows how complex intelligence emerges from simple rules!")
    
    await demo_bird_flock_murmuration()
    await demo_city_formation()
    await demo_opinion_dynamics()
    await demo_network_emergence()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Simple local rules create complex global behaviors")
    print("✓ Emergence enables self-organization without central control")
    print("✓ Collective intelligence exceeds individual capabilities")
    print("✓ Emergent systems adapt and evolve automatically")
    print("✓ Complexity arises naturally from agent interactions")
    print("\nTHE POWER OF EMERGENCE:")
    print("- Solves problems too complex for direct programming")
    print("- Creates adaptive, resilient systems")
    print("- Scales naturally to massive numbers of agents")
    print("- Produces behaviors that surprise even the designers")

if __name__ == "__main__":
    asyncio.run(main())
