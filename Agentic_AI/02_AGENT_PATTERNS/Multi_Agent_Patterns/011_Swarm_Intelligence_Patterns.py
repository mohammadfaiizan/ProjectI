#!/usr/bin/env python3
"""
Swarm Intelligence Patterns: Collective Intelligence from Simple Agents
=======================================================================

WHAT IS THE PROBLEM?
==================
Complex problems seem impossible when you think about them centrally, but nature shows us that simple agents following basic rules can solve them.

Example: Traffic Jam Confusion
BAD APPROACH (Central Control):
- Traffic control center tries to route every car
- Impossible to track millions of vehicles
- System overwhelmed by complexity
- Single point of failure

REAL WORLD EXAMPLE:
=================
How do ants find the shortest path to food?

ANT COLONY OPTIMIZATION:
1. Ants randomly explore for food
2. When ant finds food, it leaves pheromone trail back to nest
3. Other ants follow strong pheromone trails
4. More ants on good paths → stronger pheromones
5. Shorter paths get reinforced faster
6. Eventually, shortest path emerges without central planning

SWARM RULES:
- Each ant follows simple local rules
- No ant knows the global picture
- Collective intelligence emerges from interactions
- Self-organizing system finds optimal solutions

THE ALGORITHM:
=============
1. INITIALIZE: Create swarm of simple agents
2. LOCAL RULES: Each agent follows basic behavior rules
3. INTERACT: Agents influence each other locally
4. EMERGE: Global patterns emerge from local interactions
5. ADAPT: System adapts to changes automatically
6. OPTIMIZE: Swarm finds optimal solutions collectively

WHY IS THIS REVOLUTIONARY?
========================
- Solves complex problems with simple agents
- No central control or planning needed
- Robust to individual agent failures
- Scales to massive numbers of agents
- Finds solutions humans can't imagine
"""

import asyncio
import time
import random
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class SwarmType(Enum):
    ANT_COLONY = "ant_colony"
    PARTICLE_SWARM = "particle_swarm"
    BEE_COLONY = "bee_colony"
    FLOCKING = "flocking"

@dataclass
class Position:
    """2D position"""
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Pheromone:
    """Pheromone trail left by ants"""
    position: Position
    strength: float
    deposited_at: float = field(default_factory=time.time)
    
    def evaporate(self, evaporation_rate: float = 0.1) -> None:
        """Pheromones evaporate over time"""
        age = time.time() - self.deposited_at
        self.strength *= (1 - evaporation_rate * age)

class SwarmAgent:
    """Basic swarm agent with simple behaviors"""
    
    def __init__(self, agent_id: str, position: Position, swarm_type: SwarmType):
        self.agent_id = agent_id
        self.position = position
        self.swarm_type = swarm_type
        
        # Movement and behavior
        self.velocity = Position(0.0, 0.0)
        self.best_position = Position(position.x, position.y)
        self.best_fitness = float('-inf')
        
        # Swarm-specific attributes
        self.trail: List[Position] = []
        self.carrying_food = False
        self.energy = 100.0
        
        # Behavior parameters
        self.max_speed = 2.0
        self.sensing_range = 5.0
        self.communication_range = 3.0

    async def update(self, swarm: 'SwarmSystem', environment: Dict[str, Any]) -> None:
        """Update agent based on swarm type"""
        
        if self.swarm_type == SwarmType.ANT_COLONY:
            await self.ant_behavior(swarm, environment)
        elif self.swarm_type == SwarmType.PARTICLE_SWARM:
            await self.particle_behavior(swarm, environment)
        elif self.swarm_type == SwarmType.BEE_COLONY:
            await self.bee_behavior(swarm, environment)
        elif self.swarm_type == SwarmType.FLOCKING:
            await self.flocking_behavior(swarm, environment)
    
    async def ant_behavior(self, swarm: 'SwarmSystem', environment: Dict[str, Any]) -> None:
        """Ant colony optimization behavior"""
        
        if not self.carrying_food:
            # Search for food
            await self.search_for_food(swarm, environment)
        else:
            # Return to nest with food
            await self.return_to_nest(swarm, environment)
        
        # Leave pheromone trail
        await self.deposit_pheromone(swarm)
    
    async def search_for_food(self, swarm: 'SwarmSystem', environment: Dict[str, Any]) -> None:
        """Search for food sources"""
        
        food_sources = environment.get('food_sources', [])
        
        # Check if food is nearby
        for food_pos in food_sources:
            if self.position.distance_to(food_pos) < 1.0:
                self.carrying_food = True
                print(f"  {self.agent_id} found food at ({food_pos.x:.1f}, {food_pos.y:.1f})")
                return
        
        # Follow pheromone trails or explore randomly
        direction = await self.choose_direction(swarm)
        await self.move_in_direction(direction)
    
    async def return_to_nest(self, swarm: 'SwarmSystem', environment: Dict[str, Any]) -> None:
        """Return to nest with food"""
        
        nest_position = environment.get('nest_position', Position(0, 0))
        
        # Check if reached nest
        if self.position.distance_to(nest_position) < 1.0:
            self.carrying_food = False
            swarm.food_collected += 1
            print(f"  {self.agent_id} delivered food to nest (total: {swarm.food_collected})")
            return
        
        # Move toward nest
        direction = self.calculate_direction_to(nest_position)
        await self.move_in_direction(direction)
    
    async def choose_direction(self, swarm: 'SwarmSystem') -> Position:
        """Choose movement direction based on pheromones"""
        
        # Get nearby pheromones
        nearby_pheromones = [p for p in swarm.pheromone_map 
                            if self.position.distance_to(p.position) < self.sensing_range]
        
        if nearby_pheromones:
            # Follow strongest pheromone trail
            strongest = max(nearby_pheromones, key=lambda p: p.strength)
            return self.calculate_direction_to(strongest.position)
        else:
            # Random exploration
            angle = random.uniform(0, 2 * math.pi)
            return Position(math.cos(angle), math.sin(angle))
    
    def calculate_direction_to(self, target: Position) -> Position:
        """Calculate normalized direction vector to target"""
        dx = target.x - self.position.x
        dy = target.y - self.position.y
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            return Position(dx/length, dy/length)
        else:
            return Position(0, 0)
    
    async def move_in_direction(self, direction: Position) -> None:
        """Move in specified direction"""
        step_size = 0.5
        self.position.x += direction.x * step_size
        self.position.y += direction.y * step_size
        
        # Add to trail
        self.trail.append(Position(self.position.x, self.position.y))
        if len(self.trail) > 10:  # Keep last 10 positions
            self.trail.pop(0)
    
    async def deposit_pheromone(self, swarm: 'SwarmSystem') -> None:
        """Deposit pheromone at current position"""
        if self.carrying_food:
            # Stronger pheromone when carrying food
            strength = 10.0
        else:
            # Weaker exploration pheromone
            strength = 1.0
        
        pheromone = Pheromone(
            position=Position(self.position.x, self.position.y),
            strength=strength
        )
        swarm.pheromone_map.append(pheromone)
    
    async def particle_behavior(self, swarm: 'SwarmSystem', environment: Dict[str, Any]) -> None:
        """Particle swarm optimization behavior"""
        
        # Evaluate fitness at current position
        fitness = self.evaluate_fitness(environment)
        
        # Update personal best
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_position = Position(self.position.x, self.position.y)
        
        # Update velocity based on personal and global best
        await self.update_particle_velocity(swarm)
        
        # Update position
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y
    
    def evaluate_fitness(self, environment: Dict[str, Any]) -> float:
        """Evaluate fitness at current position"""
        # Simple fitness function - distance to target
        target = environment.get('target_position', Position(10, 10))
        distance = self.position.distance_to(target)
        return -distance  # Closer to target = higher fitness
    
    async def update_particle_velocity(self, swarm: 'SwarmSystem') -> None:
        """Update particle velocity using PSO formula"""
        
        # PSO parameters
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        # Random factors
        r1 = random.random()
        r2 = random.random()
        
        # Update velocity components
        cognitive_x = c1 * r1 * (self.best_position.x - self.position.x)
        cognitive_y = c1 * r1 * (self.best_position.y - self.position.y)
        
        social_x = c2 * r2 * (swarm.global_best_position.x - self.position.x)
        social_y = c2 * r2 * (swarm.global_best_position.y - self.position.y)
        
        self.velocity.x = w * self.velocity.x + cognitive_x + social_x
        self.velocity.y = w * self.velocity.y + cognitive_y + social_y
        
        # Limit velocity
        max_vel = 2.0
        if abs(self.velocity.x) > max_vel:
            self.velocity.x = max_vel if self.velocity.x > 0 else -max_vel
        if abs(self.velocity.y) > max_vel:
            self.velocity.y = max_vel if self.velocity.y > 0 else -max_vel
    
    async def bee_behavior(self, swarm: 'SwarmSystem', environment: Dict[str, Any]) -> None:
        """Bee colony optimization behavior"""
        
        # Simplified bee behavior - search for flowers
        flowers = environment.get('flowers', [])
        
        # Find nearby flowers
        nearby_flowers = [f for f in flowers 
                         if self.position.distance_to(f) < self.sensing_range]
        
        if nearby_flowers:
            # Move to best flower
            best_flower = min(nearby_flowers, key=lambda f: self.position.distance_to(f))
            direction = self.calculate_direction_to(best_flower)
            await self.move_in_direction(direction)
        else:
            # Random search
            angle = random.uniform(0, 2 * math.pi)
            direction = Position(math.cos(angle), math.sin(angle))
            await self.move_in_direction(direction)
    
    async def flocking_behavior(self, swarm: 'SwarmSystem', environment: Dict[str, Any]) -> None:
        """Flocking behavior (boids)"""
        
        nearby_agents = swarm.get_agents_within_range(self.position, self.communication_range)
        
        if nearby_agents:
            # Apply flocking rules
            separation = self.calculate_separation(nearby_agents)
            alignment = self.calculate_alignment(nearby_agents)
            cohesion = self.calculate_cohesion(nearby_agents)
            
            # Combine forces
            self.velocity.x += separation.x * 0.5 + alignment.x * 0.3 + cohesion.x * 0.2
            self.velocity.y += separation.y * 0.5 + alignment.y * 0.3 + cohesion.y * 0.2
        
        # Update position
        await self.move_in_direction(self.velocity)
    
    def calculate_separation(self, nearby_agents: List['SwarmAgent']) -> Position:
        """Separation: steer to avoid crowding local flockmates"""
        avg_direction = Position(0, 0)
        count = 0
        
        for agent in nearby_agents:
            if agent.agent_id != self.agent_id:
                distance = self.position.distance_to(agent.position)
                if distance < 2.0:  # Too close
                    # Direction away from neighbor
                    direction = self.calculate_direction_to(agent.position)
                    avg_direction.x -= direction.x / distance
                    avg_direction.y -= direction.y / distance
                    count += 1
        
        if count > 0:
            avg_direction.x /= count
            avg_direction.y /= count
        
        return avg_direction
    
    def calculate_alignment(self, nearby_agents: List['SwarmAgent']) -> Position:
        """Alignment: steer towards average heading of neighbors"""
        avg_velocity = Position(0, 0)
        count = 0
        
        for agent in nearby_agents:
            if agent.agent_id != self.agent_id:
                avg_velocity.x += agent.velocity.x
                avg_velocity.y += agent.velocity.y
                count += 1
        
        if count > 0:
            avg_velocity.x /= count
            avg_velocity.y /= count
        
        return avg_velocity
    
    def calculate_cohesion(self, nearby_agents: List['SwarmAgent']) -> Position:
        """Cohesion: steer to move toward average position of neighbors"""
        center_of_mass = Position(0, 0)
        count = 0
        
        for agent in nearby_agents:
            if agent.agent_id != self.agent_id:
                center_of_mass.x += agent.position.x
                center_of_mass.y += agent.position.y
                count += 1
        
        if count > 0:
            center_of_mass.x /= count
            center_of_mass.y /= count
            return self.calculate_direction_to(center_of_mass)
        
        return Position(0, 0)

class SwarmSystem:
    """
    Swarm intelligence system managing collective behavior
    
    EXAMPLE USAGE:
    =============
    # Create ant colony
    swarm = SwarmSystem("ant_colony", SwarmType.ANT_COLONY)
    
    # Add ants
    for i in range(50):
        ant = SwarmAgent(f"ant_{i}", Position(0, 0), SwarmType.ANT_COLONY)
        swarm.add_agent(ant)
    
    # Run foraging simulation
    result = await swarm.run_simulation(duration=60)
    """
    
    def __init__(self, system_id: str, swarm_type: SwarmType):
        self.system_id = system_id
        self.swarm_type = swarm_type
        self.agents: Dict[str, SwarmAgent] = {}
        
        # Swarm state
        self.global_best_position = Position(0, 0)
        self.global_best_fitness = float('-inf')
        self.pheromone_map: List[Pheromone] = []
        self.food_collected = 0
        
        # Environment
        self.environment: Dict[str, Any] = {}
        
        # Simulation parameters
        self.iteration_count = 0
        self.convergence_threshold = 0.01
    
    def add_agent(self, agent: SwarmAgent) -> None:
        """Add agent to swarm"""
        self.agents[agent.agent_id] = agent
    
    def setup_environment(self, **kwargs) -> None:
        """Setup environment for swarm"""
        self.environment.update(kwargs)
    
    async def run_simulation(self, duration: float = 30.0, max_iterations: int = 1000) -> Dict[str, Any]:
        """Run swarm simulation"""
        
        print(f"\nRUNNING SWARM SIMULATION: {self.swarm_type.value}")
        print(f"Agents: {len(self.agents)}, Duration: {duration}s")
        print("-" * 50)
        
        start_time = time.time()
        iteration = 0
        
        while (time.time() - start_time < duration and 
               iteration < max_iterations):
            
            # Update all agents
            await self.update_swarm()
            
            # Update global state
            self.update_global_state()
            
            # Environment maintenance
            if self.swarm_type == SwarmType.ANT_COLONY:
                self.update_pheromones()
            
            iteration += 1
            
            # Progress reporting
            if iteration % 50 == 0:
                await self.report_progress(iteration)
            
            # Brief pause for visualization
            await asyncio.sleep(0.02)
        
        simulation_time = time.time() - start_time
        
        # Generate results
        results = self.generate_results(simulation_time, iteration)
        
        print(f"\nSimulation completed in {simulation_time:.2f}s ({iteration} iterations)")
        
        return results
    
    async def update_swarm(self) -> None:
        """Update all agents in parallel"""
        
        update_tasks = []
        for agent in self.agents.values():
            task = agent.update(self, self.environment)
            update_tasks.append(task)
        
        # Update all agents simultaneously
        await asyncio.gather(*update_tasks)
    
    def update_global_state(self) -> None:
        """Update global swarm state"""
        
        if self.swarm_type == SwarmType.PARTICLE_SWARM:
            # Update global best for PSO
            for agent in self.agents.values():
                if agent.best_fitness > self.global_best_fitness:
                    self.global_best_fitness = agent.best_fitness
                    self.global_best_position = Position(
                        agent.best_position.x, 
                        agent.best_position.y
                    )
    
    def update_pheromones(self) -> None:
        """Update pheromone trails"""
        
        # Evaporate pheromones
        for pheromone in self.pheromone_map[:]:
            pheromone.evaporate(0.05)  # 5% evaporation per iteration
            
            # Remove weak pheromones
            if pheromone.strength < 0.1:
                self.pheromone_map.remove(pheromone)
    
    def get_agents_within_range(self, position: Position, range_distance: float) -> List[SwarmAgent]:
        """Get agents within range of position"""
        
        nearby_agents = []
        for agent in self.agents.values():
            if position.distance_to(agent.position) <= range_distance:
                nearby_agents.append(agent)
        
        return nearby_agents
    
    async def report_progress(self, iteration: int) -> None:
        """Report simulation progress"""
        
        if self.swarm_type == SwarmType.ANT_COLONY:
            active_pheromones = len(self.pheromone_map)
            print(f"  Iteration {iteration}: {self.food_collected} food collected, "
                  f"{active_pheromones} active pheromones")
        
        elif self.swarm_type == SwarmType.PARTICLE_SWARM:
            print(f"  Iteration {iteration}: Best fitness: {self.global_best_fitness:.3f}")
        
        else:
            # Calculate swarm spread
            positions = [agent.position for agent in self.agents.values()]
            avg_x = sum(p.x for p in positions) / len(positions)
            avg_y = sum(p.y for p in positions) / len(positions)
            spread = sum(Position(avg_x, avg_y).distance_to(p) for p in positions) / len(positions)
            print(f"  Iteration {iteration}: Average spread: {spread:.2f}")
    
    def generate_results(self, simulation_time: float, iterations: int) -> Dict[str, Any]:
        """Generate comprehensive simulation results"""
        
        results = {
            "system_id": self.system_id,
            "swarm_type": self.swarm_type.value,
            "simulation_time": simulation_time,
            "iterations": iterations,
            "agent_count": len(self.agents)
        }
        
        if self.swarm_type == SwarmType.ANT_COLONY:
            results.update({
                "food_collected": self.food_collected,
                "final_pheromone_trails": len(self.pheromone_map),
                "efficiency": self.food_collected / simulation_time if simulation_time > 0 else 0
            })
        
        elif self.swarm_type == SwarmType.PARTICLE_SWARM:
            results.update({
                "global_best_fitness": self.global_best_fitness,
                "global_best_position": (self.global_best_position.x, self.global_best_position.y),
                "convergence_achieved": self.global_best_fitness > -1.0  # Simple convergence check
            })
        
        elif self.swarm_type == SwarmType.FLOCKING:
            # Calculate final cohesion
            positions = [agent.position for agent in self.agents.values()]
            center_x = sum(p.x for p in positions) / len(positions)
            center_y = sum(p.y for p in positions) / len(positions)
            center = Position(center_x, center_y)
            
            avg_distance = sum(center.distance_to(p) for p in positions) / len(positions)
            results.update({
                "final_cohesion": 1.0 / (1.0 + avg_distance),  # Higher is more cohesive
                "swarm_center": (center_x, center_y)
            })
        
        return results

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_ant_colony_foraging():
    """Demo: Ant colony finding optimal paths to food"""
    print("\nDEMO 1: ANT COLONY FORAGING")
    print("=" * 50)
    
    # Create ant colony
    colony = SwarmSystem("ant_colony", SwarmType.ANT_COLONY)
    
    # Add ants starting at nest
    for i in range(30):
        ant = SwarmAgent(f"ant_{i}", Position(0, 0), SwarmType.ANT_COLONY)
        colony.add_agent(ant)
    
    # Setup environment with food sources
    colony.setup_environment(
        nest_position=Position(0, 0),
        food_sources=[
            Position(10, 5),   # Close food
            Position(15, 15),  # Far food
            Position(8, -3)    # Another close food
        ]
    )
    
    # Run foraging simulation
    result = await colony.run_simulation(duration=20.0)
    
    print(f"\nAnt Colony Results:")
    print(f"- Food collected: {result['food_collected']}")
    print(f"- Efficiency: {result['efficiency']:.2f} food/second")
    print(f"- Final pheromone trails: {result['final_pheromone_trails']}")

async def demo_particle_swarm_optimization():
    """Demo: Particle swarm finding optimal solution"""
    print("\nDEMO 2: PARTICLE SWARM OPTIMIZATION")
    print("=" * 50)
    
    # Create particle swarm
    swarm = SwarmSystem("pso", SwarmType.PARTICLE_SWARM)
    
    # Add particles with random starting positions
    for i in range(25):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        particle = SwarmAgent(f"particle_{i}", Position(x, y), SwarmType.PARTICLE_SWARM)
        swarm.add_agent(particle)
    
    # Target position for optimization
    swarm.setup_environment(target_position=Position(5, 7))
    
    # Run optimization
    result = await swarm.run_simulation(duration=15.0)
    
    print(f"\nParticle Swarm Results:")
    print(f"- Best fitness: {result['global_best_fitness']:.3f}")
    print(f"- Best position: ({result['global_best_position'][0]:.2f}, {result['global_best_position'][1]:.2f})")
    print(f"- Target position: (5.00, 7.00)")
    print(f"- Convergence: {'Yes' if result['convergence_achieved'] else 'No'}")

async def demo_flocking_behavior():
    """Demo: Flocking behavior (boids)"""
    print("\nDEMO 3: FLOCKING BEHAVIOR")
    print("=" * 50)
    
    # Create flocking system
    flock = SwarmSystem("boids", SwarmType.FLOCKING)
    
    # Add boids with random positions and velocities
    for i in range(20):
        x = random.uniform(-5, 5)
        y = random.uniform(-5, 5)
        boid = SwarmAgent(f"boid_{i}", Position(x, y), SwarmType.FLOCKING)
        
        # Random initial velocity
        boid.velocity = Position(
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        )
        
        flock.add_agent(boid)
    
    # Run flocking simulation
    result = await flock.run_simulation(duration=12.0)
    
    print(f"\nFlocking Results:")
    print(f"- Final cohesion: {result['final_cohesion']:.3f}")
    print(f"- Swarm center: ({result['swarm_center'][0]:.2f}, {result['swarm_center'][1]:.2f})")
    print(f"- Iterations: {result['iterations']}")

async def demo_bee_colony_flower_search():
    """Demo: Bee colony searching for flowers"""
    print("\nDEMO 4: BEE COLONY FLOWER SEARCH")
    print("=" * 50)
    
    # Create bee colony
    hive = SwarmSystem("bee_colony", SwarmType.BEE_COLONY)
    
    # Add bees starting near hive
    for i in range(15):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        bee = SwarmAgent(f"bee_{i}", Position(x, y), SwarmType.BEE_COLONY)
        hive.add_agent(bee)
    
    # Setup environment with flower patches
    hive.setup_environment(
        flowers=[
            Position(8, 3),
            Position(-5, 6),
            Position(12, -4),
            Position(2, 9),
            Position(-8, -2)
        ]
    )
    
    # Run flower search simulation
    result = await hive.run_simulation(duration=10.0)
    
    print(f"\nBee Colony Results:")
    print(f"- Simulation completed successfully")
    print(f"- Bees explored {result['iterations']} iterations")
    print(f"- Total bees: {result['agent_count']}")

async def main():
    """
    Demonstrate Swarm Intelligence Patterns for collective problem solving
    
    WHAT YOU'LL LEARN:
    ================
    1. How simple agents following basic rules create intelligent behavior
    2. How swarm systems solve complex optimization problems
    3. How emergent behavior arises from local interactions
    4. How to implement ant colony, particle swarm, and flocking algorithms
    5. How swarm intelligence scales to large numbers of agents
    
    REAL WORLD APPLICATIONS:
    =======================
    - Logistics and supply chain optimization (UPS route planning)
    - Internet routing and network optimization
    - Financial trading algorithms and market analysis
    - Robotics swarms for search and rescue
    - Traffic management and autonomous vehicle coordination
    - Data mining and machine learning optimization
    """
    
    print("SWARM INTELLIGENCE PATTERNS DEMONSTRATION")
    print("This shows how collective intelligence emerges from simple agent interactions!")
    
    await demo_ant_colony_foraging()
    await demo_particle_swarm_optimization()
    await demo_flocking_behavior()
    await demo_bee_colony_flower_search()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Simple local rules create complex global behavior")
    print("✓ Swarm systems solve problems no individual agent can solve")
    print("✓ Collective intelligence emerges without central control")
    print("✓ Self-organization adapts to changing environments")
    print("✓ Swarm algorithms find optimal solutions through exploration")

if __name__ == "__main__":
    asyncio.run(main())
