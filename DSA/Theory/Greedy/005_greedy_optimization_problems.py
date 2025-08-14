"""
Greedy Optimization Problems - Advanced Applications
===================================================

Topics: Resource allocation, load balancing, scheduling optimization
Companies: Google, Amazon, Microsoft, Facebook, Netflix, Uber, Airbnb
Difficulty: Medium to Hard
Time Complexity: O(n log n) to O(n²) depending on problem
Space Complexity: O(n) for most problems
"""

from typing import List, Tuple, Optional, Dict, Any, Set
import heapq
from collections import defaultdict, Counter
import math

class GreedyOptimizationProblems:
    
    def __init__(self):
        """Initialize with optimization tracking"""
        self.solution_steps = []
        self.optimization_stats = {}
    
    # ==========================================
    # 1. RESOURCE ALLOCATION PROBLEMS
    # ==========================================
    
    def fractional_knapsack_advanced(self, items: List[Tuple[str, float, float, int]], capacity: float) -> Tuple[Dict[str, float], float]:
        """
        Advanced Fractional Knapsack with Item Quantities
        
        Company: Amazon, Google (resource allocation)
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Problem: Maximize value with limited capacity and item quantities
        Greedy Strategy: Sort by value/weight ratio, take highest ratio items first
        
        Args:
            items: List of (name, value_per_unit, weight_per_unit, max_quantity)
            capacity: Total capacity available
        """
        print("=== ADVANCED FRACTIONAL KNAPSACK ===")
        print("Problem: Maximize value with capacity and quantity constraints")
        print("Greedy Strategy: Highest value/weight ratio first")
        print()
        
        print(f"Knapsack capacity: {capacity}")
        print("Items (name, value_per_unit, weight_per_unit, max_quantity, ratio):")
        
        # Calculate ratios and expand items
        expanded_items = []
        for name, value_per_unit, weight_per_unit, max_quantity in items:
            ratio = value_per_unit / weight_per_unit if weight_per_unit > 0 else float('inf')
            expanded_items.append((name, value_per_unit, weight_per_unit, max_quantity, ratio))
            print(f"   {name}: value={value_per_unit}, weight={weight_per_unit}, qty={max_quantity}, ratio={ratio:.3f}")
        print()
        
        # Sort by value/weight ratio
        expanded_items.sort(key=lambda x: x[4], reverse=True)
        
        print("Items sorted by ratio (greedy order):")
        for name, value_per_unit, weight_per_unit, max_quantity, ratio in expanded_items:
            print(f"   {name}: ratio={ratio:.3f}")
        print()
        
        # Greedy selection
        selected = {}
        total_value = 0.0
        remaining_capacity = capacity
        
        print("Greedy selection process:")
        for i, (name, value_per_unit, weight_per_unit, max_quantity, ratio) in enumerate(expanded_items):
            print(f"Step {i+1}: Consider item {name}")
            print(f"   Available: {max_quantity} units, each worth {value_per_unit}, weighing {weight_per_unit}")
            print(f"   Remaining capacity: {remaining_capacity}")
            
            if remaining_capacity <= 0:
                print(f"   ✗ No remaining capacity")
                break
            
            # Calculate how many units we can take
            max_units_by_weight = remaining_capacity / weight_per_unit
            max_units_by_quantity = max_quantity
            units_to_take = min(max_units_by_weight, max_units_by_quantity)
            
            if units_to_take > 0:
                selected[name] = units_to_take
                value_added = units_to_take * value_per_unit
                weight_added = units_to_take * weight_per_unit
                
                total_value += value_added
                remaining_capacity -= weight_added
                
                print(f"   ✓ Take {units_to_take:.2f} units")
                print(f"   Value added: {value_added:.2f}")
                print(f"   Weight added: {weight_added:.2f}")
                print(f"   Total value: {total_value:.2f}")
                print(f"   Remaining capacity: {remaining_capacity:.2f}")
            else:
                print(f"   ✗ Cannot take any units (weight constraint)")
            
            print()
        
        print("Final solution:")
        total_weight = 0
        for name, quantity in selected.items():
            # Find original item data
            original_item = next(item for item in expanded_items if item[0] == name)
            value_per_unit, weight_per_unit = original_item[1], original_item[2]
            
            item_value = quantity * value_per_unit
            item_weight = quantity * weight_per_unit
            total_weight += item_weight
            
            print(f"   {name}: {quantity:.2f} units, value={item_value:.2f}, weight={item_weight:.2f}")
        
        print(f"\nSummary:")
        print(f"   Total value: {total_value:.2f}")
        print(f"   Total weight: {total_weight:.2f}")
        print(f"   Capacity utilization: {(total_weight/capacity):.1%}")
        
        return selected, total_value
    
    def server_load_balancing(self, servers: List[Tuple[str, int]], tasks: List[Tuple[str, int]]) -> Dict[str, List[str]]:
        """
        Server Load Balancing using Greedy Assignment
        
        Company: Google, Amazon, Microsoft (cloud computing)
        Difficulty: Medium
        Time: O(m log n), Space: O(n + m) where n=servers, m=tasks
        
        Problem: Distribute tasks to minimize maximum server load
        Greedy Strategy: Always assign task to least loaded server
        """
        print("=== SERVER LOAD BALANCING ===")
        print("Problem: Distribute tasks to minimize maximum server load")
        print("Greedy Strategy: Always assign to least loaded server")
        print()
        
        print("Initial server capacities:")
        for name, capacity in servers:
            print(f"   {name}: capacity {capacity}")
        print()
        
        print("Tasks to assign:")
        for name, load in tasks:
            print(f"   {name}: load {load}")
        print()
        
        # Initialize server loads using min heap
        server_loads = [(0, name, capacity) for name, capacity in servers]
        heapq.heapify(server_loads)
        
        assignments = {name: [] for name, _ in servers}
        
        print("Load balancing process:")
        for i, (task_name, task_load) in enumerate(tasks):
            print(f"Step {i+1}: Assign task {task_name} (load: {task_load})")
            
            # Get server with minimum current load
            current_load, server_name, server_capacity = heapq.heappop(server_loads)
            
            print(f"   Least loaded server: {server_name} (current load: {current_load})")
            
            # Check if server can handle the task
            if current_load + task_load <= server_capacity:
                # Assign task to this server
                assignments[server_name].append(task_name)
                new_load = current_load + task_load
                
                print(f"   ✓ Assigned to {server_name}")
                print(f"   New load: {current_load} + {task_load} = {new_load}")
                
                # Put server back with updated load
                heapq.heappush(server_loads, (new_load, server_name, server_capacity))
            else:
                print(f"   ✗ Cannot assign: would exceed capacity ({current_load + task_load} > {server_capacity})")
                # Put server back unchanged
                heapq.heappush(server_loads, (current_load, server_name, server_capacity))
                
                # Try to find another server (simple fallback)
                assigned = False
                temp_servers = []
                while server_loads and not assigned:
                    load, name, cap = heapq.heappop(server_loads)
                    if load + task_load <= cap:
                        assignments[name].append(task_name)
                        heapq.heappush(server_loads, (load + task_load, name, cap))
                        print(f"   ✓ Assigned to {name} instead (load: {load} + {task_load} = {load + task_load})")
                        assigned = True
                    else:
                        temp_servers.append((load, name, cap))
                
                # Put back servers we couldn't use
                for server in temp_servers:
                    heapq.heappush(server_loads, server)
                
                if not assigned:
                    print(f"   ✗ Could not assign task {task_name} to any server!")
            
            # Show current loads
            current_loads = sorted([(load, name) for load, name, _ in server_loads])
            print(f"   Current server loads: {current_loads}")
            print()
        
        print("Final assignment:")
        max_load = 0
        for server_name, assigned_tasks in assignments.items():
            total_load = sum(load for task_name, load in tasks if task_name in assigned_tasks)
            max_load = max(max_load, total_load)
            print(f"   {server_name}: {assigned_tasks} (total load: {total_load})")
        
        print(f"\nMaximum server load: {max_load}")
        
        return assignments
    
    # ==========================================
    # 2. SCHEDULING OPTIMIZATION
    # ==========================================
    
    def minimize_weighted_completion_time(self, jobs: List[Tuple[str, int, int]]) -> Tuple[List[str], float]:
        """
        Minimize Weighted Completion Time
        
        Company: Google, Microsoft (scheduling optimization)
        Difficulty: Medium to Hard
        Time: O(n log n), Space: O(n)
        
        Problem: Schedule jobs to minimize total weighted completion time
        Greedy Strategy: Sort by weight/processing_time ratio (Smith's rule)
        
        Args:
            jobs: List of (job_name, processing_time, weight)
        """
        print("=== MINIMIZE WEIGHTED COMPLETION TIME ===")
        print("Problem: Schedule jobs to minimize total weighted completion time")
        print("Greedy Strategy: Sort by weight/processing_time ratio (Smith's rule)")
        print()
        
        print("Jobs (name, processing_time, weight, ratio):")
        jobs_with_ratio = []
        for name, proc_time, weight in jobs:
            ratio = weight / proc_time if proc_time > 0 else float('inf')
            jobs_with_ratio.append((name, proc_time, weight, ratio))
            print(f"   {name}: time={proc_time}, weight={weight}, ratio={ratio:.3f}")
        print()
        
        # Sort by weight/processing_time ratio (Smith's rule)
        jobs_with_ratio.sort(key=lambda x: x[3], reverse=True)
        
        print("Jobs sorted by weight/time ratio (optimal order):")
        for name, proc_time, weight, ratio in jobs_with_ratio:
            print(f"   {name}: ratio={ratio:.3f}")
        print()
        
        # Calculate schedule and weighted completion times
        current_time = 0
        schedule = []
        total_weighted_completion_time = 0
        
        print("Scheduling process:")
        for i, (name, proc_time, weight, ratio) in enumerate(jobs_with_ratio):
            start_time = current_time
            completion_time = current_time + proc_time
            weighted_completion_time = weight * completion_time
            
            schedule.append(name)
            total_weighted_completion_time += weighted_completion_time
            current_time = completion_time
            
            print(f"Job {i+1}: {name}")
            print(f"   Schedule: {start_time} → {completion_time}")
            print(f"   Weighted completion time: {weight} × {completion_time} = {weighted_completion_time}")
            print(f"   Total weighted completion time: {total_weighted_completion_time}")
            print()
        
        print("Final schedule:")
        print(f"   Job order: {schedule}")
        print(f"   Total weighted completion time: {total_weighted_completion_time}")
        
        return schedule, total_weighted_completion_time
    
    def energy_efficient_scheduling(self, tasks: List[Tuple[str, int, int]], max_power: int) -> Tuple[List[Tuple[str, int, int]], int]:
        """
        Energy-Efficient Task Scheduling
        
        Company: Google, Microsoft (data centers), Tesla
        Difficulty: Hard
        Time: O(n log n), Space: O(n)
        
        Problem: Schedule tasks within power budget to maximize throughput
        Greedy Strategy: Sort by performance/power ratio
        
        Args:
            tasks: List of (task_name, execution_time, power_required)
            max_power: Maximum power budget
        """
        print("=== ENERGY-EFFICIENT SCHEDULING ===")
        print("Problem: Schedule tasks within power budget")
        print("Greedy Strategy: Maximize performance/power ratio")
        print()
        
        print(f"Maximum power budget: {max_power}")
        print("Tasks (name, execution_time, power_required, efficiency):")
        
        # Calculate efficiency ratio (inverse of time per unit power)
        tasks_with_efficiency = []
        for name, exec_time, power in tasks:
            efficiency = 1.0 / (exec_time * power) if exec_time > 0 and power > 0 else 0
            tasks_with_efficiency.append((name, exec_time, power, efficiency))
            print(f"   {name}: time={exec_time}, power={power}, efficiency={efficiency:.4f}")
        print()
        
        # Sort by efficiency (performance per unit power per unit time)
        tasks_with_efficiency.sort(key=lambda x: x[3], reverse=True)
        
        print("Tasks sorted by efficiency:")
        for name, exec_time, power, efficiency in tasks_with_efficiency:
            print(f"   {name}: efficiency={efficiency:.4f}")
        print()
        
        # Greedy scheduling within power budget
        schedule = []
        current_time = 0
        total_power_used = 0
        
        print("Scheduling process:")
        for i, (name, exec_time, power, efficiency) in enumerate(tasks_with_efficiency):
            print(f"Step {i+1}: Consider task {name}")
            print(f"   Execution time: {exec_time}, Power required: {power}")
            print(f"   Current power usage: {total_power_used}")
            
            if total_power_used + power <= max_power:
                # Schedule the task
                start_time = current_time
                end_time = current_time + exec_time
                
                schedule.append((name, start_time, end_time))
                total_power_used += power
                current_time = end_time
                
                print(f"   ✓ Scheduled {name}: {start_time} → {end_time}")
                print(f"   Power usage: {total_power_used}/{max_power}")
            else:
                print(f"   ✗ Cannot schedule: would exceed power budget ({total_power_used + power} > {max_power})")
            print()
        
        print("Final schedule:")
        for name, start, end in schedule:
            print(f"   {name}: {start} → {end} (duration: {end - start})")
        
        print(f"\nSummary:")
        print(f"   Tasks scheduled: {len(schedule)}")
        print(f"   Total execution time: {current_time}")
        print(f"   Power utilization: {total_power_used}/{max_power} ({total_power_used/max_power:.1%})")
        
        return schedule, total_power_used
    
    # ==========================================
    # 3. NETWORK OPTIMIZATION
    # ==========================================
    
    def bandwidth_allocation(self, connections: List[Tuple[str, str, int]], total_bandwidth: int) -> Dict[Tuple[str, str], int]:
        """
        Bandwidth Allocation using Greedy Fair Share
        
        Company: Google, Amazon (network optimization), Netflix
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Problem: Allocate bandwidth fairly among connections
        Greedy Strategy: Allocate equally, then distribute excess by priority
        """
        print("=== BANDWIDTH ALLOCATION ===")
        print("Problem: Allocate bandwidth fairly among connections")
        print("Greedy Strategy: Equal allocation + priority-based excess distribution")
        print()
        
        print(f"Total available bandwidth: {total_bandwidth}")
        print("Connection requests (source, destination, requested_bandwidth):")
        for src, dst, requested in connections:
            print(f"   {src} → {dst}: {requested}")
        print()
        
        # Calculate fair share
        num_connections = len(connections)
        fair_share = total_bandwidth // num_connections if num_connections > 0 else 0
        remaining_bandwidth = total_bandwidth
        
        print(f"Fair share per connection: {fair_share}")
        print()
        
        # Phase 1: Allocate fair share or requested amount (whichever is smaller)
        allocation = {}
        satisfied_connections = []
        unsatisfied_connections = []
        
        print("Phase 1: Initial fair allocation")
        for src, dst, requested in connections:
            connection = (src, dst)
            initial_allocation = min(fair_share, requested)
            
            allocation[connection] = initial_allocation
            remaining_bandwidth -= initial_allocation
            
            print(f"   {src} → {dst}: allocated {initial_allocation} (requested {requested})")
            
            if initial_allocation == requested:
                satisfied_connections.append((src, dst, requested))
            else:
                unsatisfied_connections.append((src, dst, requested))
        
        print(f"   Remaining bandwidth after fair allocation: {remaining_bandwidth}")
        print()
        
        # Phase 2: Distribute remaining bandwidth to unsatisfied connections
        if remaining_bandwidth > 0 and unsatisfied_connections:
            print("Phase 2: Distribute excess bandwidth")
            
            # Sort unsatisfied connections by priority (requested bandwidth)
            unsatisfied_connections.sort(key=lambda x: x[2], reverse=True)
            
            print("Unsatisfied connections by priority:")
            for src, dst, requested in unsatisfied_connections:
                current_allocation = allocation[(src, dst)]
                deficit = requested - current_allocation
                print(f"   {src} → {dst}: current={current_allocation}, requested={requested}, deficit={deficit}")
            print()
            
            # Distribute remaining bandwidth proportionally
            total_deficit = sum(requested - allocation[(src, dst)] 
                              for src, dst, requested in unsatisfied_connections)
            
            print("Proportional distribution of excess bandwidth:")
            for src, dst, requested in unsatisfied_connections:
                connection = (src, dst)
                current_allocation = allocation[connection]
                deficit = requested - current_allocation
                
                if total_deficit > 0:
                    additional = min(deficit, (deficit / total_deficit) * remaining_bandwidth)
                    additional = int(additional)  # Round down for simplicity
                    
                    allocation[connection] += additional
                    remaining_bandwidth -= additional
                    
                    print(f"   {src} → {dst}: +{additional} → {allocation[connection]}")
        
        print(f"\nFinal bandwidth allocation:")
        total_allocated = 0
        for (src, dst), allocated in allocation.items():
            total_allocated += allocated
            print(f"   {src} → {dst}: {allocated}")
        
        print(f"\nSummary:")
        print(f"   Total allocated: {total_allocated}")
        print(f"   Unused bandwidth: {total_bandwidth - total_allocated}")
        print(f"   Utilization: {total_allocated/total_bandwidth:.1%}")
        
        return allocation
    
    def content_delivery_optimization(self, requests: List[Tuple[str, str, int]], servers: Dict[str, int]) -> Dict[str, List[Tuple[str, int]]]:
        """
        Content Delivery Network Optimization
        
        Company: Netflix, Amazon CloudFront, Google CDN
        Difficulty: Hard
        Time: O(n log m), Space: O(n + m) where n=requests, m=servers
        
        Problem: Route content requests to minimize latency and balance load
        Greedy Strategy: Route to closest server with available capacity
        """
        print("=== CONTENT DELIVERY OPTIMIZATION ===")
        print("Problem: Route requests to minimize latency and balance load")
        print("Greedy Strategy: Route to closest available server")
        print()
        
        # Simplified distance function (in practice, use real geographical distances)
        def calculate_distance(client_region: str, server_region: str) -> int:
            distance_matrix = {
                ('US', 'US'): 10, ('US', 'EU'): 100, ('US', 'ASIA'): 150,
                ('EU', 'US'): 100, ('EU', 'EU'): 10, ('EU', 'ASIA'): 120,
                ('ASIA', 'US'): 150, ('ASIA', 'EU'): 120, ('ASIA', 'ASIA'): 10
            }
            return distance_matrix.get((client_region, server_region), 200)
        
        print("Available servers (region: capacity):")
        for region, capacity in servers.items():
            print(f"   {region}: {capacity}")
        print()
        
        print("Incoming requests (client_region, content_id, bandwidth_needed):")
        for client, content, bandwidth in requests:
            print(f"   {client} requests {content}: {bandwidth} units")
        print()
        
        # Initialize server loads and assignments
        server_loads = {region: 0 for region in servers}
        assignments = {region: [] for region in servers}
        
        print("Request routing process:")
        for i, (client_region, content_id, bandwidth) in enumerate(requests):
            print(f"Request {i+1}: {client_region} → {content_id} ({bandwidth} units)")
            
            # Find best server (minimize distance, then load)
            best_server = None
            best_score = float('inf')
            
            for server_region, capacity in servers.items():
                current_load = server_loads[server_region]
                
                if current_load + bandwidth <= capacity:
                    distance = calculate_distance(client_region, server_region)
                    load_factor = current_load / capacity
                    # Score combines distance and load balancing
                    score = distance * (1 + load_factor)
                    
                    print(f"   {server_region}: distance={distance}, load={current_load}/{capacity}, score={score:.2f}")
                    
                    if score < best_score:
                        best_score = score
                        best_server = server_region
                else:
                    print(f"   {server_region}: insufficient capacity ({current_load + bandwidth} > {capacity})")
            
            if best_server:
                # Route request to best server
                assignments[best_server].append((content_id, bandwidth))
                server_loads[best_server] += bandwidth
                
                distance = calculate_distance(client_region, best_server)
                print(f"   ✓ Routed to {best_server} (distance: {distance})")
                print(f"   Server load: {server_loads[best_server]}/{servers[best_server]}")
            else:
                print(f"   ✗ No server available with sufficient capacity")
            print()
        
        print("Final server assignments:")
        total_requests = 0
        total_distance = 0
        for region, assignments_list in assignments.items():
            load = server_loads[region]
            capacity = servers[region]
            utilization = load / capacity if capacity > 0 else 0
            
            print(f"   {region}: {len(assignments_list)} requests, load={load}/{capacity} ({utilization:.1%})")
            for content, bandwidth in assignments_list:
                print(f"      {content}: {bandwidth} units")
            
            total_requests += len(assignments_list)
        
        print(f"\nSummary:")
        print(f"   Total requests routed: {total_requests}")
        print(f"   Failed requests: {len(requests) - total_requests}")
        
        return assignments


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_greedy_optimization_problems():
    """Demonstrate all greedy optimization problems"""
    print("=== GREEDY OPTIMIZATION PROBLEMS DEMONSTRATION ===\n")
    
    optimizer = GreedyOptimizationProblems()
    
    # 1. Resource Allocation
    print("1. RESOURCE ALLOCATION PROBLEMS")
    
    print("a) Advanced Fractional Knapsack:")
    items = [
        ("Gold", 100, 1, 5),     # (name, value_per_unit, weight_per_unit, max_quantity)
        ("Silver", 60, 2, 10),
        ("Bronze", 40, 3, 8),
        ("Platinum", 150, 1, 2)
    ]
    optimizer.fractional_knapsack_advanced(items, 15)
    print("\n" + "-"*40 + "\n")
    
    print("b) Server Load Balancing:")
    servers = [("Server1", 100), ("Server2", 80), ("Server3", 120)]
    tasks = [("Task1", 30), ("Task2", 45), ("Task3", 25), ("Task4", 60), ("Task5", 35)]
    optimizer.server_load_balancing(servers, tasks)
    print("\n" + "="*60 + "\n")
    
    # 2. Scheduling Optimization
    print("2. SCHEDULING OPTIMIZATION")
    
    print("a) Minimize Weighted Completion Time:")
    jobs = [("Job1", 3, 20), ("Job2", 1, 10), ("Job3", 4, 30), ("Job4", 2, 15)]
    optimizer.minimize_weighted_completion_time(jobs)
    print("\n" + "-"*40 + "\n")
    
    print("b) Energy-Efficient Scheduling:")
    tasks = [
        ("Compute1", 2, 50),     # (name, execution_time, power_required)
        ("Compute2", 1, 30),
        ("Compute3", 3, 40),
        ("Compute4", 2, 60),
        ("Compute5", 1, 25)
    ]
    optimizer.energy_efficient_scheduling(tasks, 100)
    print("\n" + "="*60 + "\n")
    
    # 3. Network Optimization
    print("3. NETWORK OPTIMIZATION")
    
    print("a) Bandwidth Allocation:")
    connections = [
        ("ClientA", "ServerX", 50),
        ("ClientB", "ServerY", 30),
        ("ClientC", "ServerZ", 70),
        ("ClientD", "ServerX", 40)
    ]
    optimizer.bandwidth_allocation(connections, 150)
    print("\n" + "-"*40 + "\n")
    
    print("b) Content Delivery Optimization:")
    requests = [
        ("US", "video1.mp4", 20),
        ("EU", "video2.mp4", 15),
        ("ASIA", "video1.mp4", 25),
        ("US", "video3.mp4", 30),
        ("EU", "video1.mp4", 10)
    ]
    servers = {"US": 50, "EU": 40, "ASIA": 60}
    optimizer.content_delivery_optimization(requests, servers)


if __name__ == "__main__":
    demonstrate_greedy_optimization_problems()
    
    print("\n=== OPTIMIZATION PROBLEMS MASTERY GUIDE ===")
    
    print("\n🎯 OPTIMIZATION CATEGORIES:")
    print("• Resource Allocation: Maximize utility with constraints")
    print("• Load Balancing: Distribute work evenly across resources")
    print("• Scheduling: Optimize task ordering and timing")
    print("• Network Optimization: Minimize latency and maximize throughput")
    print("• Energy Efficiency: Balance performance with power consumption")
    
    print("\n📊 COMPLEXITY PATTERNS:")
    print("• Most problems: O(n log n) due to sorting requirements")
    print("• Load balancing: O(m log n) for m tasks, n servers")
    print("• Network routing: O(requests × servers) for distance calculations")
    print("• Scheduling: O(n log n) for priority-based ordering")
    
    print("\n⚡ KEY STRATEGIES:")
    print("• Ratio-based optimization (value/cost, performance/power)")
    print("• Heap-based selection for dynamic optimization")
    print("• Multi-criteria optimization with weighted scoring")
    print("• Capacity-aware allocation algorithms")
    print("• Fair share + excess distribution patterns")
    
    print("\n🔧 IMPLEMENTATION APPROACHES:")
    print("• Sort by optimization criteria before greedy selection")
    print("• Use priority queues for dynamic resource selection")
    print("• Implement capacity checking and constraint validation")
    print("• Consider multi-phase allocation strategies")
    print("• Add fallback mechanisms for constraint violations")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• Cloud Computing: VM allocation, autoscaling")
    print("• Content Delivery: CDN routing, caching strategies")
    print("• Manufacturing: Production scheduling, resource planning")
    print("• Finance: Portfolio optimization, risk management")
    print("• Transportation: Route optimization, fleet management")
    print("• Energy: Smart grid management, power distribution")
    
    print("\n🎓 ADVANCED CONSIDERATIONS:")
    print("• Online vs offline optimization algorithms")
    print("• Approximation ratios for NP-hard problems")
    print("• Competitive analysis for online algorithms")
    print("• Multi-objective optimization trade-offs")
    print("• Robustness and adaptivity in dynamic environments")
    
    print("\n💡 OPTIMIZATION PRINCIPLES:")
    print("• Balance multiple objectives (performance, cost, fairness)")
    print("• Consider both immediate and long-term consequences")
    print("• Design for scalability and real-time requirements")
    print("• Implement monitoring and adaptive re-optimization")
    print("• Account for uncertainty and dynamic conditions")
