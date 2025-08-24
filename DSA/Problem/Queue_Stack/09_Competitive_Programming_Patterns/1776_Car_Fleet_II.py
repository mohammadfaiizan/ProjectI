"""
1776. Car Fleet II - Multiple Approaches
Difficulty: Hard

There are n cars traveling at different speeds in the same direction along a one-lane road. You are given an array cars where cars[i] = [positioni, speedi] represents:
- positioni is the position of the ith car (in miles)
- speedi is the speed of the ith car (in miles per hour)

For simplicity, cars can be passed through each other (i.e., they can occupy the same position at the same time).

A car catches up to another car when they occupy the same position. A car that catches up to another car will slow down to match the speed of the car it caught up to (forming a fleet).

Return an array answer where answer[i] is the time, in hours, at which the ith car catches up to the next car, or -1 if it never catches up.
"""

from typing import List

class CarFleetII:
    """Multiple approaches to solve Car Fleet II problem"""
    
    def getCollisionTimes_stack(self, cars: List[List[int]]) -> List[float]:
        """
        Approach 1: Monotonic Stack (Optimal)
        
        Use monotonic stack to efficiently calculate collision times.
        
        Time: O(n), Space: O(n)
        """
        n = len(cars)
        result = [-1.0] * n
        stack = []  # Stack of car indices
        
        # Process cars from right to left
        for i in range(n - 1, -1, -1):
            pos_i, speed_i = cars[i]
            
            # Remove cars that current car will never catch
            while stack:
                j = stack[-1]
                pos_j, speed_j = cars[j]
                
                # If car i is slower or same speed as car j, it will never catch up
                if speed_i <= speed_j:
                    stack.pop()
                    continue
                
                # Calculate collision time between car i and car j
                collision_time = (pos_j - pos_i) / (speed_i - speed_j)
                
                # If car j will collide with another car before car i catches it
                if result[j] != -1 and collision_time >= result[j]:
                    stack.pop()
                    continue
                
                # Car i will catch car j
                result[i] = collision_time
                break
            
            stack.append(i)
        
        return result
    
    def getCollisionTimes_brute_force(self, cars: List[List[int]]) -> List[float]:
        """
        Approach 2: Brute Force
        
        For each car, check all cars ahead to find the earliest collision.
        
        Time: O(n²), Space: O(1) excluding output
        """
        n = len(cars)
        result = [-1.0] * n
        
        for i in range(n - 1):
            pos_i, speed_i = cars[i]
            min_time = float('inf')
            
            for j in range(i + 1, n):
                pos_j, speed_j = cars[j]
                
                # Car i can only catch car j if it's faster
                if speed_i > speed_j:
                    collision_time = (pos_j - pos_i) / (speed_i - speed_j)
                    min_time = min(min_time, collision_time)
            
            if min_time != float('inf'):
                result[i] = min_time
        
        return result
    
    def getCollisionTimes_simulation(self, cars: List[List[int]]) -> List[float]:
        """
        Approach 3: Event Simulation
        
        Simulate the movement and collisions step by step.
        
        Time: O(n²), Space: O(n)
        """
        n = len(cars)
        result = [-1.0] * n
        
        # Create events for potential collisions
        events = []
        
        for i in range(n - 1):
            pos_i, speed_i = cars[i]
            
            for j in range(i + 1, n):
                pos_j, speed_j = cars[j]
                
                if speed_i > speed_j:
                    collision_time = (pos_j - pos_i) / (speed_i - speed_j)
                    events.append((collision_time, i, j))
        
        # Sort events by time
        events.sort()
        
        # Process events
        fleet_speed = [cars[i][1] for i in range(n)]  # Current speed of each car
        
        for time, i, j in events:
            # Check if collision is still valid
            if fleet_speed[i] > fleet_speed[j] and result[i] == -1:
                result[i] = time
                fleet_speed[i] = fleet_speed[j]  # Car i joins fleet with car j
        
        return result
    
    def getCollisionTimes_recursive(self, cars: List[List[int]]) -> List[float]:
        """
        Approach 4: Recursive Solution with Memoization
        
        Use recursion to calculate collision times.
        
        Time: O(n²), Space: O(n) due to recursion
        """
        n = len(cars)
        memo = {}
        
        def get_collision_time(i: int) -> float:
            if i in memo:
                return memo[i]
            
            if i == n - 1:
                memo[i] = -1.0
                return -1.0
            
            pos_i, speed_i = cars[i]
            min_time = float('inf')
            
            for j in range(i + 1, n):
                pos_j, speed_j = cars[j]
                
                if speed_i > speed_j:
                    collision_time = (pos_j - pos_i) / (speed_i - speed_j)
                    
                    # Check if car j will collide with another car first
                    j_collision = get_collision_time(j)
                    
                    if j_collision == -1 or collision_time <= j_collision:
                        min_time = min(min_time, collision_time)
            
            result = min_time if min_time != float('inf') else -1.0
            memo[i] = result
            return result
        
        return [get_collision_time(i) for i in range(n)]


def test_car_fleet_ii():
    """Test Car Fleet II algorithms"""
    solver = CarFleetII()
    
    test_cases = [
        ([[1,2],[2,1],[4,3],[7,2]], [1.0, -1.0, 3.0, -1.0], "Example 1"),
        ([[3,4],[5,4],[6,3],[9,1]], [2.0, 1.0, 1.5, -1.0], "Example 2"),
        ([[1,3],[2,2]], [1.0, -1.0], "Two cars"),
        ([[1,1],[2,2]], [-1.0, -1.0], "Slower car ahead"),
        ([[1,4],[2,3],[3,2],[4,1]], [0.5, 1.0, 3.0, -1.0], "Multiple collisions"),
        ([[0,1]], [-1.0], "Single car"),
        ([[1,2],[3,1],[5,3]], [2.0, -1.0, -1.0], "Mixed speeds"),
        ([[1,5],[2,4],[3,3],[4,2],[5,1]], [0.25, 0.5, 1.0, 4.0, -1.0], "Sequential"),
    ]
    
    algorithms = [
        ("Stack", solver.getCollisionTimes_stack),
        ("Brute Force", solver.getCollisionTimes_brute_force),
        ("Simulation", solver.getCollisionTimes_simulation),
        ("Recursive", solver.getCollisionTimes_recursive),
    ]
    
    print("=== Testing Car Fleet II ===")
    
    def are_close(a: List[float], b: List[float], tolerance: float = 1e-9) -> bool:
        """Check if two float arrays are approximately equal"""
        if len(a) != len(b):
            return False
        
        for x, y in zip(a, b):
            if x == -1.0 and y == -1.0:
                continue
            if x == -1.0 or y == -1.0:
                return False
            if abs(x - y) > tolerance:
                return False
        
        return True
    
    for cars, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Cars: {cars}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func([car[:] for car in cars])  # Deep copy
                status = "✓" if are_close(result, expected) else "✗"
                print(f"{alg_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    cars = [[1,2],[2,1],[4,3],[7,2]]
    
    print(f"Cars: {cars}")
    print("Format: [position, speed]")
    print("Strategy: Process cars from right to left using monotonic stack")
    
    n = len(cars)
    result = [-1.0] * n
    stack = []
    
    print(f"\nStep-by-step processing (right to left):")
    
    for i in range(n - 1, -1, -1):
        pos_i, speed_i = cars[i]
        
        print(f"\nStep {n - i}: Processing car {i} at position {pos_i}, speed {speed_i}")
        print(f"  Current stack: {stack}")
        print(f"  Current result: {result}")
        
        removed_cars = []
        
        while stack:
            j = stack[-1]
            pos_j, speed_j = cars[j]
            
            print(f"    Checking against car {j} (pos={pos_j}, speed={speed_j})")
            
            # If car i is slower or same speed as car j
            if speed_i <= speed_j:
                removed = stack.pop()
                removed_cars.append(removed)
                print(f"      Car {i} can't catch car {j} (speed {speed_i} <= {speed_j})")
                continue
            
            # Calculate collision time
            collision_time = (pos_j - pos_i) / (speed_i - speed_j)
            print(f"      Collision time with car {j}: ({pos_j} - {pos_i}) / ({speed_i} - {speed_j}) = {collision_time}")
            
            # Check if car j will collide with another car first
            if result[j] != -1 and collision_time >= result[j]:
                removed = stack.pop()
                removed_cars.append(removed)
                print(f"      Car {j} will collide earlier (at time {result[j]})")
                continue
            
            # Car i will catch car j
            result[i] = collision_time
            print(f"      Car {i} will catch car {j} at time {collision_time}")
            break
        
        if removed_cars:
            print(f"    Removed cars from stack: {removed_cars}")
        
        stack.append(i)
        print(f"  Added car {i} to stack: {stack}")
        print(f"  Updated result: {result}")
    
    print(f"\nFinal result: {result}")


def visualize_car_movement():
    """Visualize car movement and collisions"""
    print("\n=== Car Movement Visualization ===")
    
    cars = [[1,2],[2,1],[4,3],[7,2]]
    solver = CarFleetII()
    collision_times = solver.getCollisionTimes_stack(cars)
    
    print(f"Cars: {cars}")
    print(f"Collision times: {collision_times}")
    
    # Show positions at different time points
    time_points = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    print(f"\nCar positions over time:")
    print(f"{'Time':<6} | {'Car 0':<8} | {'Car 1':<8} | {'Car 2':<8} | {'Car 3':<8}")
    print("-" * 50)
    
    for t in time_points:
        positions = []
        
        for i, (pos, speed) in enumerate(cars):
            # Calculate position at time t, considering collisions
            if collision_times[i] != -1 and t >= collision_times[i]:
                # Car has collided, find its fleet speed
                current_pos = pos + speed * collision_times[i]
                
                # Find the car it collided with
                for j in range(i + 1, len(cars)):
                    pos_j, speed_j = cars[j]
                    if abs((pos_j - pos) / (speed - speed_j) - collision_times[i]) < 1e-9:
                        # This is the car it collided with
                        remaining_time = t - collision_times[i]
                        current_pos += speed_j * remaining_time
                        break
                
                positions.append(f"{current_pos:.1f}")
            else:
                current_pos = pos + speed * t
                positions.append(f"{current_pos:.1f}")
        
        print(f"{t:<6} | {positions[0]:<8} | {positions[1]:<8} | {positions[2]:<8} | {positions[3]:<8}")


def demonstrate_competitive_programming_patterns():
    """Demonstrate competitive programming patterns"""
    print("\n=== Competitive Programming Patterns ===")
    
    solver = CarFleetII()
    
    # Pattern 1: Monotonic stack for optimization
    print("1. Monotonic Stack Optimization:")
    print("   Process elements in reverse order")
    print("   Maintain stack of potentially relevant elements")
    
    example1 = [[1,2],[2,1],[4,3],[7,2]]
    result1 = solver.getCollisionTimes_stack(example1)
    print(f"   {example1} -> {result1}")
    
    # Pattern 2: Collision detection
    print(f"\n2. Collision Detection:")
    print("   Calculate intersection time of two moving objects")
    print("   Formula: time = (pos_diff) / (speed_diff)")
    
    # Pattern 3: Event processing
    print(f"\n3. Event Processing:")
    print("   Handle events in chronological order")
    print("   Update state based on event outcomes")
    
    # Pattern 4: Greedy elimination
    print(f"\n4. Greedy Elimination:")
    print("   Remove elements that can never be optimal")
    print("   Maintain only relevant candidates")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack", "O(n)", "O(n)", "Each car pushed/popped once"),
        ("Brute Force", "O(n²)", "O(1)", "Check all pairs of cars"),
        ("Simulation", "O(n²)", "O(n)", "Process all collision events"),
        ("Recursive", "O(n²)", "O(n)", "Recursive calls with memoization"),
    ]
    
    print(f"{'Approach':<15} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 55)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<15} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nMonotonic Stack approach is optimal for competitive programming")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = CarFleetII()
    
    edge_cases = [
        ([[0,1]], [-1.0], "Single car"),
        ([[1,1],[2,1]], [-1.0, -1.0], "Same speeds"),
        ([[1,1],[2,2]], [-1.0, -1.0], "Slower car behind"),
        ([[2,1],[1,2]], [1.0, -1.0], "Faster car behind"),
        ([[1,3],[2,2],[3,1]], [0.5, 1.0, -1.0], "Sequential catches"),
        ([[1,1],[3,2],[5,3]], [-1.0, -1.0, -1.0], "All different speeds"),
        ([[0,2],[1,1]], [1.0, -1.0], "Adjacent cars"),
        ([[1,10],[2,1]], [0.1, -1.0], "Large speed difference"),
        ([[1,1],[1,2]], [-1.0, -1.0], "Same position"),
    ]
    
    for cars, expected, description in edge_cases:
        try:
            result = solver.getCollisionTimes_stack([car[:] for car in cars])
            
            # Check if results are approximately equal
            def are_close(a, b, tol=1e-9):
                if len(a) != len(b):
                    return False
                for x, y in zip(a, b):
                    if x == -1.0 and y == -1.0:
                        continue
                    if x == -1.0 or y == -1.0:
                        return False
                    if abs(x - y) > tol:
                        return False
                return True
            
            status = "✓" if are_close(result, expected) else "✗"
            print(f"{description:25} | {status} | {cars} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solver = CarFleetII()
    
    # Application 1: Traffic flow analysis
    print("1. Traffic Flow Analysis:")
    print("   Predict when vehicles will form convoys")
    print("   Optimize traffic light timing")
    
    traffic_data = [[0,60],[50,45],[100,70],[150,50]]  # [position_km, speed_kmh]
    collision_times = solver.getCollisionTimes_stack(traffic_data)
    
    print(f"   Vehicle data (position km, speed km/h): {traffic_data}")
    print(f"   Convoy formation times (hours): {collision_times}")
    
    # Application 2: Autonomous vehicle coordination
    print(f"\n2. Autonomous Vehicle Coordination:")
    print("   Predict vehicle interactions for path planning")
    print("   Coordinate merging and lane changes")
    
    av_data = [[10,30],[20,25],[30,35],[40,20]]  # Autonomous vehicles
    av_times = solver.getCollisionTimes_stack(av_data)
    
    print(f"   AV positions/speeds: {av_data}")
    print(f"   Interaction times: {av_times}")
    
    # Application 3: Supply chain logistics
    print(f"\n3. Supply Chain Logistics:")
    print("   Predict when shipments will catch up")
    print("   Optimize delivery schedules")
    
    shipment_data = [[0,50],[100,40],[200,60],[300,35]]  # [distance, speed]
    shipment_times = solver.getCollisionTimes_stack(shipment_data)
    
    print(f"   Shipment data: {shipment_data}")
    print(f"   Catch-up times: {shipment_times}")


if __name__ == "__main__":
    test_car_fleet_ii()
    demonstrate_stack_approach()
    visualize_car_movement()
    demonstrate_competitive_programming_patterns()
    analyze_time_complexity()
    test_edge_cases()
    demonstrate_real_world_applications()

"""
Car Fleet II demonstrates advanced competitive programming patterns
with monotonic stack optimization, collision detection, and event
processing for complex dynamic system simulations.
"""
