"""
735. Asteroid Collision - Multiple Approaches
Difficulty: Medium

We are given an array asteroids of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.
"""

from typing import List

class AsteroidCollision:
    """Multiple approaches to simulate asteroid collisions"""
    
    def asteroidCollision_stack(self, asteroids: List[int]) -> List[int]:
        """
        Approach 1: Stack Simulation (Optimal)
        
        Use stack to simulate collisions as they happen.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        
        for asteroid in asteroids:
            # Process current asteroid
            while stack and asteroid < 0 < stack[-1]:
                # Collision: right-moving (positive) meets left-moving (negative)
                if stack[-1] < -asteroid:
                    # Right-moving asteroid explodes
                    stack.pop()
                    continue
                elif stack[-1] == -asteroid:
                    # Both explode
                    stack.pop()
                
                # Left-moving asteroid explodes (or both exploded)
                break
            else:
                # No collision or asteroid survived
                stack.append(asteroid)
        
        return stack
    
    def asteroidCollision_simulation(self, asteroids: List[int]) -> List[int]:
        """
        Approach 2: Direct Simulation
        
        Simulate collisions by repeatedly scanning the array.
        
        Time: O(n²) worst case, Space: O(n)
        """
        result = asteroids[:]
        
        while True:
            collision_occurred = False
            new_result = []
            i = 0
            
            while i < len(result):
                if (i + 1 < len(result) and 
                    result[i] > 0 and result[i + 1] < 0):
                    # Collision detected
                    collision_occurred = True
                    left_size = result[i]
                    right_size = -result[i + 1]
                    
                    if left_size > right_size:
                        # Left asteroid survives
                        new_result.append(result[i])
                    elif left_size < right_size:
                        # Right asteroid survives
                        new_result.append(result[i + 1])
                    # If equal, both explode (add nothing)
                    
                    i += 2  # Skip both asteroids
                else:
                    # No collision
                    new_result.append(result[i])
                    i += 1
            
            result = new_result
            
            if not collision_occurred:
                break
        
        return result
    
    def asteroidCollision_recursive(self, asteroids: List[int]) -> List[int]:
        """
        Approach 3: Recursive Solution
        
        Use recursion to handle collisions.
        
        Time: O(n), Space: O(n) due to recursion
        """
        def process_asteroid(stack: List[int], asteroid: int) -> List[int]:
            if not stack or asteroid > 0 or stack[-1] < 0:
                # No collision possible
                return stack + [asteroid]
            
            # Collision: stack[-1] > 0 and asteroid < 0
            if stack[-1] < -asteroid:
                # Right-moving asteroid explodes, continue with left-moving
                return process_asteroid(stack[:-1], asteroid)
            elif stack[-1] == -asteroid:
                # Both explode
                return stack[:-1]
            else:
                # Left-moving asteroid explodes
                return stack
        
        result = []
        for asteroid in asteroids:
            result = process_asteroid(result, asteroid)
        
        return result
    
    def asteroidCollision_two_pointers(self, asteroids: List[int]) -> List[int]:
        """
        Approach 4: Two Pointers Approach
        
        Use two pointers to track collisions.
        
        Time: O(n), Space: O(n)
        """
        result = []
        
        for asteroid in asteroids:
            # Handle collisions with existing asteroids
            while result and result[-1] > 0 and asteroid < 0:
                if result[-1] < -asteroid:
                    # Previous asteroid explodes
                    result.pop()
                elif result[-1] == -asteroid:
                    # Both explode
                    result.pop()
                    asteroid = 0  # Mark current as exploded
                    break
                else:
                    # Current asteroid explodes
                    asteroid = 0
                    break
            
            # Add asteroid if it survived
            if asteroid != 0:
                result.append(asteroid)
        
        return result
    
    def asteroidCollision_state_machine(self, asteroids: List[int]) -> List[int]:
        """
        Approach 5: State Machine Approach
        
        Use state machine to track collision states.
        
        Time: O(n), Space: O(n)
        """
        stack = []
        
        for asteroid in asteroids:
            state = "MOVING"
            current = asteroid
            
            while state == "MOVING":
                if not stack or current > 0 or stack[-1] < 0:
                    # No collision
                    stack.append(current)
                    state = "ADDED"
                else:
                    # Potential collision
                    if stack[-1] < -current:
                        # Stack top explodes
                        stack.pop()
                        # Continue checking
                    elif stack[-1] == -current:
                        # Both explode
                        stack.pop()
                        state = "EXPLODED"
                    else:
                        # Current explodes
                        state = "EXPLODED"
        
        return stack


def test_asteroid_collision():
    """Test asteroid collision algorithms"""
    solver = AsteroidCollision()
    
    test_cases = [
        ([5,10,-5], [5,10], "Example 1"),
        ([8,-8], [], "Example 2"),
        ([10,2,-5], [10], "Example 3"),
        ([-2,-1,1,2], [-2,-1,1,2], "No collisions"),
        ([1,-2,-2,-2], [-2,-2,-2], "Multiple collisions"),
        ([5,10,-5,-10], [], "Chain collisions"),
        ([1,2,3,-3,-2,-1], [], "Perfect cancellation"),
        ([10,5,-5], [10], "Partial collision"),
        ([-1,-2,-3], [-1,-2,-3], "All left-moving"),
        ([1,2,3], [1,2,3], "All right-moving"),
        ([5,-5], [], "Equal size collision"),
        ([1,-1,2,-2,3,-3], [], "Alternating collisions"),
    ]
    
    algorithms = [
        ("Stack", solver.asteroidCollision_stack),
        ("Simulation", solver.asteroidCollision_simulation),
        ("Recursive", solver.asteroidCollision_recursive),
        ("Two Pointers", solver.asteroidCollision_two_pointers),
        ("State Machine", solver.asteroidCollision_state_machine),
    ]
    
    print("=== Testing Asteroid Collision ===")
    
    for asteroids, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input: {asteroids}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(asteroids[:])  # Copy to avoid modification
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    asteroids = [5, 10, -5]
    
    print(f"Input: {asteroids}")
    print("Strategy: Use stack to handle collisions as they occur")
    
    stack = []
    
    for i, asteroid in enumerate(asteroids):
        print(f"\nStep {i+1}: Processing asteroid {asteroid}")
        print(f"  Current stack: {stack}")
        
        if asteroid > 0:
            print(f"  Right-moving asteroid, add to stack")
            stack.append(asteroid)
        else:
            print(f"  Left-moving asteroid, check for collisions")
            
            # Handle collisions
            while stack and stack[-1] > 0:
                print(f"    Collision: {stack[-1]} (right) vs {asteroid} (left)")
                
                if stack[-1] < -asteroid:
                    exploded = stack.pop()
                    print(f"    Right asteroid {exploded} explodes")
                    continue
                elif stack[-1] == -asteroid:
                    exploded = stack.pop()
                    print(f"    Both asteroids explode: {exploded} and {-asteroid}")
                    asteroid = 0  # Mark as exploded
                    break
                else:
                    print(f"    Left asteroid {asteroid} explodes")
                    asteroid = 0
                    break
            
            if asteroid != 0:
                print(f"    Left asteroid {asteroid} survives, add to stack")
                stack.append(asteroid)
        
        print(f"  Stack after: {stack}")
    
    print(f"\nFinal result: {stack}")


def visualize_collision_scenarios():
    """Visualize different collision scenarios"""
    print("\n=== Collision Scenarios Visualization ===")
    
    scenarios = [
        ([5, -3], "Right meets smaller left"),
        ([3, -5], "Right meets larger left"),
        ([5, -5], "Equal size collision"),
        ([5, 10, -5], "Left collides with rightmost"),
        ([5, 10, -15], "Left destroys multiple right"),
        ([-5, 5], "No collision (same direction)"),
        ([5, -3, 2], "Collision then continuation"),
    ]
    
    solver = AsteroidCollision()
    
    for asteroids, description in scenarios:
        print(f"\nScenario: {description}")
        print(f"  Input:  {asteroids}")
        
        # Show direction visualization
        visual = []
        for a in asteroids:
            if a > 0:
                visual.append(f"{a}→")
            else:
                visual.append(f"←{-a}")
        
        print(f"  Visual: {' '.join(visual)}")
        
        result = solver.asteroidCollision_stack(asteroids)
        print(f"  Result: {result}")


def demonstrate_competitive_programming_patterns():
    """Demonstrate competitive programming patterns"""
    print("\n=== Competitive Programming Patterns ===")
    
    solver = AsteroidCollision()
    
    # Pattern 1: Stack for collision detection
    print("1. Stack for Collision Detection:")
    print("   Use stack to maintain active elements")
    print("   Process collisions immediately when detected")
    
    example1 = [10, 5, -5, -10]
    result1 = solver.asteroidCollision_stack(example1)
    print(f"   {example1} -> {result1}")
    
    # Pattern 2: Simulation with state tracking
    print(f"\n2. State-based Processing:")
    print("   Track different states: MOVING, EXPLODED, ADDED")
    print("   Handle complex collision chains")
    
    # Pattern 3: Amortized analysis
    print(f"\n3. Amortized Analysis:")
    print("   Each asteroid is pushed/popped at most once")
    print("   Total time complexity: O(n)")
    
    # Pattern 4: Direction-based logic
    print(f"\n4. Direction-based Logic:")
    print("   Collisions only occur between right-moving and left-moving")
    print("   Same direction asteroids never collide")
    
    example4 = [-2, -1, 1, 2]
    result4 = solver.asteroidCollision_stack(example4)
    print(f"   {example4} -> {result4} (no collisions)")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack", "O(n)", "O(n)", "Each asteroid processed once"),
        ("Simulation", "O(n²)", "O(n)", "Multiple passes in worst case"),
        ("Recursive", "O(n)", "O(n)", "Recursion stack overhead"),
        ("Two Pointers", "O(n)", "O(n)", "Similar to stack approach"),
        ("State Machine", "O(n)", "O(n)", "State-based processing"),
    ]
    
    print(f"{'Approach':<15} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 55)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<15} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nStack approach is optimal for competitive programming")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = AsteroidCollision()
    
    edge_cases = [
        ([], [], "Empty array"),
        ([1], [1], "Single asteroid"),
        ([-1], [-1], "Single left-moving"),
        ([1, -1], [], "Single collision"),
        ([1000, -1], [1000], "Large vs small"),
        ([-1000, 1], [-1000, 1], "No collision"),
        ([1, 2, 3, -6], [-6], "One destroys all"),
        ([6, -1, -2, -3], [6], "One survives all"),
        ([1, -1, 2, -2, 3, -3], [], "Perfect cancellation"),
        ([5, 5, -5, -5], [], "Multiple equal collisions"),
        ([1, 2, -1, -2], [], "Cross collisions"),
        ([10, -5, 3, -8], [10, 3], "Complex scenario"),
    ]
    
    for asteroids, expected, description in edge_cases:
        try:
            result = solver.asteroidCollision_stack(asteroids[:])
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | {asteroids} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solver = AsteroidCollision()
    
    # Application 1: Traffic flow simulation
    print("1. Traffic Flow Simulation:")
    print("   Vehicles moving in opposite directions on collision course")
    
    vehicles = [60, 80, -70, -90]  # speeds: positive = east, negative = west
    result = solver.asteroidCollision_stack(vehicles)
    print(f"   Vehicle speeds: {vehicles}")
    print(f"   After collisions: {result}")
    print("   (Larger speed vehicle survives collision)")
    
    # Application 2: Chemical reaction simulation
    print(f"\n2. Chemical Reaction Simulation:")
    print("   Particles with different energies colliding")
    
    particles = [100, 50, -75, 25, -150]
    result = solver.asteroidCollision_stack(particles)
    print(f"   Particle energies: {particles}")
    print(f"   After reactions: {result}")
    
    # Application 3: Game collision system
    print(f"\n3. Game Collision System:")
    print("   Projectiles moving in opposite directions")
    
    projectiles = [10, 20, -15, 5, -25]
    result = solver.asteroidCollision_stack(projectiles)
    print(f"   Projectile powers: {projectiles}")
    print(f"   Surviving projectiles: {result}")


if __name__ == "__main__":
    test_asteroid_collision()
    demonstrate_stack_approach()
    visualize_collision_scenarios()
    demonstrate_competitive_programming_patterns()
    analyze_time_complexity()
    test_edge_cases()
    demonstrate_real_world_applications()

"""
Asteroid Collision demonstrates competitive programming patterns
with stack-based simulation, collision detection, and efficient
state management for dynamic interaction problems.
"""
