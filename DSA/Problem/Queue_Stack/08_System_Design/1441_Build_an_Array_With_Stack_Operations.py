"""
1441. Build an Array With Stack Operations - Multiple Approaches
Difficulty: Easy

You are given an integer array target and an integer n.

You have an empty stack and you can perform the following operations:
- "Push": pushes an integer from 1 to n to the top of the stack.
- "Pop": removes the top element of the stack.

You can perform these operations in any order.

Return the stack operations needed to build target as the final stack (from bottom to top), in order.

You are guaranteed that all numbers in target are unique and sorted in ascending order. Also, 1 <= target[i] <= n.
"""

from typing import List

class BuildArrayWithStackOperations:
    """Multiple approaches to build array with stack operations"""
    
    def buildArray_simulation(self, target: List[int], n: int) -> List[str]:
        """
        Approach 1: Direct Simulation (Optimal)
        
        Simulate the stack operations directly.
        
        Time: O(target[-1]), Space: O(1) excluding output
        """
        result = []
        target_index = 0
        
        for num in range(1, n + 1):
            if target_index >= len(target):
                break
            
            # Always push the current number
            result.append("Push")
            
            if num == target[target_index]:
                # This number is in target, keep it
                target_index += 1
            else:
                # This number is not in target, pop it
                result.append("Pop")
        
        return result
    
    def buildArray_set_approach(self, target: List[int], n: int) -> List[str]:
        """
        Approach 2: Set-based Approach
        
        Use set for O(1) lookup of target elements.
        
        Time: O(target[-1]), Space: O(len(target))
        """
        result = []
        target_set = set(target)
        
        for num in range(1, target[-1] + 1):
            result.append("Push")
            
            if num not in target_set:
                result.append("Pop")
        
        return result
    
    def buildArray_two_pointers(self, target: List[int], n: int) -> List[str]:
        """
        Approach 3: Two Pointers Approach
        
        Use two pointers to track current number and target index.
        
        Time: O(target[-1]), Space: O(1) excluding output
        """
        result = []
        current_num = 1
        target_index = 0
        
        while target_index < len(target) and current_num <= n:
            result.append("Push")
            
            if current_num == target[target_index]:
                target_index += 1
            else:
                result.append("Pop")
            
            current_num += 1
        
        return result
    
    def buildArray_stack_simulation(self, target: List[int], n: int) -> List[str]:
        """
        Approach 4: Actual Stack Simulation
        
        Use actual stack to simulate the process.
        
        Time: O(target[-1]), Space: O(len(target))
        """
        result = []
        stack = []
        target_index = 0
        
        for num in range(1, n + 1):
            if target_index >= len(target):
                break
            
            # Push current number
            stack.append(num)
            result.append("Push")
            
            if num == target[target_index]:
                # Keep this number
                target_index += 1
            else:
                # Remove this number
                stack.pop()
                result.append("Pop")
        
        return result
    
    def buildArray_recursive(self, target: List[int], n: int) -> List[str]:
        """
        Approach 5: Recursive Approach
        
        Use recursion to build operations.
        
        Time: O(target[-1]), Space: O(target[-1]) due to recursion
        """
        def build_operations(current_num: int, target_index: int) -> List[str]:
            if target_index >= len(target) or current_num > n:
                return []
            
            operations = ["Push"]
            
            if current_num == target[target_index]:
                # Keep this number, move to next target
                operations.extend(build_operations(current_num + 1, target_index + 1))
            else:
                # Remove this number, try next
                operations.append("Pop")
                operations.extend(build_operations(current_num + 1, target_index))
            
            return operations
        
        return build_operations(1, 0)
    
    def buildArray_functional(self, target: List[int], n: int) -> List[str]:
        """
        Approach 6: Functional Programming Style
        
        Use functional programming concepts.
        
        Time: O(target[-1]), Space: O(1) excluding output
        """
        from itertools import takewhile
        
        def generate_operations():
            target_iter = iter(target)
            current_target = next(target_iter, None)
            
            for num in range(1, n + 1):
                if current_target is None:
                    break
                
                yield "Push"
                
                if num == current_target:
                    current_target = next(target_iter, None)
                else:
                    yield "Pop"
        
        return list(generate_operations())


def test_build_array_with_stack_operations():
    """Test build array with stack operations algorithms"""
    solver = BuildArrayWithStackOperations()
    
    test_cases = [
        ([1,3], 3, ["Push","Push","Pop","Push"], "Example 1"),
        ([1,2,3], 3, ["Push","Push","Push"], "Example 2"),
        ([1,2], 4, ["Push","Push"], "Example 3"),
        ([2,3,4], 4, ["Push","Pop","Push","Push","Push"], "Skip first element"),
        ([1], 1, ["Push"], "Single element"),
        ([5], 5, ["Push","Pop","Push","Pop","Push","Pop","Push","Pop","Push"], "Last element only"),
        ([1,3,5], 5, ["Push","Push","Pop","Push","Push","Pop","Push"], "Every other element"),
        ([2,4,6], 6, ["Push","Pop","Push","Push","Pop","Push","Push","Pop","Push"], "Even numbers"),
    ]
    
    algorithms = [
        ("Simulation", solver.buildArray_simulation),
        ("Set Approach", solver.buildArray_set_approach),
        ("Two Pointers", solver.buildArray_two_pointers),
        ("Stack Simulation", solver.buildArray_stack_simulation),
        ("Recursive", solver.buildArray_recursive),
        ("Functional", solver.buildArray_functional),
    ]
    
    print("=== Testing Build Array With Stack Operations ===")
    
    for target, n, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Target: {target}, n: {n}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(target, n)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_simulation_approach():
    """Demonstrate simulation approach step by step"""
    print("\n=== Simulation Approach Step-by-Step Demo ===")
    
    target = [1, 3]
    n = 3
    
    print(f"Target: {target}, n: {n}")
    print("Strategy: Iterate through numbers 1 to n, push each, pop if not in target")
    
    result = []
    target_index = 0
    
    print(f"\nStep-by-step simulation:")
    
    for num in range(1, n + 1):
        print(f"\nProcessing number {num}:")
        print(f"  Target index: {target_index}")
        print(f"  Looking for: {target[target_index] if target_index < len(target) else 'None'}")
        
        # Always push
        result.append("Push")
        print(f"  Operation: Push -> Stack conceptually has {num}")
        
        if target_index < len(target) and num == target[target_index]:
            print(f"  {num} matches target[{target_index}] = {target[target_index]}")
            print(f"  Keep {num} in stack")
            target_index += 1
        else:
            print(f"  {num} not needed, pop it")
            result.append("Pop")
        
        print(f"  Operations so far: {result}")
    
    print(f"\nFinal operations: {result}")


def visualize_stack_operations():
    """Visualize stack operations"""
    print("\n=== Stack Operations Visualization ===")
    
    target = [2, 3, 4]
    n = 4
    
    print(f"Target: {target}, n: {n}")
    print("Visualizing actual stack state during operations:")
    
    stack = []
    operations = []
    target_set = set(target)
    
    print(f"\nInitial stack: {stack}")
    
    for num in range(1, n + 1):
        print(f"\nStep {num}: Processing number {num}")
        
        # Push operation
        stack.append(num)
        operations.append("Push")
        print(f"  Push {num} -> Stack: {stack}")
        
        if num not in target_set:
            # Pop operation
            popped = stack.pop()
            operations.append("Pop")
            print(f"  Pop {popped} -> Stack: {stack}")
        else:
            print(f"  Keep {num} (it's in target)")
    
    print(f"\nFinal stack: {stack}")
    print(f"Target achieved: {stack == target}")
    print(f"Operations: {operations}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solver = BuildArrayWithStackOperations()
    
    # Application 1: Assembly line production
    print("1. Assembly Line Production Control:")
    
    # Simulate production line where we need specific product IDs
    required_products = [2, 5, 7, 9]
    total_products = 10
    
    operations = solver.buildArray_simulation(required_products, total_products)
    
    print(f"  Required products: {required_products}")
    print(f"  Production line processes products 1-{total_products}")
    print(f"  Operations needed:")
    
    current_product = 1
    for op in operations:
        if op == "Push":
            print(f"    Process product {current_product}")
            current_product += 1
        else:
            print(f"    Reject product {current_product - 1}")
    
    # Application 2: Data filtering system
    print(f"\n2. Data Stream Filtering:")
    
    # Filter specific data packets from a stream
    wanted_packets = [1, 4, 6]
    stream_size = 8
    
    filter_ops = solver.buildArray_simulation(wanted_packets, stream_size)
    
    print(f"  Wanted packets: {wanted_packets}")
    print(f"  Stream contains packets 1-{stream_size}")
    print(f"  Filtering operations: {filter_ops}")
    
    # Application 3: Task scheduling
    print(f"\n3. Task Scheduling System:")
    
    # Schedule specific tasks from a queue
    priority_tasks = [3, 5, 8]
    total_tasks = 10
    
    schedule_ops = solver.buildArray_simulation(priority_tasks, total_tasks)
    
    print(f"  Priority tasks: {priority_tasks}")
    print(f"  Task queue has tasks 1-{total_tasks}")
    print(f"  Scheduling operations: {schedule_ops}")
    
    # Show execution
    task_num = 1
    scheduled_tasks = []
    
    for op in schedule_ops:
        if op == "Push":
            print(f"    Consider task {task_num}")
            if task_num in priority_tasks:
                scheduled_tasks.append(task_num)
                print(f"      -> Schedule task {task_num}")
            task_num += 1
        else:
            print(f"      -> Skip task {task_num - 1}")
    
    print(f"  Final scheduled tasks: {scheduled_tasks}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Simulation", "O(target[-1])", "O(1)", "Iterate up to max target"),
        ("Set Approach", "O(target[-1])", "O(len(target))", "Set for O(1) lookup"),
        ("Two Pointers", "O(target[-1])", "O(1)", "Two pointer technique"),
        ("Stack Simulation", "O(target[-1])", "O(len(target))", "Actual stack operations"),
        ("Recursive", "O(target[-1])", "O(target[-1])", "Recursion stack overhead"),
        ("Functional", "O(target[-1])", "O(1)", "Generator-based approach"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<15} | {'Space':<15} | {'Notes'}")
    print("-" * 75)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<15} | {space_comp:<15} | {notes}")
    
    print(f"\nSimulation and Two Pointers are optimal")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = BuildArrayWithStackOperations()
    
    edge_cases = [
        ([1], 1, ["Push"], "Single element, n=1"),
        ([1], 5, ["Push"], "Single element, large n"),
        ([5], 5, ["Push","Pop","Push","Pop","Push","Pop","Push","Pop","Push"], "Last element only"),
        ([1,2,3,4,5], 5, ["Push","Push","Push","Push","Push"], "All consecutive elements"),
        ([2], 2, ["Push","Pop","Push"], "Skip first element"),
        ([1,3,5,7,9], 10, ["Push","Push","Pop","Push","Push","Pop","Push","Push","Pop","Push","Push","Pop","Push"], "Odd numbers"),
        ([2,4,6,8], 8, ["Push","Pop","Push","Push","Pop","Push","Push","Pop","Push","Push","Pop","Push"], "Even numbers"),
    ]
    
    for target, n, expected, description in edge_cases:
        try:
            result = solver.buildArray_simulation(target, n)
            status = "✓" if result == expected else "✗"
            print(f"{description:30} | {status} | target={target}, n={n}")
            if result != expected:
                print(f"  Expected: {expected}")
                print(f"  Got:      {result}")
        except Exception as e:
            print(f"{description:30} | ERROR: {str(e)[:30]}")


def verify_operations():
    """Verify that operations actually produce the target"""
    print("\n=== Verifying Operations Produce Target ===")
    
    solver = BuildArrayWithStackOperations()
    
    test_cases = [
        ([1,3], 3),
        ([1,2,3], 3),
        ([2,3,4], 4),
        ([1,3,5], 5),
    ]
    
    for target, n in test_cases:
        operations = solver.buildArray_simulation(target, n)
        
        # Simulate the operations
        stack = []
        current_num = 1
        
        for op in operations:
            if op == "Push":
                stack.append(current_num)
                current_num += 1
            elif op == "Pop":
                if stack:
                    stack.pop()
        
        success = stack == target
        status = "✓" if success else "✗"
        
        print(f"Target: {target:10} | {status} | Operations: {len(operations):2} | Final stack: {stack}")


def benchmark_approaches():
    """Benchmark different approaches"""
    import time
    
    approaches = [
        ("Simulation", BuildArrayWithStackOperations().buildArray_simulation),
        ("Set Approach", BuildArrayWithStackOperations().buildArray_set_approach),
        ("Two Pointers", BuildArrayWithStackOperations().buildArray_two_pointers),
        ("Stack Simulation", BuildArrayWithStackOperations().buildArray_stack_simulation),
    ]
    
    # Large test case
    target = list(range(1, 1000, 2))  # Odd numbers up to 999
    n = 1000
    
    print(f"\n=== Performance Benchmark ===")
    print(f"Target size: {len(target)}, n: {n}")
    
    for name, func in approaches:
        start_time = time.time()
        
        try:
            result = func(target, n)
            end_time = time.time()
            print(f"{name:20} | Time: {end_time - start_time:.4f}s | Operations: {len(result)}")
        except Exception as e:
            print(f"{name:20} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_build_array_with_stack_operations()
    demonstrate_simulation_approach()
    visualize_stack_operations()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()
    verify_operations()
    benchmark_approaches()

"""
Build an Array With Stack Operations demonstrates system design
for stack-based operations simulation, including multiple approaches
for generating operation sequences and real-world applications.
"""
