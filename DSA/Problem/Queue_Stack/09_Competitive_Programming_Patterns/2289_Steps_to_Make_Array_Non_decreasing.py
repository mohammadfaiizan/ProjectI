"""
2289. Steps to Make Array Non-decreasing - Multiple Approaches
Difficulty: Hard

You are given a 0-indexed integer array nums. In one step, remove all elements nums[i] where nums[i - 1] > nums[i] for all valid i.

Return the number of steps performed until nums becomes non-decreasing.
"""

from typing import List

class StepsToMakeArrayNonDecreasing:
    """Multiple approaches to calculate steps to make array non-decreasing"""
    
    def totalSteps_stack(self, nums: List[int]) -> int:
        """
        Approach 1: Monotonic Stack (Optimal)
        
        Use monotonic stack to track when each element will be removed.
        
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        stack = []  # Stack of (index, steps_to_remove)
        max_steps = 0
        
        for i in range(n):
            steps = 0
            
            # Remove elements that are smaller than current element
            while stack and nums[stack[-1][0]] > nums[i]:
                # Current element will remove the top element
                steps = max(steps + 1, stack[-1][1])
                stack.pop()
            
            # If stack is not empty, current element might be removed later
            if stack:
                stack.append((i, steps))
            else:
                # First element or all previous elements removed
                stack.append((i, 0))
            
            max_steps = max(max_steps, steps)
        
        return max_steps
    
    def totalSteps_simulation(self, nums: List[int]) -> int:
        """
        Approach 2: Direct Simulation
        
        Simulate the removal process step by step.
        
        Time: O(n²), Space: O(n)
        """
        arr = nums[:]
        steps = 0
        
        while True:
            to_remove = []
            
            # Find elements to remove in this step
            for i in range(1, len(arr)):
                if arr[i - 1] > arr[i]:
                    to_remove.append(i)
            
            if not to_remove:
                break
            
            # Remove elements (in reverse order to maintain indices)
            for i in reversed(to_remove):
                arr.pop(i)
            
            steps += 1
        
        return steps
    
    def totalSteps_dp(self, nums: List[int]) -> int:
        """
        Approach 3: Dynamic Programming
        
        Use DP to calculate steps for each element.
        
        Time: O(n²), Space: O(n)
        """
        n = len(nums)
        dp = [0] * n  # dp[i] = steps to remove element i
        
        for i in range(1, n):
            if nums[i - 1] > nums[i]:
                dp[i] = 1
                
                # Check if previous elements will also be removed
                j = i - 1
                while j >= 0 and nums[j] > nums[i]:
                    if dp[j] > 0:
                        dp[i] = max(dp[i], dp[j] + 1)
                    j -= 1
        
        return max(dp) if dp else 0
    
    def totalSteps_recursive(self, nums: List[int]) -> int:
        """
        Approach 4: Recursive Solution
        
        Use recursion to simulate the removal process.
        
        Time: O(n²), Space: O(n) due to recursion
        """
        def simulate(arr: List[int]) -> int:
            if len(arr) <= 1:
                return 0
            
            to_remove = []
            for i in range(1, len(arr)):
                if arr[i - 1] > arr[i]:
                    to_remove.append(i)
            
            if not to_remove:
                return 0
            
            # Create new array without removed elements
            new_arr = []
            remove_set = set(to_remove)
            
            for i in range(len(arr)):
                if i not in remove_set:
                    new_arr.append(arr[i])
            
            return 1 + simulate(new_arr)
        
        return simulate(nums)
    
    def totalSteps_optimized_stack(self, nums: List[int]) -> int:
        """
        Approach 5: Optimized Stack with Time Tracking
        
        Enhanced stack approach with better time tracking.
        
        Time: O(n), Space: O(n)
        """
        stack = []  # Stack of (value, time_to_remove)
        max_time = 0
        
        for num in nums:
            time = 0
            
            # Process elements that will be removed by current element
            while stack and stack[-1][0] > num:
                time = max(time + 1, stack[-1][1])
                stack.pop()
            
            # Add current element to stack
            if stack:
                stack.append((num, time))
                max_time = max(max_time, time)
            else:
                stack.append((num, 0))
        
        return max_time


def test_steps_to_make_array_non_decreasing():
    """Test steps to make array non-decreasing algorithms"""
    solver = StepsToMakeArrayNonDecreasing()
    
    test_cases = [
        ([5,3,4,4,7,3,6,11,8,5,11], 3, "Example 1"),
        ([4,5,7,7,13], 0, "Example 2"),
        ([10,1,2,3,4,5,6,1,2,3], 6, "Example 3"),
        ([1], 0, "Single element"),
        ([1,2,3,4,5], 0, "Already non-decreasing"),
        ([5,4,3,2,1], 4, "Strictly decreasing"),
        ([3,2,1,4], 2, "Mixed pattern"),
        ([1,2,1,3,1], 2, "Alternating"),
        ([10,5,8,3,6,2,4,1], 5, "Complex case"),
        ([2,1,3,1,4,1], 3, "Multiple removals"),
        ([5,5,5,5], 0, "All equal"),
        ([6,2,7,3,8,4,9], 3, "Interleaved"),
    ]
    
    algorithms = [
        ("Stack", solver.totalSteps_stack),
        ("Simulation", solver.totalSteps_simulation),
        ("DP", solver.totalSteps_dp),
        ("Recursive", solver.totalSteps_recursive),
        ("Optimized Stack", solver.totalSteps_optimized_stack),
    ]
    
    print("=== Testing Steps to Make Array Non-decreasing ===")
    
    for nums, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Input: {nums}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums[:])  # Copy to avoid modification
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    nums = [5, 3, 4, 4, 7, 3, 6, 11, 8, 5, 11]
    
    print(f"Input: {nums}")
    print("Strategy: Use stack to track when each element will be removed")
    print("Key insight: Track the time when each element gets removed")
    
    stack = []
    max_steps = 0
    
    print(f"\nStep-by-step processing:")
    
    for i, num in enumerate(nums):
        print(f"\nStep {i+1}: Processing {num}")
        print(f"  Current stack: {[(val, time) for val, time in stack]}")
        
        steps = 0
        removed = []
        
        # Remove elements that are greater than current
        while stack and stack[-1][0] > num:
            val, time = stack.pop()
            removed.append((val, time))
            steps = max(steps + 1, time)
        
        if removed:
            print(f"  Removed elements: {removed}")
            print(f"  Steps calculated: {steps}")
        
        # Add current element
        if stack:
            stack.append((num, steps))
            print(f"  Added ({num}, {steps}) to stack")
        else:
            stack.append((num, 0))
            print(f"  Added ({num}, 0) to stack (first element)")
        
        max_steps = max(max_steps, steps)
        print(f"  Max steps so far: {max_steps}")
    
    print(f"\nFinal result: {max_steps} steps")


def visualize_removal_process():
    """Visualize the step-by-step removal process"""
    print("\n=== Removal Process Visualization ===")
    
    nums = [5, 3, 4, 4, 7, 3, 6, 11, 8, 5, 11]
    
    print(f"Initial array: {nums}")
    print("Simulating removal process:")
    
    arr = nums[:]
    step = 0
    
    while True:
        print(f"\nStep {step}: {arr}")
        
        to_remove = []
        for i in range(1, len(arr)):
            if arr[i - 1] > arr[i]:
                to_remove.append(i)
        
        if not to_remove:
            print("No more elements to remove - array is non-decreasing")
            break
        
        print(f"  Elements to remove at indices: {to_remove}")
        print(f"  Values to remove: {[arr[i] for i in to_remove]}")
        
        # Remove elements
        for i in reversed(to_remove):
            arr.pop(i)
        
        step += 1
    
    print(f"\nFinal array: {arr}")
    print(f"Total steps: {step}")


def demonstrate_competitive_programming_patterns():
    """Demonstrate competitive programming patterns"""
    print("\n=== Competitive Programming Patterns ===")
    
    solver = StepsToMakeArrayNonDecreasing()
    
    # Pattern 1: Monotonic stack for time tracking
    print("1. Monotonic Stack for Time Tracking:")
    print("   Track when each element will be removed")
    print("   Maintain stack of elements that haven't been removed yet")
    
    example1 = [5, 3, 4, 4, 7, 3, 6]
    result1 = solver.totalSteps_stack(example1)
    print(f"   {example1} -> {result1} steps")
    
    # Pattern 2: Simulation vs optimization
    print(f"\n2. Simulation vs Optimization:")
    print("   Direct simulation: O(n²) time")
    print("   Stack optimization: O(n) time")
    
    # Pattern 3: Time propagation
    print(f"\n3. Time Propagation:")
    print("   When element A removes element B, time propagates")
    print("   Time = max(current_time + 1, removed_element_time)")
    
    # Pattern 4: Removal cascades
    print(f"\n4. Removal Cascades:")
    print("   One element can trigger removal of multiple elements")
    print("   Stack helps track these cascading effects")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack", "O(n)", "O(n)", "Each element pushed/popped once"),
        ("Simulation", "O(n²)", "O(n)", "Multiple passes over array"),
        ("DP", "O(n²)", "O(n)", "Nested loops for DP calculation"),
        ("Recursive", "O(n²)", "O(n)", "Recursive simulation"),
        ("Optimized Stack", "O(n)", "O(n)", "Enhanced stack approach"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<8} | {space_comp:<8} | {notes}")
    
    print(f"\nStack approaches are optimal for competitive programming")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = StepsToMakeArrayNonDecreasing()
    
    edge_cases = [
        ([], 0, "Empty array"),
        ([1], 0, "Single element"),
        ([1, 2], 0, "Two increasing"),
        ([2, 1], 1, "Two decreasing"),
        ([1, 1], 0, "Two equal"),
        ([1, 2, 3], 0, "Strictly increasing"),
        ([3, 2, 1], 2, "Strictly decreasing"),
        ([1, 3, 2], 1, "Peak in middle"),
        ([2, 1, 3], 1, "Valley in middle"),
        ([5, 4, 3, 2, 1], 4, "Long decreasing"),
        ([1, 5, 2, 6, 3], 2, "Alternating pattern"),
        ([10, 1, 2, 3, 4], 1, "Large first element"),
    ]
    
    for nums, expected, description in edge_cases:
        try:
            result = solver.totalSteps_stack(nums[:])
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | {nums} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def demonstrate_time_propagation():
    """Demonstrate time propagation concept"""
    print("\n=== Time Propagation Demo ===")
    
    nums = [10, 5, 8, 3, 6]
    
    print(f"Input: {nums}")
    print("Demonstrating how removal times propagate:")
    
    stack = []
    
    for i, num in enumerate(nums):
        print(f"\nProcessing {num}:")
        
        time = 0
        while stack and stack[-1][0] > num:
            val, prev_time = stack.pop()
            time = max(time + 1, prev_time)
            print(f"  {num} will remove {val} (prev_time={prev_time})")
            print(f"  New time: max({time-1} + 1, {prev_time}) = {time}")
        
        if stack:
            stack.append((num, time))
            print(f"  Added ({num}, {time}) - will be removed at time {time}")
        else:
            stack.append((num, 0))
            print(f"  Added ({num}, 0) - won't be removed")
    
    max_time = max(time for _, time in stack)
    print(f"\nMaximum removal time: {max_time}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    solver = StepsToMakeArrayNonDecreasing()
    
    # Application 1: Task scheduling optimization
    print("1. Task Scheduling Optimization:")
    print("   Remove tasks that violate priority constraints")
    print("   Minimize number of scheduling rounds")
    
    task_priorities = [5, 3, 4, 2, 6, 1, 7]
    steps = solver.totalSteps_stack(task_priorities)
    print(f"   Task priorities: {task_priorities}")
    print(f"   Scheduling rounds needed: {steps}")
    
    # Application 2: Data cleanup process
    print(f"\n2. Data Cleanup Process:")
    print("   Remove data points that violate ordering constraints")
    print("   Minimize cleanup iterations")
    
    data_values = [100, 80, 90, 70, 95, 60, 110]
    cleanup_steps = solver.totalSteps_stack(data_values)
    print(f"   Data values: {data_values}")
    print(f"   Cleanup iterations: {cleanup_steps}")
    
    # Application 3: Quality control system
    print(f"\n3. Quality Control System:")
    print("   Remove products that don't meet quality thresholds")
    print("   Optimize inspection process")
    
    quality_scores = [85, 70, 75, 65, 80, 60, 90]
    inspection_rounds = solver.totalSteps_stack(quality_scores)
    print(f"   Quality scores: {quality_scores}")
    print(f"   Inspection rounds: {inspection_rounds}")


def benchmark_approaches():
    """Benchmark different approaches"""
    import time
    import random
    
    approaches = [
        ("Stack", StepsToMakeArrayNonDecreasing().totalSteps_stack),
        ("Optimized Stack", StepsToMakeArrayNonDecreasing().totalSteps_optimized_stack),
    ]
    
    # Generate test data
    test_sizes = [1000, 5000, 10000]
    
    print(f"\n=== Performance Benchmark ===")
    
    for size in test_sizes:
        print(f"\nArray size: {size}")
        
        # Generate random array
        nums = [random.randint(1, 1000) for _ in range(size)]
        
        for name, func in approaches:
            try:
                start_time = time.time()
                
                # Run multiple times for better measurement
                for _ in range(5):
                    func(nums[:])
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 5
                
                print(f"  {name:20} | Avg time: {avg_time:.6f}s")
                
            except Exception as e:
                print(f"  {name:20} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_steps_to_make_array_non_decreasing()
    demonstrate_stack_approach()
    visualize_removal_process()
    demonstrate_competitive_programming_patterns()
    analyze_time_complexity()
    test_edge_cases()
    demonstrate_time_propagation()
    demonstrate_real_world_applications()
    benchmark_approaches()

"""
Steps to Make Array Non-decreasing demonstrates advanced competitive
programming patterns with monotonic stack optimization, time propagation,
and cascading removal effects for complex array transformation problems.
"""
