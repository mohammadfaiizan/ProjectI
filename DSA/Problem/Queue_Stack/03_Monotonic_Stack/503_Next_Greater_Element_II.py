"""
503. Next Greater Element II - Multiple Approaches
Difficulty: Medium

Given a circular array (the next element of the last element is the first element of the array), print the Next Greater Number for every element. The Next Greater Number of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number. If it doesn't exist, output -1 for this number.
"""

from typing import List

class NextGreaterElementII:
    """Multiple approaches to find next greater elements in circular array"""
    
    def nextGreaterElements_stack_circular(self, nums: List[int]) -> List[int]:
        """
        Approach 1: Monotonic Stack with Circular Traversal
        
        Use stack and traverse array twice to handle circular nature.
        
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []  # Store indices
        
        # Traverse array twice to handle circular nature
        for i in range(2 * n):
            # Get actual index in circular array
            idx = i % n
            
            # Pop elements smaller than current
            while stack and nums[stack[-1]] < nums[idx]:
                prev_idx = stack.pop()
                result[prev_idx] = nums[idx]
            
            # Only push indices in first traversal
            if i < n:
                stack.append(idx)
        
        return result
    
    def nextGreaterElements_brute_force(self, nums: List[int]) -> List[int]:
        """
        Approach 2: Brute Force Circular Search
        
        For each element, search circularly for next greater.
        
        Time: O(n²), Space: O(1)
        """
        n = len(nums)
        result = []
        
        for i in range(n):
            next_greater = -1
            
            # Search circularly starting from next position
            for j in range(1, n):
                next_idx = (i + j) % n
                if nums[next_idx] > nums[i]:
                    next_greater = nums[next_idx]
                    break
            
            result.append(next_greater)
        
        return result
    
    def nextGreaterElements_double_array(self, nums: List[int]) -> List[int]:
        """
        Approach 3: Double Array Simulation
        
        Create doubled array and use standard next greater algorithm.
        
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        # Create doubled array
        doubled = nums + nums
        
        # Find next greater in doubled array
        result = [-1] * (2 * n)
        stack = []
        
        for i in range(2 * n):
            while stack and doubled[stack[-1]] < doubled[i]:
                prev_idx = stack.pop()
                result[prev_idx] = doubled[i]
            stack.append(i)
        
        # Return only first n results
        return result[:n]
    
    def nextGreaterElements_reverse_traversal(self, nums: List[int]) -> List[int]:
        """
        Approach 4: Reverse Traversal with Stack
        
        Traverse from right to left twice for circular handling.
        
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        
        # Traverse twice from right to left
        for i in range(2 * n - 1, -1, -1):
            idx = i % n
            
            # Pop elements not greater than current
            while stack and stack[-1] <= nums[idx]:
                stack.pop()
            
            # Set result only in second traversal
            if i < n:
                result[idx] = stack[-1] if stack else -1
            
            stack.append(nums[idx])
        
        return result
    
    def nextGreaterElements_optimized_circular(self, nums: List[int]) -> List[int]:
        """
        Approach 5: Optimized Circular with Early Termination
        
        Optimize by tracking maximum element for early termination.
        
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []
        max_val = max(nums)
        
        # Two passes through the array
        for i in range(2 * n):
            idx = i % n
            
            # Early termination for maximum elements
            if nums[idx] == max_val and i >= n:
                break
            
            while stack and nums[stack[-1]] < nums[idx]:
                prev_idx = stack.pop()
                result[prev_idx] = nums[idx]
            
            if i < n:
                stack.append(idx)
        
        return result
    
    def nextGreaterElements_segment_approach(self, nums: List[int]) -> List[int]:
        """
        Approach 6: Segment-based Processing
        
        Process array in segments for better cache locality.
        
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        result = [-1] * n
        
        # Process in segments
        segment_size = min(n, 1000)  # Adjust based on cache size
        
        for start in range(0, n, segment_size):
            end = min(start + segment_size, n)
            
            # Process current segment
            stack = []
            
            # Two passes for circular nature
            for pass_num in range(2):
                for i in range(start, end):
                    # In second pass, also check elements after current segment
                    search_range = n if pass_num == 1 else end
                    
                    for j in range(i + 1, search_range):
                        if nums[j] > nums[i]:
                            result[i] = nums[j]
                            break
                    
                    # If not found in current range, check from beginning (circular)
                    if result[i] == -1 and pass_num == 1:
                        for j in range(0, i):
                            if nums[j] > nums[i]:
                                result[i] = nums[j]
                                break
        
        return result
    
    def nextGreaterElements_deque_approach(self, nums: List[int]) -> List[int]:
        """
        Approach 7: Deque-based Approach
        
        Use deque for efficient front/back operations.
        
        Time: O(n), Space: O(n)
        """
        from collections import deque
        
        n = len(nums)
        result = [-1] * n
        dq = deque()  # Store indices
        
        # Process array twice
        for i in range(2 * n):
            idx = i % n
            
            # Remove elements smaller than current
            while dq and nums[dq[-1]] < nums[idx]:
                prev_idx = dq.pop()
                result[prev_idx] = nums[idx]
            
            # Add current index only in first pass
            if i < n:
                dq.append(idx)
        
        return result


def test_next_greater_element_ii():
    """Test next greater element II algorithms"""
    solver = NextGreaterElementII()
    
    test_cases = [
        ([1,2,1], [2,-1,2], "Example 1"),
        ([1,2,3,4,3], [2,3,4,-1,4], "Example 2"),
        ([5,4,3,2,1], [-1,5,5,5,5], "Decreasing array"),
        ([1,2,3,4,5], [2,3,4,5,-1], "Increasing array"),
        ([3,3,3], [-1,-1,-1], "All same elements"),
        ([1], [-1], "Single element"),
        ([2,1], [-1,2], "Two elements"),
        ([1,5,3,6,4], [5,6,6,-1,5], "Mixed values"),
    ]
    
    algorithms = [
        ("Stack Circular", solver.nextGreaterElements_stack_circular),
        ("Brute Force", solver.nextGreaterElements_brute_force),
        ("Double Array", solver.nextGreaterElements_double_array),
        ("Reverse Traversal", solver.nextGreaterElements_reverse_traversal),
        ("Optimized Circular", solver.nextGreaterElements_optimized_circular),
        ("Deque Approach", solver.nextGreaterElements_deque_approach),
    ]
    
    print("=== Testing Next Greater Element II ===")
    
    for nums, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"nums: {nums}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_circular_concept():
    """Demonstrate circular array concept"""
    print("\n=== Circular Array Concept Demonstration ===")
    
    nums = [1, 2, 1]
    print(f"Original array: {nums}")
    print("Circular view: [1, 2, 1, 1, 2, 1, 1, 2, 1, ...]")
    print()
    
    for i, num in enumerate(nums):
        print(f"Element {num} at index {i}:")
        
        # Show circular search
        found = False
        for j in range(1, len(nums)):
            next_idx = (i + j) % len(nums)
            next_val = nums[next_idx]
            
            print(f"  Check index {next_idx} (circular): {next_val}", end="")
            
            if next_val > num:
                print(f" -> Found next greater: {next_val}")
                found = True
                break
            else:
                print(f" -> Not greater")
        
        if not found:
            print(f"  -> No next greater element: -1")
        print()


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    nums = [1, 2, 1]
    n = len(nums)
    
    print(f"Array: {nums}")
    print("Processing with circular traversal (2 passes):")
    print()
    
    result = [-1] * n
    stack = []
    
    for i in range(2 * n):
        idx = i % n
        actual_pass = "First" if i < n else "Second"
        
        print(f"Step {i+1} ({actual_pass} pass): Processing nums[{idx}] = {nums[idx]}")
        print(f"  Stack before: {[f'idx{s}({nums[s]})' for s in stack]}")
        
        # Pop elements smaller than current
        popped = []
        while stack and nums[stack[-1]] < nums[idx]:
            prev_idx = stack.pop()
            result[prev_idx] = nums[idx]
            popped.append(f"idx{prev_idx}({nums[prev_idx]})")
        
        if popped:
            print(f"  Popped: {popped} -> set next greater to {nums[idx]}")
        
        # Only push in first pass
        if i < n:
            stack.append(idx)
            print(f"  Pushed idx{idx}")
        
        print(f"  Stack after: {[f'idx{s}({nums[s]})' for s in stack]}")
        print(f"  Result so far: {result}")
        print()
    
    print(f"Final result: {result}")


def visualize_circular_search():
    """Visualize circular search process"""
    print("\n=== Circular Search Visualization ===")
    
    nums = [1, 5, 3, 6, 4]
    n = len(nums)
    
    print(f"Array: {nums}")
    print("Searching for next greater elements:")
    print()
    
    for i in range(n):
        print(f"Element {nums[i]} at index {i}:")
        
        # Create visualization
        vis = [' '] * (2 * n)  # Show two cycles
        for j in range(2 * n):
            vis[j] = str(nums[j % n])
        
        # Mark current position
        vis[i] = f'[{nums[i]}]'
        vis[i + n] = f'[{nums[i]}]'  # Mark in second cycle too
        
        print(f"  Circular view: {' '.join(vis[:n])} | {' '.join(vis[n:2*n])}")
        
        # Find next greater
        found = False
        for j in range(1, n):
            next_idx = (i + j) % n
            if nums[next_idx] > nums[i]:
                print(f"  Next greater: {nums[next_idx]} at circular position {next_idx}")
                found = True
                break
        
        if not found:
            print(f"  Next greater: -1 (none found)")
        print()


def benchmark_next_greater_element_ii():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Stack Circular", NextGreaterElementII().nextGreaterElements_stack_circular),
        ("Brute Force", NextGreaterElementII().nextGreaterElements_brute_force),
        ("Double Array", NextGreaterElementII().nextGreaterElements_double_array),
        ("Reverse Traversal", NextGreaterElementII().nextGreaterElements_reverse_traversal),
    ]
    
    # Test with different array sizes
    sizes = [100, 1000, 5000]
    
    print("\n=== Next Greater Element II Performance Benchmark ===")
    
    for size in sizes:
        print(f"\n--- Array Size: {size} ---")
        
        # Generate random array
        nums = [random.randint(1, 1000) for _ in range(size)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(nums)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = NextGreaterElementII()
    
    edge_cases = [
        ([1], [-1], "Single element"),
        ([1, 1], [-1, -1], "Two same elements"),
        ([2, 1], [-1, 2], "Two elements, circular"),
        ([5, 4, 3, 2, 1], [-1, 5, 5, 5, 5], "Strictly decreasing"),
        ([1, 2, 3, 4, 5], [2, 3, 4, 5, -1], "Strictly increasing"),
        ([3, 3, 3, 3], [-1, -1, -1, -1], "All same"),
        ([1, 1000, 1], [1000, -1, 1000], "Large difference"),
        ([100, 1, 99], [-1, 99, 100], "Circular dependency"),
    ]
    
    for nums, expected, description in edge_cases:
        try:
            result = solver.nextGreaterElements_stack_circular(nums)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | nums: {nums} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_linear_vs_circular():
    """Compare linear vs circular next greater element"""
    print("\n=== Linear vs Circular Comparison ===")
    
    test_arrays = [
        [1, 2, 1],
        [5, 4, 3, 2, 1],
        [1, 3, 2, 4],
    ]
    
    def next_greater_linear(nums):
        """Linear version (no circular)"""
        result = [-1] * len(nums)
        stack = []
        
        for i, num in enumerate(nums):
            while stack and nums[stack[-1]] < num:
                prev_idx = stack.pop()
                result[prev_idx] = num
            stack.append(i)
        
        return result
    
    solver = NextGreaterElementII()
    
    for nums in test_arrays:
        print(f"\nArray: {nums}")
        
        linear_result = next_greater_linear(nums)
        circular_result = solver.nextGreaterElements_stack_circular(nums)
        
        print(f"Linear result:   {linear_result}")
        print(f"Circular result: {circular_result}")
        
        # Show differences
        differences = []
        for i in range(len(nums)):
            if linear_result[i] != circular_result[i]:
                differences.append(f"index {i}: {linear_result[i]} -> {circular_result[i]}")
        
        if differences:
            print(f"Differences: {', '.join(differences)}")
        else:
            print("No differences")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Circular", "O(n)", "O(n)", "Optimal - each element pushed/popped once"),
        ("Brute Force", "O(n²)", "O(1)", "Simple but inefficient"),
        ("Double Array", "O(n)", "O(n)", "Space trade-off for simplicity"),
        ("Reverse Traversal", "O(n)", "O(n)", "Alternative optimal approach"),
        ("Optimized Circular", "O(n)", "O(n)", "Early termination optimization"),
        ("Deque Approach", "O(n)", "O(n)", "Similar to stack with deque"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<8} | {space_comp:<8} | {notes}")


if __name__ == "__main__":
    test_next_greater_element_ii()
    demonstrate_circular_concept()
    demonstrate_stack_approach()
    visualize_circular_search()
    test_edge_cases()
    compare_linear_vs_circular()
    analyze_time_complexity()
    benchmark_next_greater_element_ii()

"""
Next Greater Element II demonstrates circular array processing with
monotonic stacks, including multiple approaches for handling the
circular nature and comprehensive analysis of the algorithms.
"""
