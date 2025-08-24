"""
496. Next Greater Element I - Multiple Approaches
Difficulty: Easy

The next greater element of some element x in an array is the first greater element that is to the right of x in the same array.

You are given two distinct arrays nums1 and nums2, where nums1 is a subset of nums2.

Find all the next greater elements for nums1's elements in the corresponding positions of nums2.
"""

from typing import List, Dict

class NextGreaterElementI:
    """Multiple approaches to find next greater elements"""
    
    def nextGreaterElement_stack_approach(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        Approach 1: Monotonic Stack with HashMap
        
        Use stack to find next greater elements and map for quick lookup.
        
        Time: O(n + m), Space: O(n)
        """
        # Build next greater element mapping for nums2
        next_greater = {}
        stack = []
        
        for num in nums2:
            # Pop elements smaller than current
            while stack and stack[-1] < num:
                next_greater[stack.pop()] = num
            stack.append(num)
        
        # Elements remaining in stack have no next greater element
        while stack:
            next_greater[stack.pop()] = -1
        
        # Build result for nums1
        return [next_greater[num] for num in nums1]
    
    def nextGreaterElement_brute_force(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        Approach 2: Brute Force
        
        For each element in nums1, find its position in nums2 and search for next greater.
        
        Time: O(m * n), Space: O(1)
        """
        result = []
        
        for num in nums1:
            # Find position of num in nums2
            pos = nums2.index(num)
            
            # Search for next greater element
            next_greater = -1
            for i in range(pos + 1, len(nums2)):
                if nums2[i] > num:
                    next_greater = nums2[i]
                    break
            
            result.append(next_greater)
        
        return result
    
    def nextGreaterElement_hashmap_precompute(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        Approach 3: HashMap with Precomputed Next Greater
        
        Precompute next greater for all elements in nums2.
        
        Time: O(n²) preprocessing + O(m), Space: O(n)
        """
        # Precompute next greater elements for nums2
        next_greater = {}
        
        for i in range(len(nums2)):
            next_greater[nums2[i]] = -1
            for j in range(i + 1, len(nums2)):
                if nums2[j] > nums2[i]:
                    next_greater[nums2[i]] = nums2[j]
                    break
        
        # Build result for nums1
        return [next_greater[num] for num in nums1]
    
    def nextGreaterElement_reverse_iteration(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        Approach 4: Reverse Iteration with Stack
        
        Iterate nums2 from right to left using stack.
        
        Time: O(n + m), Space: O(n)
        """
        next_greater = {}
        stack = []
        
        # Process nums2 from right to left
        for i in range(len(nums2) - 1, -1, -1):
            num = nums2[i]
            
            # Pop elements not greater than current
            while stack and stack[-1] <= num:
                stack.pop()
            
            # Next greater element is top of stack (or -1 if empty)
            next_greater[num] = stack[-1] if stack else -1
            
            # Push current element
            stack.append(num)
        
        return [next_greater[num] for num in nums1]
    
    def nextGreaterElement_two_pointers(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        Approach 5: Two Pointers Approach
        
        Use two pointers to find next greater elements.
        
        Time: O(m * n), Space: O(1)
        """
        result = []
        
        for target in nums1:
            # Find target in nums2
            found_pos = -1
            for i in range(len(nums2)):
                if nums2[i] == target:
                    found_pos = i
                    break
            
            # Find next greater element
            next_greater = -1
            for i in range(found_pos + 1, len(nums2)):
                if nums2[i] > target:
                    next_greater = nums2[i]
                    break
            
            result.append(next_greater)
        
        return result
    
    def nextGreaterElement_optimized_lookup(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """
        Approach 6: Optimized with Position Lookup
        
        Create position mapping for faster lookup.
        
        Time: O(n + m), Space: O(n)
        """
        # Create position mapping for nums2
        pos_map = {num: i for i, num in enumerate(nums2)}
        
        # Build next greater mapping using stack
        next_greater = {}
        stack = []
        
        for num in nums2:
            while stack and stack[-1] < num:
                next_greater[stack.pop()] = num
            stack.append(num)
        
        # Elements in stack have no next greater
        for num in stack:
            next_greater[num] = -1
        
        return [next_greater[num] for num in nums1]


def test_next_greater_element_i():
    """Test next greater element I algorithms"""
    solver = NextGreaterElementI()
    
    test_cases = [
        ([4,1,2], [1,3,4,2], [3,3,-1], "Example 1"),
        ([2,4], [1,2,3,4], [3,-1], "Example 2"),
        ([1,3,5,2,4], [6,5,4,3,2,1,7], [7,7,7,7,7], "All have next greater"),
        ([1], [1,2], [2], "Single element"),
        ([2], [1,2], [-1], "No next greater"),
        ([1,2,3], [3,2,1], [-1,-1,-1], "Decreasing nums2"),
        ([3,2,1], [1,2,3], [3,3,-1], "Increasing nums2"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.nextGreaterElement_stack_approach),
        ("Brute Force", solver.nextGreaterElement_brute_force),
        ("HashMap Precompute", solver.nextGreaterElement_hashmap_precompute),
        ("Reverse Iteration", solver.nextGreaterElement_reverse_iteration),
        ("Two Pointers", solver.nextGreaterElement_two_pointers),
        ("Optimized Lookup", solver.nextGreaterElement_optimized_lookup),
    ]
    
    print("=== Testing Next Greater Element I ===")
    
    for nums1, nums2, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"nums1: {nums1}, nums2: {nums2}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums1, nums2)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    nums1 = [4, 1, 2]
    nums2 = [1, 3, 4, 2]
    
    print(f"nums1: {nums1}")
    print(f"nums2: {nums2}")
    print("\nBuilding next greater mapping for nums2:")
    
    next_greater = {}
    stack = []
    
    for i, num in enumerate(nums2):
        print(f"\nStep {i+1}: Processing {num}")
        print(f"  Stack before: {stack}")
        
        # Pop elements smaller than current
        while stack and stack[-1] < num:
            popped = stack.pop()
            next_greater[popped] = num
            print(f"  Found next greater for {popped}: {num}")
        
        stack.append(num)
        print(f"  Stack after: {stack}")
    
    # Elements remaining in stack have no next greater
    while stack:
        remaining = stack.pop()
        next_greater[remaining] = -1
        print(f"  No next greater for {remaining}: -1")
    
    print(f"\nNext greater mapping: {next_greater}")
    
    result = [next_greater[num] for num in nums1]
    print(f"Result for nums1: {result}")


def benchmark_next_greater_element_i():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Stack Approach", NextGreaterElementI().nextGreaterElement_stack_approach),
        ("Brute Force", NextGreaterElementI().nextGreaterElement_brute_force),
        ("Reverse Iteration", NextGreaterElementI().nextGreaterElement_reverse_iteration),
    ]
    
    # Generate test data
    sizes = [(100, 50), (1000, 500), (5000, 2500)]
    
    print("\n=== Next Greater Element I Performance Benchmark ===")
    
    for n2_size, n1_size in sizes:
        print(f"\n--- nums2 size: {n2_size}, nums1 size: {n1_size} ---")
        
        # Generate nums2
        nums2 = list(range(1, n2_size + 1))
        random.shuffle(nums2)
        
        # Generate nums1 as subset of nums2
        nums1 = random.sample(nums2, n1_size)
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(nums1, nums2)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def visualize_next_greater_concept():
    """Visualize the next greater element concept"""
    print("\n=== Next Greater Element Concept Visualization ===")
    
    nums2 = [1, 3, 4, 2, 5]
    
    print(f"Array: {nums2}")
    print("Finding next greater elements:")
    print()
    
    for i, num in enumerate(nums2):
        print(f"Element {num} at index {i}:")
        
        # Find next greater
        next_greater = -1
        for j in range(i + 1, len(nums2)):
            if nums2[j] > num:
                next_greater = nums2[j]
                break
        
        # Visualize
        vis = [' '] * len(nums2)
        vis[i] = '█'  # Current element
        
        if next_greater != -1:
            next_pos = nums2.index(next_greater, i + 1)
            vis[next_pos] = '▲'  # Next greater element
            
            # Show arrow
            for k in range(i + 1, next_pos):
                vis[k] = '-'
        
        vis_str = ''.join(vis)
        print(f"  {vis_str}")
        print(f"  Next greater: {next_greater}")
        print()


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = NextGreaterElementI()
    
    edge_cases = [
        ([], [1, 2, 3], [], "Empty nums1"),
        ([1], [1], [-1], "Single element, no next greater"),
        ([1], [1, 2], [2], "Single element, has next greater"),
        ([1, 2, 3], [1, 2, 3], [2, 3, -1], "Same arrays"),
        ([3, 2, 1], [1, 2, 3], [3, 3, -1], "Reverse order"),
        ([5, 5], [5, 6, 5, 7], [6, 6], "Duplicate elements"),
        ([10], [1, 2, 3, 10], [-1], "Largest element"),
    ]
    
    for nums1, nums2, expected, description in edge_cases:
        try:
            result = solver.nextGreaterElement_stack_approach(nums1, nums2)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | nums1: {nums1}, nums2: {nums2} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_stack_vs_brute_force():
    """Compare stack vs brute force approaches"""
    print("\n=== Stack vs Brute Force Comparison ===")
    
    test_cases = [
        ([1, 2, 3], [3, 2, 1, 4]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6]),
        ([10, 5, 15], [5, 10, 15, 20]),
    ]
    
    solver = NextGreaterElementI()
    
    for i, (nums1, nums2) in enumerate(test_cases):
        print(f"\nTest case {i+1}: nums1={nums1}, nums2={nums2}")
        
        # Stack approach
        result_stack = solver.nextGreaterElement_stack_approach(nums1, nums2)
        print(f"Stack approach:  {result_stack}")
        
        # Brute force approach
        result_brute = solver.nextGreaterElement_brute_force(nums1, nums2)
        print(f"Brute force:     {result_brute}")
        
        # Check consistency
        consistent = result_stack == result_brute
        print(f"Results match:   {'✓' if consistent else '✗'}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Approach", "O(n + m)", "O(n)", "Optimal for most cases"),
        ("Brute Force", "O(m * n)", "O(1)", "Simple but inefficient"),
        ("HashMap Precompute", "O(n²) + O(m)", "O(n)", "Expensive preprocessing"),
        ("Reverse Iteration", "O(n + m)", "O(n)", "Alternative optimal approach"),
        ("Two Pointers", "O(m * n)", "O(1)", "Space efficient but slow"),
        ("Optimized Lookup", "O(n + m)", "O(n)", "Fast with position mapping"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<15} | {'Space':<8} | {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<15} | {space_comp:<8} | {notes}")


def demonstrate_monotonic_stack_property():
    """Demonstrate monotonic stack property"""
    print("\n=== Monotonic Stack Property Demonstration ===")
    
    nums = [2, 1, 2, 4, 3, 1]
    
    print(f"Processing array: {nums}")
    print("Monotonic stack maintains decreasing order:")
    print()
    
    stack = []
    next_greater = {}
    
    for i, num in enumerate(nums):
        print(f"Step {i+1}: Processing {num}")
        print(f"  Stack before: {stack}")
        
        # Show what gets popped and why
        popped_elements = []
        while stack and stack[-1] < num:
            popped = stack.pop()
            popped_elements.append(popped)
            next_greater[popped] = num
        
        if popped_elements:
            print(f"  Popped {popped_elements} (smaller than {num})")
        
        stack.append(num)
        print(f"  Stack after: {stack}")
        print(f"  Stack property: {'Decreasing' if all(stack[i] >= stack[i+1] for i in range(len(stack)-1)) else 'Not monotonic'}")
        print()
    
    # Handle remaining elements
    for remaining in stack:
        next_greater[remaining] = -1
    
    print(f"Final next greater mapping: {next_greater}")


if __name__ == "__main__":
    test_next_greater_element_i()
    demonstrate_stack_approach()
    visualize_next_greater_concept()
    demonstrate_monotonic_stack_property()
    test_edge_cases()
    compare_stack_vs_brute_force()
    analyze_time_complexity()
    benchmark_next_greater_element_i()

"""
Next Greater Element I demonstrates the fundamental monotonic stack pattern
for finding next greater elements with multiple implementation approaches
and comprehensive analysis of the monotonic stack data structure.
"""
