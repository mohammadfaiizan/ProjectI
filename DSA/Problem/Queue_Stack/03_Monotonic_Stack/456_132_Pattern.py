"""
456. 132 Pattern - Multiple Approaches
Difficulty: Medium

Given an array of n integers nums, a 132 pattern is a subsequence of three integers nums[i], nums[j], nums[k] such that i < j < k and nums[i] < nums[k] < nums[j].

Return true if there is a 132 pattern in nums, false otherwise.
"""

from typing import List

class Pattern132:
    """Multiple approaches to find 132 pattern"""
    
    def find132pattern_stack_approach(self, nums: List[int]) -> bool:
        """
        Approach 1: Monotonic Stack (Optimal)
        
        Use stack to track potential middle elements and find pattern.
        
        Time: O(n), Space: O(n)
        """
        if len(nums) < 3:
            return False
        
        stack = []  # Store potential j values (middle elements)
        third = float('-inf')  # The k value (third element)
        
        # Traverse from right to left
        for i in range(len(nums) - 1, -1, -1):
            # If current element is smaller than third, we found 132 pattern
            if nums[i] < third:
                return True
            
            # Pop elements smaller than current from stack
            # These become potential third elements
            while stack and stack[-1] < nums[i]:
                third = stack.pop()
            
            # Current element becomes potential middle element
            stack.append(nums[i])
        
        return False
    
    def find132pattern_brute_force(self, nums: List[int]) -> bool:
        """
        Approach 2: Brute Force
        
        Check all possible triplets.
        
        Time: O(n³), Space: O(1)
        """
        n = len(nums)
        
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    if nums[i] < nums[k] < nums[j]:
                        return True
        
        return False
    
    def find132pattern_optimized_brute_force(self, nums: List[int]) -> bool:
        """
        Approach 3: Optimized Brute Force
        
        For each j, find minimum on left and check right side.
        
        Time: O(n²), Space: O(1)
        """
        n = len(nums)
        
        for j in range(1, n - 1):
            # Find minimum element to the left of j
            min_left = float('inf')
            for i in range(j):
                min_left = min(min_left, nums[i])
            
            # If no element smaller than nums[j] on left, continue
            if min_left >= nums[j]:
                continue
            
            # Find element on right that satisfies pattern
            for k in range(j + 1, n):
                if min_left < nums[k] < nums[j]:
                    return True
        
        return False
    
    def find132pattern_precompute_min(self, nums: List[int]) -> bool:
        """
        Approach 4: Precompute Minimum Array
        
        Precompute minimum elements to the left for each position.
        
        Time: O(n²), Space: O(n)
        """
        n = len(nums)
        if n < 3:
            return False
        
        # Precompute minimum to the left of each position
        min_left = [0] * n
        min_left[0] = nums[0]
        
        for i in range(1, n):
            min_left[i] = min(min_left[i-1], nums[i])
        
        # For each middle element, check right side
        for j in range(1, n - 1):
            if min_left[j-1] >= nums[j]:
                continue
            
            for k in range(j + 1, n):
                if min_left[j-1] < nums[k] < nums[j]:
                    return True
        
        return False
    
    def find132pattern_stack_with_min(self, nums: List[int]) -> bool:
        """
        Approach 5: Stack with Minimum Tracking
        
        Use stack to track candidates and minimum values.
        
        Time: O(n), Space: O(n)
        """
        n = len(nums)
        if n < 3:
            return False
        
        stack = []  # Store (value, min_before) pairs
        min_so_far = nums[0]
        
        for i in range(1, n):
            # Pop elements that can't form valid pattern
            while stack and stack[-1][0] <= nums[i]:
                stack.pop()
            
            # Check if current element can be the third in pattern
            if stack and stack[-1][1] < nums[i]:
                return True
            
            # Add current element with its minimum predecessor
            stack.append((nums[i], min_so_far))
            min_so_far = min(min_so_far, nums[i])
        
        return False
    
    def find132pattern_interval_approach(self, nums: List[int]) -> bool:
        """
        Approach 6: Interval-based Approach
        
        Track intervals where pattern might exist.
        
        Time: O(n), Space: O(n)
        """
        intervals = []  # Store (min, max) intervals
        
        for num in nums:
            # Remove intervals where num is too large to be first element
            intervals = [(min_val, max_val) for min_val, max_val in intervals if num <= min_val or num >= max_val]
            
            # Check if num can be the middle element of existing intervals
            for min_val, max_val in intervals:
                if min_val < num < max_val:
                    return True
            
            # Merge intervals and add new one
            if intervals and num < intervals[-1][0]:
                intervals[-1] = (num, intervals[-1][1])
            else:
                intervals.append((num, num))
        
        return False
    
    def find132pattern_divide_conquer(self, nums: List[int]) -> bool:
        """
        Approach 7: Divide and Conquer
        
        Recursively check left, right, and cross patterns.
        
        Time: O(n log n), Space: O(log n)
        """
        def find_pattern(left: int, right: int) -> bool:
            if right - left < 2:
                return False
            
            # Check pattern in left half
            mid = (left + right) // 2
            if find_pattern(left, mid) or find_pattern(mid, right):
                return True
            
            # Check cross pattern
            min_left = min(nums[left:mid])
            max_left = max(nums[left:mid])
            min_right = min(nums[mid:right])
            max_right = max(nums[mid:right])
            
            # Pattern: left_min < right_val < left_max
            for i in range(mid, right):
                if min_left < nums[i] < max_left:
                    return True
            
            return False
        
        return find_pattern(0, len(nums))


def test_132_pattern():
    """Test 132 pattern algorithms"""
    solver = Pattern132()
    
    test_cases = [
        ([1,2,3,4], False, "Increasing sequence"),
        ([3,1,4,2], True, "Example with pattern"),
        ([-1,3,2,0], True, "Pattern with negatives"),
        ([1,0,1,-4,-3], False, "No pattern"),
        ([3,5,0,3,4], True, "Complex pattern"),
        ([1,3,2], True, "Simple 132"),
        ([2,1,3], False, "No pattern"),
        ([1,4,0,2,3], True, "Multiple candidates"),
        ([], False, "Empty array"),
        ([1], False, "Single element"),
        ([1,2], False, "Two elements"),
        ([4,3,2,1], False, "Decreasing sequence"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.find132pattern_stack_approach),
        ("Brute Force", solver.find132pattern_brute_force),
        ("Optimized Brute Force", solver.find132pattern_optimized_brute_force),
        ("Precompute Min", solver.find132pattern_precompute_min),
        ("Stack with Min", solver.find132pattern_stack_with_min),
        ("Interval Approach", solver.find132pattern_interval_approach),
    ]
    
    print("=== Testing 132 Pattern ===")
    
    for nums, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    nums = [3, 1, 4, 2]
    print(f"Array: {nums}")
    print("Processing from right to left to find 132 pattern:")
    
    stack = []
    third = float('-inf')
    
    for i in range(len(nums) - 1, -1, -1):
        print(f"\nStep {len(nums) - i}: Processing nums[{i}] = {nums[i]}")
        print(f"  Stack before: {stack}")
        print(f"  Third (k) value: {third}")
        
        # Check if we found 132 pattern
        if nums[i] < third:
            print(f"  Found 132 pattern! {nums[i]} < {third}")
            print(f"  Pattern: nums[{i}] < k < j where k={third}")
            return True
        
        # Pop smaller elements to update third
        popped = []
        while stack and stack[-1] < nums[i]:
            third = stack.pop()
            popped.append(third)
        
        if popped:
            print(f"  Popped {popped} from stack, updated third to {third}")
        
        stack.append(nums[i])
        print(f"  Stack after: {stack}")
    
    print(f"\nNo 132 pattern found")
    return False


def demonstrate_pattern_concept():
    """Demonstrate 132 pattern concept"""
    print("\n=== 132 Pattern Concept Demonstration ===")
    
    print("132 Pattern: nums[i] < nums[k] < nums[j] where i < j < k")
    print("In other words: first < third < second (positionally)")
    
    examples = [
        ([1, 3, 2], "Simple 132: 1 < 2 < 3"),
        ([3, 1, 4, 2], "Pattern: 1 < 2 < 4"),
        ([1, 2, 3, 4], "No pattern (increasing)"),
        ([4, 3, 2, 1], "No pattern (decreasing)"),
        ([3, 5, 0, 3, 4], "Pattern: 0 < 3 < 5"),
    ]
    
    for nums, description in examples:
        print(f"\nArray: {nums}")
        print(f"Analysis: {description}")
        
        # Find actual pattern if exists
        found_pattern = False
        for i in range(len(nums) - 2):
            for j in range(i + 1, len(nums) - 1):
                for k in range(j + 1, len(nums)):
                    if nums[i] < nums[k] < nums[j]:
                        print(f"Found pattern: nums[{i}]={nums[i]} < nums[{k}]={nums[k]} < nums[{j}]={nums[j]}")
                        found_pattern = True
                        break
                if found_pattern:
                    break
            if found_pattern:
                break
        
        if not found_pattern:
            print("No 132 pattern exists")


def visualize_stack_process():
    """Visualize stack processing"""
    print("\n=== Stack Process Visualization ===")
    
    nums = [3, 1, 4, 2]
    print(f"Array: {nums}")
    print("Processing right to left:")
    
    # Show the array with indices
    print("\nArray visualization:")
    print("Index:", " ".join(f"{i:2}" for i in range(len(nums))))
    print("Value:", " ".join(f"{v:2}" for v in nums))
    print("Order: ←←←← (right to left)")
    
    stack = []
    third = float('-inf')
    
    for i in range(len(nums) - 1, -1, -1):
        print(f"\n--- Processing index {i} (value {nums[i]}) ---")
        
        # Show current state
        print(f"Current element: {nums[i]}")
        print(f"Stack (potential j values): {stack}")
        print(f"Third (k value): {third}")
        
        # Check for pattern
        if nums[i] < third:
            print(f"✓ PATTERN FOUND: {nums[i]} < {third} < (some j in stack)")
            break
        
        # Update stack and third
        while stack and stack[-1] < nums[i]:
            third = stack.pop()
            print(f"  Popped {third} from stack → new third = {third}")
        
        stack.append(nums[i])
        print(f"  Added {nums[i]} to stack")
        print(f"  New stack: {stack}")


def benchmark_132_pattern():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Stack Approach", Pattern132().find132pattern_stack_approach),
        ("Optimized Brute Force", Pattern132().find132pattern_optimized_brute_force),
        ("Precompute Min", Pattern132().find132pattern_precompute_min),
        ("Stack with Min", Pattern132().find132pattern_stack_with_min),
    ]
    
    # Test with different array sizes
    sizes = [100, 1000, 5000]
    
    print("\n=== 132 Pattern Performance Benchmark ===")
    
    for size in sizes:
        print(f"\n--- Array Size: {size} ---")
        
        # Generate random array
        nums = [random.randint(-100, 100) for _ in range(size)]
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(nums)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Found: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = Pattern132()
    
    edge_cases = [
        ([], False, "Empty array"),
        ([1], False, "Single element"),
        ([1, 2], False, "Two elements"),
        ([1, 2, 3], False, "Increasing triplet"),
        ([3, 2, 1], False, "Decreasing triplet"),
        ([1, 3, 2], True, "Perfect 132"),
        ([2, 1, 3], False, "123 pattern"),
        ([0, 0, 0], False, "All same"),
        ([-1, 0, 1], False, "Increasing with negatives"),
        ([1, -1, 0], True, "132 with negatives"),
        ([100, 1, 50], True, "Large numbers"),
    ]
    
    for nums, expected, description in edge_cases:
        try:
            result = solver.find132pattern_stack_approach(nums)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | nums: {nums} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def analyze_pattern_properties():
    """Analyze properties of 132 pattern"""
    print("\n=== 132 Pattern Properties Analysis ===")
    
    print("Key insights about 132 pattern:")
    print("1. Need at least 3 elements")
    print("2. First element must be smallest")
    print("3. Second element must be largest")
    print("4. Third element must be in between")
    print("5. Positions: i < j < k")
    print("6. Values: nums[i] < nums[k] < nums[j]")
    
    print("\nWhy stack approach works:")
    print("- Process right to left")
    print("- Stack maintains potential 'j' values (largest)")
    print("- 'third' tracks potential 'k' values (middle)")
    print("- When nums[i] < third, we found the pattern")
    
    print("\nStack invariant:")
    print("- Stack is monotonically increasing")
    print("- Elements popped become candidates for 'k'")
    print("- Current element becomes candidate for 'j'")


def demonstrate_why_right_to_left():
    """Demonstrate why we process right to left"""
    print("\n=== Why Process Right to Left? ===")
    
    nums = [3, 1, 4, 2]
    print(f"Array: {nums}")
    
    print("\nIf we process left to right:")
    print("- We see 3 first, but don't know if it's the largest")
    print("- We see 1 next, could be the smallest")
    print("- We see 4, could be largest, but what about the middle?")
    print("- Hard to track all possibilities")
    
    print("\nProcessing right to left:")
    print("- We maintain stack of potential 'j' (largest) values")
    print("- When we pop from stack, those become 'k' (middle) candidates")
    print("- Current element is potential 'i' (smallest)")
    print("- If current < any previous 'k', we found pattern")
    
    print(f"\nFor {nums}:")
    print("Step 1: See 2, stack=[2], third=-inf")
    print("Step 2: See 4, 4>2 so stack=[4], third=2")
    print("Step 3: See 1, 1<2 (third) → Found pattern!")
    print("Pattern: 1 < 2 < 4")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        [3, 1, 4, 2],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [1, 3, 2, 4, 0, 5],
    ]
    
    solver = Pattern132()
    
    approaches = [
        ("Stack", solver.find132pattern_stack_approach),
        ("Brute Force", solver.find132pattern_brute_force),
        ("Precompute Min", solver.find132pattern_precompute_min),
        ("Stack with Min", solver.find132pattern_stack_with_min),
    ]
    
    for i, nums in enumerate(test_cases):
        print(f"\nTest case {i+1}: {nums}")
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(nums)
                results[name] = result
                print(f"{name:15} | Result: {result}")
            except Exception as e:
                print(f"{name:15} | ERROR: {str(e)[:40]}")
        
        # Check consistency
        if results:
            first_result = list(results.values())[0]
            all_same = all(result == first_result for result in results.values())
            print(f"All approaches agree: {'✓' if all_same else '✗'}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Stack Approach", "O(n)", "O(n)", "Optimal - each element processed once"),
        ("Brute Force", "O(n³)", "O(1)", "Check all triplets"),
        ("Optimized Brute Force", "O(n²)", "O(1)", "Fix middle, scan left/right"),
        ("Precompute Min", "O(n²)", "O(n)", "Precompute + scan right"),
        ("Stack with Min", "O(n)", "O(n)", "Alternative optimal approach"),
        ("Interval Approach", "O(n)", "O(n)", "Track valid intervals"),
        ("Divide Conquer", "O(n log n)", "O(log n)", "Recursive splitting"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<8} | {space_comp:<8} | {notes}")


if __name__ == "__main__":
    test_132_pattern()
    demonstrate_pattern_concept()
    demonstrate_stack_approach()
    visualize_stack_process()
    demonstrate_why_right_to_left()
    test_edge_cases()
    compare_approaches()
    analyze_pattern_properties()
    analyze_time_complexity()
    benchmark_132_pattern()

"""
132 Pattern demonstrates advanced monotonic stack techniques for
pattern recognition in arrays, including multiple approaches for
finding specific subsequence patterns with optimal time complexity.
"""
