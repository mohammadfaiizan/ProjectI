"""
1124. Longest Well-Performing Interval - Multiple Approaches
Difficulty: Medium

We are given hours, a list of the number of hours worked per day for a given employee.

A day is considered to be a tiring day if and only if the number of hours worked is (strictly) greater than 8.

A well-performing interval is an interval of days for which the number of tiring days is strictly larger than the number of non-tiring days.

Return the length of the longest well-performing interval.
"""

from typing import List, Dict

class LongestWellPerformingInterval:
    """Multiple approaches to find longest well-performing interval"""
    
    def longestWPI_stack_approach(self, hours: List[int]) -> int:
        """
        Approach 1: Monotonic Stack (Optimal)
        
        Convert to prefix sum problem and use stack to find longest subarray.
        
        Time: O(n), Space: O(n)
        """
        n = len(hours)
        # Convert hours to +1 (tiring) or -1 (non-tiring)
        score = [1 if h > 8 else -1 for h in hours]
        
        # Calculate prefix sums
        prefix = [0]
        for s in score:
            prefix.append(prefix[-1] + s)
        
        # Use stack to find longest subarray with positive sum
        stack = []  # Store indices of prefix sums
        max_length = 0
        
        # Build decreasing stack
        for i in range(len(prefix)):
            if not stack or prefix[stack[-1]] > prefix[i]:
                stack.append(i)
        
        # Find longest subarray
        for i in range(len(prefix) - 1, -1, -1):
            while stack and prefix[stack[-1]] < prefix[i]:
                j = stack.pop()
                max_length = max(max_length, i - j)
        
        return max_length
    
    def longestWPI_hashmap_approach(self, hours: List[int]) -> int:
        """
        Approach 2: HashMap with Prefix Sum
        
        Use hashmap to track first occurrence of each prefix sum.
        
        Time: O(n), Space: O(n)
        """
        score = 0
        max_length = 0
        first_occurrence = {}  # prefix_sum -> first index
        
        for i, h in enumerate(hours):
            # Update score
            score += 1 if h > 8 else -1
            
            if score > 0:
                # Entire prefix is well-performing
                max_length = i + 1
            else:
                # Look for previous occurrence of (score - 1)
                if score - 1 in first_occurrence:
                    length = i - first_occurrence[score - 1]
                    max_length = max(max_length, length)
                
                # Record first occurrence of current score
                if score not in first_occurrence:
                    first_occurrence[score] = i
        
        return max_length
    
    def longestWPI_brute_force(self, hours: List[int]) -> int:
        """
        Approach 3: Brute Force
        
        Check all possible subarrays.
        
        Time: O(n²), Space: O(1)
        """
        n = len(hours)
        max_length = 0
        
        for i in range(n):
            tiring_days = 0
            non_tiring_days = 0
            
            for j in range(i, n):
                if hours[j] > 8:
                    tiring_days += 1
                else:
                    non_tiring_days += 1
                
                if tiring_days > non_tiring_days:
                    max_length = max(max_length, j - i + 1)
        
        return max_length
    
    def longestWPI_prefix_sum_optimized(self, hours: List[int]) -> int:
        """
        Approach 4: Optimized Prefix Sum
        
        Use prefix sum with optimized search.
        
        Time: O(n), Space: O(n)
        """
        n = len(hours)
        prefix = [0]
        
        # Build prefix sum array
        for h in hours:
            prefix.append(prefix[-1] + (1 if h > 8 else -1))
        
        max_length = 0
        seen = {}  # prefix_sum -> earliest index
        
        for i in range(len(prefix)):
            if prefix[i] not in seen:
                seen[prefix[i]] = i
            
            # For positive prefix sum, entire prefix is valid
            if prefix[i] > 0:
                max_length = max(max_length, i)
            
            # Look for prefix[i] - 1 to find valid subarray
            if prefix[i] - 1 in seen:
                length = i - seen[prefix[i] - 1]
                max_length = max(max_length, length)
        
        return max_length
    
    def longestWPI_sliding_window(self, hours: List[int]) -> int:
        """
        Approach 5: Modified Sliding Window
        
        Use sliding window approach with score tracking.
        
        Time: O(n²), Space: O(1)
        """
        n = len(hours)
        max_length = 0
        
        for start in range(n):
            score = 0
            
            for end in range(start, n):
                # Update score for current window
                score += 1 if hours[end] > 8 else -1
                
                # Check if current window is well-performing
                if score > 0:
                    max_length = max(max_length, end - start + 1)
        
        return max_length
    
    def longestWPI_two_pointers(self, hours: List[int]) -> int:
        """
        Approach 6: Two Pointers with Prefix Sum
        
        Use two pointers to find longest valid interval.
        
        Time: O(n²), Space: O(n)
        """
        n = len(hours)
        # Convert to +1/-1 array
        score = [1 if h > 8 else -1 for h in hours]
        
        # Calculate prefix sums
        prefix = [0]
        for s in score:
            prefix.append(prefix[-1] + s)
        
        max_length = 0
        
        # Try all possible starting points
        for i in range(n):
            for j in range(i + 1, n + 1):
                if prefix[j] - prefix[i] > 0:
                    max_length = max(max_length, j - i)
        
        return max_length
    
    def longestWPI_divide_conquer(self, hours: List[int]) -> int:
        """
        Approach 7: Divide and Conquer
        
        Use divide and conquer to find longest interval.
        
        Time: O(n log n), Space: O(log n)
        """
        def solve(left: int, right: int, scores: List[int]) -> int:
            if left >= right:
                return 0
            
            mid = (left + right) // 2
            
            # Find longest interval crossing the middle
            max_cross = 0
            
            # Extend to the left
            left_sum = 0
            min_left_sum = 0
            left_pos = mid
            
            for i in range(mid, left - 1, -1):
                left_sum += scores[i]
                if left_sum < min_left_sum:
                    min_left_sum = left_sum
                    left_pos = i
            
            # Extend to the right
            right_sum = 0
            for i in range(mid + 1, right):
                right_sum += scores[i]
                total_sum = left_sum + right_sum
                if total_sum > 0:
                    max_cross = max(max_cross, i - left_pos + 1)
            
            # Recursively solve left and right parts
            left_max = solve(left, mid, scores)
            right_max = solve(mid + 1, right, scores)
            
            return max(max_cross, left_max, right_max)
        
        # Convert hours to scores
        scores = [1 if h > 8 else -1 for h in hours]
        return solve(0, len(scores), scores)


def test_longest_well_performing_interval():
    """Test longest well-performing interval algorithms"""
    solver = LongestWellPerformingInterval()
    
    test_cases = [
        ([9,9,6,0,6,6,9], 3, "Example 1"),
        ([6,6,6], 0, "All non-tiring"),
        ([9,9,9], 3, "All tiring"),
        ([9,6,9,6,9], 5, "Alternating pattern"),
        ([6,9,9], 2, "Tiring at end"),
        ([9,9,6], 2, "Non-tiring at end"),
        ([6], 0, "Single non-tiring"),
        ([9], 1, "Single tiring"),
        ([9,6,6,9,9,6,9], 5, "Complex pattern"),
        ([6,9,6,9,6], 3, "Mixed pattern"),
    ]
    
    algorithms = [
        ("Stack Approach", solver.longestWPI_stack_approach),
        ("HashMap Approach", solver.longestWPI_hashmap_approach),
        ("Brute Force", solver.longestWPI_brute_force),
        ("Prefix Sum Optimized", solver.longestWPI_prefix_sum_optimized),
        ("Sliding Window", solver.longestWPI_sliding_window),
        ("Two Pointers", solver.longestWPI_two_pointers),
    ]
    
    print("=== Testing Longest Well-Performing Interval ===")
    
    for hours, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Hours: {hours}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(hours)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_problem_transformation():
    """Demonstrate problem transformation"""
    print("\n=== Problem Transformation Demonstration ===")
    
    hours = [9, 9, 6, 0, 6, 6, 9]
    print(f"Original hours: {hours}")
    
    # Transform to +1/-1 based on > 8 hours
    scores = [1 if h > 8 else -1 for h in hours]
    print(f"Transformed scores: {scores}")
    print("  +1 = tiring day (> 8 hours)")
    print("  -1 = non-tiring day (<= 8 hours)")
    
    # Calculate prefix sums
    prefix = [0]
    for score in scores:
        prefix.append(prefix[-1] + score)
    
    print(f"Prefix sums: {prefix}")
    print("  prefix[i] = sum of scores from index 0 to i-1")
    
    print("\nWell-performing interval = subarray with sum > 0")
    print("This means: tiring days > non-tiring days")
    
    # Show some intervals
    print("\nExample intervals:")
    for i in range(len(hours)):
        for j in range(i + 1, len(hours) + 1):
            interval_sum = prefix[j] - prefix[i]
            if interval_sum > 0:
                interval_hours = hours[i:j]
                tiring = sum(1 for h in interval_hours if h > 8)
                non_tiring = len(interval_hours) - tiring
                print(f"  Interval [{i}:{j}] = {interval_hours}")
                print(f"    Tiring: {tiring}, Non-tiring: {non_tiring}, Length: {j-i}")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    hours = [9, 9, 6, 0, 6, 6, 9]
    print(f"Hours: {hours}")
    
    # Transform and calculate prefix sums
    scores = [1 if h > 8 else -1 for h in hours]
    prefix = [0]
    for score in scores:
        prefix.append(prefix[-1] + score)
    
    print(f"Scores: {scores}")
    print(f"Prefix sums: {prefix}")
    
    # Build decreasing stack
    print("\nBuilding decreasing stack:")
    stack = []
    
    for i in range(len(prefix)):
        print(f"  Step {i+1}: prefix[{i}] = {prefix[i]}")
        
        if not stack or prefix[stack[-1]] > prefix[i]:
            stack.append(i)
            print(f"    Added index {i} to stack: {[f'{idx}({prefix[idx]})' for idx in stack]}")
        else:
            print(f"    Skipped (not decreasing)")
    
    print(f"\nFinal decreasing stack: {[f'{idx}({prefix[idx]})' for idx in stack]}")
    
    # Find longest subarray
    print("\nFinding longest well-performing interval:")
    max_length = 0
    
    for i in range(len(prefix) - 1, -1, -1):
        print(f"\n  Processing prefix[{i}] = {prefix[i]}")
        
        popped = []
        while stack and prefix[stack[-1]] < prefix[i]:
            j = stack.pop()
            length = i - j
            max_length = max(max_length, length)
            popped.append(f'{j}({prefix[j]})')
            print(f"    Popped index {j}: interval [{j}:{i}] has length {length}")
        
        if popped:
            print(f"    Popped: {popped}")
            print(f"    Max length so far: {max_length}")
    
    print(f"\nFinal result: {max_length}")


def demonstrate_hashmap_approach():
    """Demonstrate hashmap approach step by step"""
    print("\n=== HashMap Approach Step-by-Step Demo ===")
    
    hours = [9, 9, 6, 0, 6, 6, 9]
    print(f"Hours: {hours}")
    
    score = 0
    max_length = 0
    first_occurrence = {}
    
    print("\nProcessing each day:")
    
    for i, h in enumerate(hours):
        print(f"\nDay {i}: {h} hours")
        
        # Update score
        old_score = score
        score += 1 if h > 8 else -1
        print(f"  Score change: {old_score} -> {score}")
        
        if score > 0:
            # Entire prefix is well-performing
            max_length = i + 1
            print(f"  Entire prefix is well-performing: length {max_length}")
        else:
            # Look for previous occurrence of (score - 1)
            target = score - 1
            print(f"  Looking for previous occurrence of {target}")
            
            if target in first_occurrence:
                length = i - first_occurrence[target]
                max_length = max(max_length, length)
                print(f"  Found at index {first_occurrence[target]}: interval length {length}")
                print(f"  Max length updated to: {max_length}")
            else:
                print(f"  Not found")
            
            # Record first occurrence of current score
            if score not in first_occurrence:
                first_occurrence[score] = i
                print(f"  Recorded first occurrence of {score} at index {i}")
        
        print(f"  First occurrences: {first_occurrence}")
        print(f"  Max length so far: {max_length}")
    
    print(f"\nFinal result: {max_length}")


def visualize_well_performing_intervals():
    """Visualize well-performing intervals"""
    print("\n=== Well-Performing Intervals Visualization ===")
    
    hours = [9, 9, 6, 0, 6, 6, 9]
    print(f"Hours: {hours}")
    
    # Show tiring vs non-tiring days
    print("\nDay classification:")
    for i, h in enumerate(hours):
        day_type = "Tiring" if h > 8 else "Non-tiring"
        print(f"  Day {i}: {h} hours -> {day_type}")
    
    print("\nChecking all possible intervals:")
    
    max_length = 0
    best_intervals = []
    
    for i in range(len(hours)):
        for j in range(i + 1, len(hours) + 1):
            interval = hours[i:j]
            tiring = sum(1 for h in interval if h > 8)
            non_tiring = len(interval) - tiring
            is_well_performing = tiring > non_tiring
            
            if is_well_performing:
                length = j - i
                if length > max_length:
                    max_length = length
                    best_intervals = [(i, j)]
                elif length == max_length:
                    best_intervals.append((i, j))
                
                print(f"  [{i}:{j}] = {interval} -> Tiring: {tiring}, Non-tiring: {non_tiring} ✓ (Length: {length})")
            else:
                print(f"  [{i}:{j}] = {interval} -> Tiring: {tiring}, Non-tiring: {non_tiring} ✗")
    
    print(f"\nLongest well-performing intervals (length {max_length}):")
    for start, end in best_intervals:
        interval = hours[start:end]
        print(f"  [{start}:{end}] = {interval}")


def benchmark_longest_well_performing():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Stack Approach", LongestWellPerformingInterval().longestWPI_stack_approach),
        ("HashMap Approach", LongestWellPerformingInterval().longestWPI_hashmap_approach),
        ("Brute Force", LongestWellPerformingInterval().longestWPI_brute_force),
        ("Prefix Sum Optimized", LongestWellPerformingInterval().longestWPI_prefix_sum_optimized),
    ]
    
    # Generate test arrays of different sizes
    def generate_hours(size: int) -> List[int]:
        return [random.randint(0, 12) for _ in range(size)]
    
    test_sizes = [100, 1000, 5000]
    
    print("\n=== Longest Well-Performing Interval Performance Benchmark ===")
    
    for size in test_sizes:
        print(f"\n--- Array Size: {size} ---")
        
        hours = generate_hours(size)
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(hours)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = LongestWellPerformingInterval()
    
    edge_cases = [
        ([], 0, "Empty array"),
        ([6], 0, "Single non-tiring day"),
        ([9], 1, "Single tiring day"),
        ([6, 6], 0, "Two non-tiring days"),
        ([9, 9], 2, "Two tiring days"),
        ([6, 9], 1, "Non-tiring then tiring"),
        ([9, 6], 1, "Tiring then non-tiring"),
        ([8], 0, "Exactly 8 hours"),
        ([0, 12], 1, "Extreme values"),
        ([9, 6, 9], 3, "Tiring-Non-Tiring-Tiring"),
    ]
    
    for hours, expected, description in edge_cases:
        try:
            result = solver.longestWPI_stack_approach(hours)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | hours: {hours} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        [9, 9, 6, 0, 6, 6, 9],
        [6, 6, 6],
        [9, 9, 9],
        [9, 6, 9, 6, 9],
    ]
    
    solver = LongestWellPerformingInterval()
    
    approaches = [
        ("Stack", solver.longestWPI_stack_approach),
        ("HashMap", solver.longestWPI_hashmap_approach),
        ("Brute Force", solver.longestWPI_brute_force),
        ("Prefix Sum", solver.longestWPI_prefix_sum_optimized),
    ]
    
    for i, hours in enumerate(test_cases):
        print(f"\nTest case {i+1}: {hours}")
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(hours)
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
        ("Stack Approach", "O(n)", "O(n)", "Monotonic stack with prefix sums"),
        ("HashMap Approach", "O(n)", "O(n)", "Single pass with hashmap lookup"),
        ("Brute Force", "O(n²)", "O(1)", "Check all possible subarrays"),
        ("Prefix Sum Optimized", "O(n)", "O(n)", "Prefix sum with optimized search"),
        ("Sliding Window", "O(n²)", "O(1)", "Modified sliding window approach"),
        ("Two Pointers", "O(n²)", "O(n)", "Two pointers with prefix sum"),
        ("Divide Conquer", "O(n log n)", "O(log n)", "Recursive divide and conquer"),
    ]
    
    print(f"{'Approach':<25} | {'Time':<12} | {'Space':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<25} | {time_comp:<12} | {space_comp:<8} | {notes}")


def demonstrate_prefix_sum_insight():
    """Demonstrate prefix sum insight"""
    print("\n=== Prefix Sum Insight Demonstration ===")
    
    hours = [9, 9, 6, 0, 6, 6, 9]
    print(f"Hours: {hours}")
    
    # Transform problem
    print("\nProblem transformation:")
    print("1. Convert hours to +1 (tiring) or -1 (non-tiring)")
    print("2. Find longest subarray with positive sum")
    print("3. Positive sum means: tiring days > non-tiring days")
    
    scores = [1 if h > 8 else -1 for h in hours]
    print(f"\nScores: {scores}")
    
    # Show prefix sums
    prefix = [0]
    for score in scores:
        prefix.append(prefix[-1] + score)
    
    print(f"Prefix sums: {prefix}")
    
    print("\nKey insight:")
    print("- For subarray [i, j], sum = prefix[j+1] - prefix[i]")
    print("- We want: prefix[j+1] - prefix[i] > 0")
    print("- This means: prefix[j+1] > prefix[i]")
    
    print("\nFinding longest subarray with positive sum:")
    for i in range(len(prefix)):
        for j in range(i + 1, len(prefix)):
            if prefix[j] > prefix[i]:
                length = j - i
                subarray = hours[i:j]
                print(f"  prefix[{j}] > prefix[{i}] ({prefix[j]} > {prefix[i]})")
                print(f"    Subarray [{i}:{j}] = {subarray}, length = {length}")


if __name__ == "__main__":
    test_longest_well_performing_interval()
    demonstrate_problem_transformation()
    demonstrate_stack_approach()
    demonstrate_hashmap_approach()
    visualize_well_performing_intervals()
    demonstrate_prefix_sum_insight()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_longest_well_performing()

"""
Longest Well-Performing Interval demonstrates monotonic stack applications
for interval optimization problems, including problem transformation techniques
and multiple approaches for finding optimal subarrays with constraints.
"""
