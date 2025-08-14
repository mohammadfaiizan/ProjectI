"""
Greedy Mathematical Problems
===========================

Topics: Number theory, combinatorics, mathematical optimization
Companies: Google, Amazon, Microsoft, Facebook, Apple, Quant firms
Difficulty: Medium to Hard
Time Complexity: O(n) to O(n log n) depending on problem
Space Complexity: O(1) to O(n) for mathematical computations
"""

from typing import List, Tuple, Optional, Dict, Any, Set
import math
from collections import defaultdict, Counter
import heapq

class GreedyMathematicalProblems:
    
    def __init__(self):
        """Initialize with mathematical problem tracking"""
        self.solution_steps = []
        self.computation_stats = {}
    
    # ==========================================
    # 1. NUMBER THEORY PROBLEMS
    # ==========================================
    
    def largest_number_from_digits(self, nums: List[int]) -> str:
        """
        Largest Number from Given Digits
        
        Company: Google, Amazon, Microsoft
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Problem: Arrange digits to form the largest possible number
        Greedy Strategy: Custom comparator - x+y vs y+x comparison
        
        LeetCode 179
        """
        print("=== LARGEST NUMBER FROM DIGITS ===")
        print("Problem: Arrange digits to form largest possible number")
        print("Greedy Strategy: Custom comparison (x+y vs y+x)")
        print()
        
        if not nums:
            return "0"
        
        print(f"Input numbers: {nums}")
        
        # Convert to strings for comparison
        str_nums = [str(num) for num in nums]
        
        print(f"String representations: {str_nums}")
        print()
        
        # Custom comparator: for two numbers x, y
        # x should come before y if x+y > y+x
        from functools import cmp_to_key
        
        def compare(x, y):
            if x + y > y + x:
                return -1  # x should come before y
            elif x + y < y + x:
                return 1   # y should come before x
            else:
                return 0   # equal
        
        print("Pairwise comparisons:")
        for i in range(len(str_nums)):
            for j in range(i + 1, len(str_nums)):
                x, y = str_nums[i], str_nums[j]
                xy = x + y
                yx = y + x
                comparison = ">" if xy > yx else "<" if xy < yx else "="
                
                print(f"   {x} + {y} = {xy} {comparison} {y} + {x} = {yx}")
                if xy > yx:
                    print(f"      → {x} should come before {y}")
                elif xy < yx:
                    print(f"      → {y} should come before {x}")
                else:
                    print(f"      → Order doesn't matter")
        print()
        
        # Sort using custom comparator
        sorted_nums = sorted(str_nums, key=cmp_to_key(compare))
        
        print(f"Sorted order: {sorted_nums}")
        
        # Handle edge case where all numbers are 0
        result = ''.join(sorted_nums)
        if result[0] == '0':
            result = '0'
        
        print(f"Largest number: {result}")
        return result
    
    def create_maximum_number(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        """
        Create Maximum Number from Two Arrays
        
        Company: Google, Amazon
        Difficulty: Hard
        Time: O(k * (m + n)²), Space: O(k)
        
        Problem: Pick k digits from two arrays to form maximum number
        Greedy Strategy: Try all combinations of lengths, maximize each subsequence
        """
        print("=== CREATE MAXIMUM NUMBER ===")
        print("Problem: Pick k digits from two arrays to form maximum number")
        print("Greedy Strategy: Optimize subsequence selection and merging")
        print()
        
        print(f"Array 1: {nums1}")
        print(f"Array 2: {nums2}")
        print(f"Required length k: {k}")
        print()
        
        def max_subsequence(nums: List[int], length: int) -> List[int]:
            """Get maximum subsequence of given length from array"""
            if length == 0:
                return []
            if length >= len(nums):
                return nums[:]
            
            # Use stack to maintain maximum subsequence
            stack = []
            to_remove = len(nums) - length
            
            for num in nums:
                while stack and stack[-1] < num and to_remove > 0:
                    stack.pop()
                    to_remove -= 1
                stack.append(num)
            
            # Remove excess elements from end
            return stack[:length]
        
        def merge_arrays(arr1: List[int], arr2: List[int]) -> List[int]:
            """Merge two arrays to form maximum number"""
            result = []
            i = j = 0
            
            while i < len(arr1) and j < len(arr2):
                # Compare remaining arrays lexicographically
                if arr1[i:] > arr2[j:]:
                    result.append(arr1[i])
                    i += 1
                else:
                    result.append(arr2[j])
                    j += 1
            
            # Add remaining elements
            result.extend(arr1[i:])
            result.extend(arr2[j:])
            
            return result
        
        max_result = []
        
        print("Trying all combinations of lengths:")
        
        # Try all possible distributions of k elements
        for i in range(max(0, k - len(nums2)), min(k, len(nums1)) + 1):
            j = k - i
            
            print(f"  Take {i} from array1, {j} from array2:")
            
            # Get maximum subsequences
            subseq1 = max_subsequence(nums1, i)
            subseq2 = max_subsequence(nums2, j)
            
            print(f"    Max subsequence from array1: {subseq1}")
            print(f"    Max subsequence from array2: {subseq2}")
            
            # Merge to form maximum number
            merged = merge_arrays(subseq1, subseq2)
            print(f"    Merged result: {merged}")
            
            # Update maximum result
            if merged > max_result:
                max_result = merged
                print(f"    ✓ New maximum found!")
            else:
                print(f"    ✗ Not better than current maximum")
            print()
        
        print(f"Final maximum number: {max_result}")
        return max_result
    
    # ==========================================
    # 2. COMBINATORIAL OPTIMIZATION
    # ==========================================
    
    def maximum_units_in_truck(self, box_types: List[List[int]], truck_size: int) -> int:
        """
        Maximum Units You Can Get in a Truck
        
        Company: Amazon, Google
        Difficulty: Easy
        Time: O(n log n), Space: O(1)
        
        Problem: Maximize units in truck given box types and truck capacity
        Greedy Strategy: Sort by units per box (value density)
        
        LeetCode 1710
        """
        print("=== MAXIMUM UNITS IN TRUCK ===")
        print("Problem: Maximize units loaded in truck")
        print("Greedy Strategy: Load boxes with highest units per box first")
        print()
        
        print(f"Truck capacity: {truck_size}")
        print("Box types [boxes_count, units_per_box]:")
        for i, (boxes, units) in enumerate(box_types):
            print(f"   Type {i+1}: {boxes} boxes, {units} units each")
        print()
        
        # Sort by units per box in descending order
        sorted_boxes = sorted(box_types, key=lambda x: x[1], reverse=True)
        
        print("Box types sorted by units per box (greedy order):")
        for i, (boxes, units) in enumerate(sorted_boxes):
            print(f"   Type {i+1}: {boxes} boxes, {units} units each")
        print()
        
        total_units = 0
        remaining_capacity = truck_size
        
        print("Greedy loading process:")
        for i, (available_boxes, units_per_box) in enumerate(sorted_boxes):
            if remaining_capacity <= 0:
                break
            
            # Take as many boxes as possible
            boxes_to_take = min(available_boxes, remaining_capacity)
            units_added = boxes_to_take * units_per_box
            
            total_units += units_added
            remaining_capacity -= boxes_to_take
            
            print(f"Step {i+1}: Box type with {units_per_box} units each")
            print(f"   Available: {available_boxes} boxes")
            print(f"   Can take: {boxes_to_take} boxes")
            print(f"   Units added: {units_added}")
            print(f"   Total units: {total_units}")
            print(f"   Remaining capacity: {remaining_capacity}")
            print()
        
        print(f"Maximum units loaded: {total_units}")
        return total_units
    
    def minimum_cost_to_make_array_equal(self, nums: List[int], cost: List[int]) -> int:
        """
        Minimum Cost to Make Array Equal
        
        Company: Google, Facebook
        Difficulty: Hard
        Time: O(n log n), Space: O(n)
        
        Problem: Make all elements equal with minimum weighted cost
        Greedy Strategy: Target value is weighted median
        """
        print("=== MINIMUM COST TO MAKE ARRAY EQUAL ===")
        print("Problem: Make all elements equal with minimum cost")
        print("Greedy Strategy: Target the weighted median")
        print()
        
        n = len(nums)
        print("Array elements and their costs:")
        for i in range(n):
            print(f"   nums[{i}] = {nums[i]}, cost = {cost[i]}")
        print()
        
        # Create pairs and sort by value
        pairs = list(zip(nums, cost))
        pairs.sort()
        
        print("Sorted by value:")
        for i, (value, weight) in enumerate(pairs):
            print(f"   {i}: value = {value}, cost = {weight}")
        print()
        
        # Find weighted median
        total_cost = sum(cost)
        cumulative_cost = 0
        target_value = pairs[0][0]  # Default to first value
        
        print("Finding weighted median:")
        print(f"Total cost: {total_cost}, Half: {total_cost // 2}")
        
        for i, (value, weight) in enumerate(pairs):
            cumulative_cost += weight
            print(f"   Value {value}: cumulative cost = {cumulative_cost}")
            
            if cumulative_cost >= total_cost // 2:
                target_value = value
                print(f"   ✓ Weighted median found: {target_value}")
                break
        
        print()
        
        # Calculate minimum cost to make all elements equal to target
        min_cost = 0
        
        print(f"Calculating cost to make all elements equal to {target_value}:")
        for i in range(n):
            change_cost = abs(nums[i] - target_value) * cost[i]
            min_cost += change_cost
            
            print(f"   Element {nums[i]} → {target_value}: |{nums[i]} - {target_value}| × {cost[i]} = {change_cost}")
        
        print(f"\nMinimum total cost: {min_cost}")
        return min_cost
    
    # ==========================================
    # 3. OPTIMIZATION WITH CONSTRAINTS
    # ==========================================
    
    def maximum_profit_job_scheduling(self, start_time: List[int], end_time: List[int], profit: List[int]) -> int:
        """
        Maximum Profit in Job Scheduling (Greedy + DP approach)
        
        Company: Google, Amazon
        Difficulty: Hard
        Time: O(n log n), Space: O(n)
        
        Problem: Schedule non-overlapping jobs to maximize profit
        Greedy Strategy: Sort by end time, use DP for optimal selection
        """
        print("=== MAXIMUM PROFIT JOB SCHEDULING ===")
        print("Problem: Schedule non-overlapping jobs to maximize profit")
        print("Strategy: Sort by end time + dynamic programming")
        print()
        
        n = len(start_time)
        
        print("Jobs (start, end, profit):")
        jobs = []
        for i in range(n):
            jobs.append((start_time[i], end_time[i], profit[i]))
            print(f"   Job {i+1}: [{start_time[i]}, {end_time[i]}], profit = {profit[i]}")
        print()
        
        # Sort jobs by end time
        jobs.sort(key=lambda x: x[1])
        
        print("Jobs sorted by end time:")
        for i, (start, end, prof) in enumerate(jobs):
            print(f"   Job {i+1}: [{start}, {end}], profit = {prof}")
        print()
        
        # Find latest non-overlapping job using binary search
        def find_latest_non_overlapping(jobs, i):
            """Find latest job that doesn't overlap with job i"""
            target_end = jobs[i][0]  # Start time of job i
            
            left, right = 0, i - 1
            result = -1
            
            while left <= right:
                mid = (left + right) // 2
                if jobs[mid][1] <= target_end:  # End time <= start time of job i
                    result = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            return result
        
        # Dynamic programming approach
        dp = [0] * n
        dp[0] = jobs[0][2]  # First job profit
        
        print("Dynamic programming calculation:")
        print(f"   dp[0] = {dp[0]} (take job 1)")
        
        for i in range(1, n):
            # Option 1: Don't take current job
            profit_without = dp[i-1]
            
            # Option 2: Take current job
            profit_with = jobs[i][2]  # Current job profit
            
            # Find latest non-overlapping job
            latest_non_overlapping = find_latest_non_overlapping(jobs, i)
            
            if latest_non_overlapping != -1:
                profit_with += dp[latest_non_overlapping]
                print(f"   Job {i+1}: can combine with jobs up to {latest_non_overlapping+1}")
            else:
                print(f"   Job {i+1}: no compatible previous jobs")
            
            dp[i] = max(profit_without, profit_with)
            
            print(f"   dp[{i}] = max({profit_without}, {profit_with}) = {dp[i]}")
        
        print(f"\nMaximum profit: {dp[n-1]}")
        return dp[n-1]
    
    def minimum_number_of_arrows(self, points: List[List[int]]) -> int:
        """
        Minimum Number of Arrows to Burst Balloons
        
        Company: Microsoft, Amazon
        Difficulty: Medium
        Time: O(n log n), Space: O(1)
        
        Problem: Find minimum arrows to burst all balloons
        Greedy Strategy: Sort by end coordinate, shoot at end of each interval
        
        LeetCode 452
        """
        print("=== MINIMUM ARROWS TO BURST BALLOONS ===")
        print("Problem: Find minimum arrows needed to burst all balloons")
        print("Greedy Strategy: Sort by end coordinate, shoot at interval ends")
        print()
        
        if not points:
            return 0
        
        print("Balloon intervals [start, end]:")
        for i, (start, end) in enumerate(points):
            print(f"   Balloon {i+1}: [{start}, {end}]")
        print()
        
        # Sort by end coordinate
        sorted_points = sorted(points, key=lambda x: x[1])
        
        print("Balloons sorted by end coordinate:")
        for i, (start, end) in enumerate(sorted_points):
            print(f"   Balloon {i+1}: [{start}, {end}]")
        print()
        
        arrows = 1
        arrow_position = sorted_points[0][1]  # Shoot at end of first interval
        
        print("Greedy arrow shooting:")
        print(f"Arrow 1: Position {arrow_position}")
        print(f"   ✓ Bursts balloon [{sorted_points[0][0]}, {sorted_points[0][1]}]")
        
        for i in range(1, len(sorted_points)):
            start, end = sorted_points[i]
            
            print(f"\nConsider balloon [{start}, {end}]:")
            
            if start <= arrow_position:
                # Current arrow can burst this balloon
                print(f"   ✓ Arrow at {arrow_position} can burst this balloon")
            else:
                # Need new arrow
                arrows += 1
                arrow_position = end
                print(f"   ✗ Need new arrow")
                print(f"   Arrow {arrows}: Position {arrow_position}")
                print(f"   ✓ Bursts balloon [{start}, {end}]")
        
        print(f"\nMinimum arrows needed: {arrows}")
        return arrows
    
    # ==========================================
    # 4. MATHEMATICAL SEQUENCES
    # ==========================================
    
    def wiggle_subsequence(self, nums: List[int]) -> int:
        """
        Wiggle Subsequence
        
        Company: Google, Facebook
        Difficulty: Medium
        Time: O(n), Space: O(1)
        
        Problem: Find longest wiggle subsequence
        Greedy Strategy: Count direction changes in sequence
        
        LeetCode 376
        """
        print("=== WIGGLE SUBSEQUENCE ===")
        print("Problem: Find longest wiggle subsequence")
        print("Greedy Strategy: Count direction changes")
        print()
        
        if len(nums) < 2:
            return len(nums)
        
        print(f"Input sequence: {nums}")
        print()
        
        # Find differences between consecutive elements
        diffs = []
        for i in range(1, len(nums)):
            diff = nums[i] - nums[i-1]
            if diff != 0:  # Ignore zero differences
                diffs.append(diff)
        
        print("Non-zero differences:")
        for i, diff in enumerate(diffs):
            sign = "+" if diff > 0 else "-"
            print(f"   Position {i}: {sign}{abs(diff)}")
        
        if not diffs:
            return 1  # All elements are same
        
        print()
        
        # Count direction changes
        wiggle_length = 1  # Start with first element
        prev_positive = diffs[0] > 0
        
        print("Counting direction changes:")
        print(f"Start with element {nums[0]}")
        print(f"First direction: {'positive' if prev_positive else 'negative'}")
        wiggle_length += 1  # Add second element
        
        for i in range(1, len(diffs)):
            current_positive = diffs[i] > 0
            
            print(f"Position {i+1}: difference is {'positive' if current_positive else 'negative'}")
            
            if current_positive != prev_positive:
                # Direction changed - add to wiggle sequence
                wiggle_length += 1
                prev_positive = current_positive
                print(f"   ✓ Direction change! Wiggle length: {wiggle_length}")
            else:
                print(f"   ✗ Same direction, continue")
        
        print(f"\nLongest wiggle subsequence length: {wiggle_length}")
        return wiggle_length
    
    def increasing_triplet_subsequence(self, nums: List[int]) -> bool:
        """
        Increasing Triplet Subsequence
        
        Company: Facebook, Google
        Difficulty: Medium
        Time: O(n), Space: O(1)
        
        Problem: Check if there exists increasing triplet in array
        Greedy Strategy: Maintain two smallest values seen so far
        
        LeetCode 334
        """
        print("=== INCREASING TRIPLET SUBSEQUENCE ===")
        print("Problem: Check if increasing triplet exists")
        print("Greedy Strategy: Track two smallest values")
        print()
        
        print(f"Input array: {nums}")
        print()
        
        first = float('inf')   # Smallest value
        second = float('inf')  # Second smallest value
        
        print("Greedy scanning process:")
        for i, num in enumerate(nums):
            print(f"Step {i+1}: Process {num}")
            print(f"   Current first (smallest): {first if first != float('inf') else 'None'}")
            print(f"   Current second: {second if second != float('inf') else 'None'}")
            
            if num <= first:
                first = num
                print(f"   ✓ Updated first to {first}")
            elif num <= second:
                second = num
                print(f"   ✓ Updated second to {second}")
            else:
                # Found third number larger than both first and second
                print(f"   ✓ Found triplet: {first} < {second} < {num}")
                print(f"   Increasing triplet exists!")
                return True
            
            print()
        
        print("No increasing triplet found")
        return False


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_greedy_mathematical_problems():
    """Demonstrate all greedy mathematical problems"""
    print("=== GREEDY MATHEMATICAL PROBLEMS DEMONSTRATION ===\n")
    
    math_problems = GreedyMathematicalProblems()
    
    # 1. Number Theory
    print("1. NUMBER THEORY PROBLEMS")
    
    print("a) Largest Number from Digits:")
    math_problems.largest_number_from_digits([10, 2])
    print("\n" + "-"*40 + "\n")
    
    math_problems.largest_number_from_digits([3, 30, 34, 5, 9])
    print("\n" + "-"*40 + "\n")
    
    print("b) Create Maximum Number:")
    math_problems.create_maximum_number([3, 4, 6, 5], [9, 1, 2, 5, 8, 3], 5)
    print("\n" + "="*60 + "\n")
    
    # 2. Combinatorial Optimization
    print("2. COMBINATORIAL OPTIMIZATION")
    
    print("a) Maximum Units in Truck:")
    box_types = [[1, 3], [2, 2], [3, 1]]
    math_problems.maximum_units_in_truck(box_types, 4)
    print("\n" + "-"*40 + "\n")
    
    print("b) Minimum Cost to Make Array Equal:")
    math_problems.minimum_cost_to_make_array_equal([1, 3, 5, 2], [2, 3, 1, 14])
    print("\n" + "="*60 + "\n")
    
    # 3. Optimization with Constraints
    print("3. OPTIMIZATION WITH CONSTRAINTS")
    
    print("a) Maximum Profit Job Scheduling:")
    start_times = [1, 2, 3, 3]
    end_times = [3, 4, 5, 6]
    profits = [50, 10, 40, 70]
    math_problems.maximum_profit_job_scheduling(start_times, end_times, profits)
    print("\n" + "-"*40 + "\n")
    
    print("b) Minimum Arrows to Burst Balloons:")
    balloons = [[10, 16], [2, 8], [1, 6], [7, 12]]
    math_problems.minimum_number_of_arrows(balloons)
    print("\n" + "="*60 + "\n")
    
    # 4. Mathematical Sequences
    print("4. MATHEMATICAL SEQUENCES")
    
    print("a) Wiggle Subsequence:")
    math_problems.wiggle_subsequence([1, 7, 4, 9, 2, 5])
    print("\n" + "-"*40 + "\n")
    
    print("b) Increasing Triplet Subsequence:")
    math_problems.increasing_triplet_subsequence([1, 2, 3, 4, 5])
    print("\n" + "-"*40 + "\n")
    
    math_problems.increasing_triplet_subsequence([5, 4, 3, 2, 1])


if __name__ == "__main__":
    demonstrate_greedy_mathematical_problems()
    
    print("\n=== MATHEMATICAL PROBLEMS MASTERY GUIDE ===")
    
    print("\n🎯 MATHEMATICAL PROBLEM CATEGORIES:")
    print("• Number Theory: Digit manipulation, number construction")
    print("• Combinatorial: Optimal selection from sets with constraints")
    print("• Sequence Analysis: Pattern detection and optimization")
    print("• Weighted Optimization: Problems with cost/benefit ratios")
    print("• Geometric: Interval problems, coordinate-based optimization")
    
    print("\n📊 COMPLEXITY PATTERNS:")
    print("• Number construction: O(n log n) for sorting-based approaches")
    print("• Weighted problems: O(n log n) for median finding")
    print("• Sequence problems: O(n) for single-pass algorithms")
    print("• Interval optimization: O(n log n) for sorting intervals")
    print("• Selection problems: O(n log n) for priority-based selection")
    
    print("\n⚡ KEY MATHEMATICAL STRATEGIES:")
    print("• Custom comparators for optimal ordering")
    print("• Weighted median for cost minimization")
    print("• Direction change counting for sequence problems")
    print("• Interval endpoint optimization")
    print("• Greedy state tracking (first, second smallest/largest)")
    
    print("\n🔢 MATHEMATICAL INSIGHTS:")
    print("• Lexicographic ordering in string/number problems")
    print("• Weighted median minimizes total absolute deviation")
    print("• Greedy works when local optimality guarantees global optimality")
    print("• Binary search integration for optimization problems")
    print("• Mathematical properties enable O(1) space solutions")
    
    print("\n🔧 IMPLEMENTATION TECHNIQUES:")
    print("• Use functools.cmp_to_key for custom sorting")
    print("• Implement weighted median finding algorithms")
    print("• Track multiple state variables for sequence problems")
    print("• Use binary search for non-overlapping interval finding")
    print("• Apply mathematical properties for space optimization")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• Financial: Portfolio optimization, risk management")
    print("• Operations Research: Resource allocation, scheduling")
    print("• Computer Graphics: Geometric optimization")
    print("• Data Compression: Optimal encoding schemes")
    print("• Machine Learning: Feature selection, model optimization")
    
    print("\n🎓 ADVANCED MATHEMATICAL CONCEPTS:")
    print("• Convexity properties in optimization problems")
    print("• Majorization theory for inequality problems")
    print("• Matroid theory for independence constraints")
    print("• Approximation algorithms for NP-hard problems")
    print("• Online algorithms for streaming data")
    
    print("\n💡 PROBLEM-SOLVING PRINCIPLES:")
    print("• Identify the mathematical property that enables greedy choice")
    print("• Look for monotonicity and convexity in objective functions")
    print("• Consider weighted vs unweighted versions of problems")
    print("• Apply mathematical inequalities to prove optimality")
    print("• Use invariants and mathematical induction for correctness")
