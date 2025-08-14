"""
Heap Interview Problems - Most Asked Questions
==============================================

Topics: Common heap interview questions with solutions and explanations
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber, LinkedIn
Difficulty: Easy to Hard
Time Complexity: Usually O(n log k) or O(n log n)
Space Complexity: O(k) or O(n) for heap storage
"""

from typing import List, Optional, Dict, Tuple, Any
import heapq
from collections import Counter, defaultdict
import math

class HeapInterviewProblems:
    
    def __init__(self):
        """Initialize with problem tracking"""
        self.problem_count = 0
        self.solution_steps = []
    
    # ==========================================
    # 1. CLASSIC HEAP INTERVIEW PROBLEMS
    # ==========================================
    
    def kth_largest_element_in_array(self, nums: List[int], k: int) -> int:
        """
        Kth Largest Element in an Array
        
        Company: Facebook, Amazon, Apple, Google
        Difficulty: Medium
        Time: O(n log k), Space: O(k)
        
        LeetCode 215: Most asked heap problem
        """
        print(f"Finding {k}th largest element in: {nums}")
        print("Using min heap of size k for optimal space complexity")
        print()
        
        # Use min heap to keep k largest elements
        min_heap = []
        
        for i, num in enumerate(nums):
            print(f"Step {i+1}: Processing {num}")
            
            if len(min_heap) < k:
                # Heap not full, just add
                heapq.heappush(min_heap, num)
                print(f"   Heap size < k, added {num}")
            else:
                # Heap full, only add if larger than minimum
                if num > min_heap[0]:
                    replaced = heapq.heapreplace(min_heap, num)
                    print(f"   {num} > min {replaced}, replaced minimum")
                else:
                    print(f"   {num} ≤ min {min_heap[0]}, not added")
            
            print(f"   Current heap: {min_heap}")
            if len(min_heap) == k:
                print(f"   Current {k}th largest: {min_heap[0]}")
            print()
        
        kth_largest = min_heap[0]
        print(f"The {k}th largest element is: {kth_largest}")
        return kth_largest
    
    def top_k_frequent_elements(self, nums: List[int], k: int) -> List[int]:
        """
        Top K Frequent Elements
        
        Company: Amazon, Facebook, Google, Uber
        Difficulty: Medium
        Time: O(n log k), Space: O(n)
        
        LeetCode 347: Very popular heap + hash map problem
        """
        # Count frequencies
        freq_count = Counter(nums)
        
        print(f"Top {k} frequent elements in: {nums}")
        print(f"Frequency count: {dict(freq_count)}")
        print()
        
        # Use min heap to keep k most frequent elements
        min_heap = []
        
        for element, frequency in freq_count.items():
            print(f"Processing element {element} with frequency {frequency}")
            
            if len(min_heap) < k:
                heapq.heappush(min_heap, (frequency, element))
                print(f"   Added to heap: {min_heap}")
            elif frequency > min_heap[0][0]:
                replaced = heapq.heapreplace(min_heap, (frequency, element))
                print(f"   Replaced {replaced} with ({frequency}, {element})")
            else:
                print(f"   Frequency {frequency} ≤ min frequency {min_heap[0][0]}, skipped")
            
            print(f"   Current top {len(min_heap)} frequent: {[elem for _, elem in sorted(min_heap, reverse=True)]}")
            print()
        
        # Extract elements (most frequent first)
        result = [element for freq, element in sorted(min_heap, reverse=True)]
        print(f"Top {k} frequent elements: {result}")
        return result
    
    def merge_k_sorted_lists(self, lists: List[List[int]]) -> List[int]:
        """
        Merge k Sorted Lists
        
        Company: Google, Facebook, Amazon, Microsoft
        Difficulty: Hard
        Time: O(n log k), Space: O(k)
        
        LeetCode 23: Classic heap problem for merging
        """
        if not lists:
            return []
        
        # Min heap: (value, list_index, element_index)
        min_heap = []
        result = []
        
        print(f"Merging {len(lists)} sorted lists:")
        for i, lst in enumerate(lists):
            print(f"   List {i}: {lst}")
        print()
        
        # Initialize heap with first element from each non-empty list
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(min_heap, (lst[0], i, 0))
        
        print(f"Initial heap: {min_heap}")
        print()
        
        step = 1
        while min_heap:
            value, list_idx, elem_idx = heapq.heappop(min_heap)
            result.append(value)
            
            print(f"Step {step}: Extracted {value} from list {list_idx}")
            
            # Add next element from same list if available
            if elem_idx + 1 < len(lists[list_idx]):
                next_value = lists[list_idx][elem_idx + 1]
                heapq.heappush(min_heap, (next_value, list_idx, elem_idx + 1))
                print(f"   Added next element {next_value} from list {list_idx}")
            
            print(f"   Result so far: {result}")
            print(f"   Heap: {min_heap}")
            print()
            step += 1
        
        print(f"Final merged result: {result}")
        return result
    
    def find_median_from_data_stream(self) -> None:
        """
        Find Median from Data Stream
        
        Company: Google, Facebook, Amazon
        Difficulty: Hard
        Time: O(log n) addNum, O(1) findMedian
        Space: O(n)
        
        LeetCode 295: Two heaps classic problem
        """
        
        class MedianFinder:
            def __init__(self):
                # Max heap for smaller half (use negative values)
                self.max_heap = []
                # Min heap for larger half
                self.min_heap = []
            
            def addNum(self, num: int) -> None:
                """Add number to data structure"""
                print(f"Adding number: {num}")
                
                # Decide which heap to add to
                if not self.max_heap or num <= -self.max_heap[0]:
                    heapq.heappush(self.max_heap, -num)
                    print(f"   Added to max_heap (smaller half)")
                else:
                    heapq.heappush(self.min_heap, num)
                    print(f"   Added to min_heap (larger half)")
                
                # Balance heaps
                self._balance()
                
                print(f"   Max heap (smaller): {[-x for x in self.max_heap]}")
                print(f"   Min heap (larger): {self.min_heap}")
                print(f"   Current median: {self.findMedian()}")
                print()
            
            def findMedian(self) -> float:
                """Find median of all numbers"""
                if len(self.max_heap) == len(self.min_heap):
                    if not self.max_heap:
                        return 0.0
                    return (-self.max_heap[0] + self.min_heap[0]) / 2.0
                elif len(self.max_heap) > len(self.min_heap):
                    return float(-self.max_heap[0])
                else:
                    return float(self.min_heap[0])
            
            def _balance(self) -> None:
                """Balance heap sizes"""
                if len(self.max_heap) > len(self.min_heap) + 1:
                    value = -heapq.heappop(self.max_heap)
                    heapq.heappush(self.min_heap, value)
                    print(f"   Rebalanced: moved {value} to min_heap")
                elif len(self.min_heap) > len(self.max_heap) + 1:
                    value = heapq.heappop(self.min_heap)
                    heapq.heappush(self.max_heap, -value)
                    print(f"   Rebalanced: moved {value} to max_heap")
        
        print("=== FIND MEDIAN FROM DATA STREAM ===")
        median_finder = MedianFinder()
        
        # Test with stream of numbers
        numbers = [1, 2, 3, 4, 5]
        for num in numbers:
            median_finder.addNum(num)
    
    # ==========================================
    # 2. MEDIUM DIFFICULTY PROBLEMS
    # ==========================================
    
    def last_stone_weight(self, stones: List[int]) -> int:
        """
        Last Stone Weight
        
        Company: Amazon, Facebook
        Difficulty: Easy
        Time: O(n log n), Space: O(n)
        
        LeetCode 1046: Simulate stone smashing using max heap
        """
        # Convert to max heap using negative values
        max_heap = [-stone for stone in stones]
        heapq.heapify(max_heap)
        
        print(f"Last Stone Weight simulation with stones: {stones}")
        print(f"Initial max heap: {[-x for x in max_heap]}")
        print()
        
        round_num = 1
        while len(max_heap) > 1:
            print(f"Round {round_num}:")
            
            # Take two heaviest stones
            first = -heapq.heappop(max_heap)
            second = -heapq.heappop(max_heap)
            
            print(f"   Heaviest stones: {first} and {second}")
            
            if first != second:
                # Stones have different weights, put difference back
                difference = first - second
                heapq.heappush(max_heap, -difference)
                print(f"   Difference {difference} added back to heap")
            else:
                print(f"   Stones have same weight, both destroyed")
            
            remaining_stones = [-x for x in max_heap]
            print(f"   Remaining stones: {remaining_stones}")
            print()
            round_num += 1
        
        result = -max_heap[0] if max_heap else 0
        print(f"Last remaining stone weight: {result}")
        return result
    
    def kth_smallest_element_in_sorted_matrix(self, matrix: List[List[int]], k: int) -> int:
        """
        Kth Smallest Element in a Sorted Matrix
        
        Company: Google, Facebook, Amazon
        Difficulty: Medium
        Time: O(min(k, n) + k log(min(k, n))), Space: O(min(k, n))
        
        LeetCode 378: Use heap to traverse sorted matrix efficiently
        """
        n = len(matrix)
        
        print(f"Finding {k}th smallest element in sorted matrix:")
        for row in matrix:
            print(f"   {row}")
        print()
        
        # Min heap: (value, row, col)
        min_heap = []
        
        # Add first element of each row to heap
        for i in range(min(k, n)):
            heapq.heappush(min_heap, (matrix[i][0], i, 0))
        
        print(f"Initial heap with first elements: {min_heap}")
        print()
        
        # Extract k-1 elements, then return kth
        for step in range(k):
            value, row, col = heapq.heappop(min_heap)
            
            print(f"Step {step + 1}: Extracted {value} from position ({row}, {col})")
            
            if step == k - 1:
                print(f"Found {k}th smallest element: {value}")
                return value
            
            # Add next element from same row if available
            if col + 1 < len(matrix[row]):
                next_value = matrix[row][col + 1]
                heapq.heappush(min_heap, (next_value, row, col + 1))
                print(f"   Added next element {next_value} from row {row}")
            
            print(f"   Current heap: {min_heap}")
            print()
        
        return -1  # Should never reach here
    
    def reorganize_string(self, s: str) -> str:
        """
        Reorganize String (no two adjacent characters are the same)
        
        Company: Google, Facebook, Amazon
        Difficulty: Medium
        Time: O(n log k), Space: O(k) where k is unique characters
        
        LeetCode 767: Use max heap to greedily place most frequent characters
        """
        # Count character frequencies
        char_count = Counter(s)
        
        print(f"Reorganizing string: '{s}'")
        print(f"Character frequencies: {dict(char_count)}")
        
        # Check if reorganization is possible
        max_freq = max(char_count.values())
        if max_freq > (len(s) + 1) // 2:
            print(f"Impossible to reorganize: max frequency {max_freq} > {(len(s) + 1) // 2}")
            return ""
        
        # Max heap with character frequencies
        max_heap = [(-count, char) for char, count in char_count.items()]
        heapq.heapify(max_heap)
        
        print(f"Initial max heap: {[(-count, char) for count, char in max_heap]}")
        print()
        
        result = []
        prev_char = None
        prev_count = 0
        
        step = 1
        while max_heap:
            print(f"Step {step}:")
            
            # Get most frequent character
            count, char = heapq.heappop(max_heap)
            count = -count
            
            result.append(char)
            print(f"   Added '{char}' (frequency was {count})")
            print(f"   Result so far: '{''.join(result)}'")
            
            # Add back previous character if it still has occurrences
            if prev_count > 0:
                heapq.heappush(max_heap, (-prev_count, prev_char))
                print(f"   Added back '{prev_char}' with count {prev_count}")
            
            # Update previous character info
            prev_char = char
            prev_count = count - 1
            
            print(f"   Heap after step: {[(-c, ch) for c, ch in max_heap]}")
            print()
            step += 1
        
        # Check if we used all characters
        if len(result) != len(s):
            print("Failed to use all characters")
            return ""
        
        final_result = ''.join(result)
        print(f"Successfully reorganized string: '{final_result}'")
        return final_result
    
    def ugly_number_ii(self, n: int) -> int:
        """
        Ugly Number II
        
        Company: Facebook, Amazon, Google
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        LeetCode 264: Find nth ugly number using heap
        """
        print(f"Finding the {n}th ugly number")
        print("Ugly numbers have only 2, 3, and 5 as prime factors")
        print()
        
        # Min heap to generate ugly numbers in order
        min_heap = [1]
        seen = {1}
        factors = [2, 3, 5]
        
        print("Generating ugly numbers in sequence:")
        
        for i in range(n):
            # Get next ugly number
            ugly = heapq.heappop(min_heap)
            
            print(f"Ugly number #{i + 1}: {ugly}")
            
            if i == n - 1:
                print(f"The {n}th ugly number is: {ugly}")
                return ugly
            
            # Generate next ugly numbers by multiplying with 2, 3, 5
            for factor in factors:
                new_ugly = ugly * factor
                if new_ugly not in seen:
                    seen.add(new_ugly)
                    heapq.heappush(min_heap, new_ugly)
            
            # Show next few in heap
            next_few = sorted(min_heap)[:5]
            print(f"   Next ugly numbers in heap: {next_few}...")
            print()
        
        return -1
    
    # ==========================================
    # 3. HARD DIFFICULTY PROBLEMS
    # ==========================================
    
    def sliding_window_maximum(self, nums: List[int], k: int) -> List[int]:
        """
        Sliding Window Maximum
        
        Company: Amazon, Google, Microsoft
        Difficulty: Hard
        Time: O(n log k), Space: O(k)
        
        LeetCode 239: Find maximum in each sliding window
        Note: Deque solution is O(n), but heap shows different approach
        """
        if not nums or k <= 0:
            return []
        
        # Max heap storing (value, index) pairs
        max_heap = []
        result = []
        
        print(f"Sliding Window Maximum: nums={nums}, k={k}")
        print("Using max heap approach (deque is more efficient)")
        print()
        
        for i in range(len(nums)):
            # Add current element to heap (negative for max heap)
            heapq.heappush(max_heap, (-nums[i], i))
            
            print(f"Step {i+1}: Added nums[{i}] = {nums[i]}")
            
            # Remove elements outside current window
            while max_heap and max_heap[0][1] <= i - k:
                removed_val, removed_idx = heapq.heappop(max_heap)
                print(f"   Removed element {-removed_val} at index {removed_idx} (outside window)")
            
            # If window is complete, record maximum
            if i >= k - 1:
                window_max = -max_heap[0][0]
                result.append(window_max)
                
                window_start = i - k + 1
                window_elements = nums[window_start:i+1]
                print(f"   Window [{window_start}:{i+1}]: {window_elements} -> Max = {window_max}")
            
            print()
        
        print(f"Sliding window maximums: {result}")
        return result
    
    def smallest_range_covering_elements(self, nums: List[List[int]]) -> List[int]:
        """
        Smallest Range Covering Elements from K Lists
        
        Company: Google, Amazon
        Difficulty: Hard
        Time: O(n log k), Space: O(k)
        
        LeetCode 632: Find smallest range that includes at least one number from each list
        """
        if not nums:
            return []
        
        # Min heap: (value, list_index, element_index)
        min_heap = []
        max_val = float('-inf')
        
        print(f"Finding smallest range covering elements from {len(nums)} lists:")
        for i, lst in enumerate(nums):
            print(f"   List {i}: {lst}")
        print()
        
        # Initialize heap with first element from each list
        for i, lst in enumerate(nums):
            if lst:
                heapq.heappush(min_heap, (lst[0], i, 0))
                max_val = max(max_val, lst[0])
        
        print(f"Initial heap: {min_heap}")
        print(f"Initial max value: {max_val}")
        print()
        
        # Track smallest range
        range_start, range_end = 0, float('inf')
        
        step = 1
        while len(min_heap) == len(nums):  # All lists represented
            min_val, list_idx, elem_idx = heapq.heappop(min_heap)
            
            print(f"Step {step}: Current range [{min_val}, {max_val}], size = {max_val - min_val + 1}")
            
            # Update smallest range if current is better
            if max_val - min_val < range_end - range_start:
                range_start, range_end = min_val, max_val
                print(f"   New best range: [{range_start}, {range_end}]")
            
            # Add next element from same list if available
            if elem_idx + 1 < len(nums[list_idx]):
                next_val = nums[list_idx][elem_idx + 1]
                heapq.heappush(min_heap, (next_val, list_idx, elem_idx + 1))
                max_val = max(max_val, next_val)
                print(f"   Added {next_val} from list {list_idx}, new max = {max_val}")
            else:
                print(f"   List {list_idx} exhausted, stopping")
                break
            
            print()
            step += 1
        
        result = [range_start, range_end]
        print(f"Smallest range: {result}")
        return result
    
    def super_ugly_numbers(self, n: int, primes: List[int]) -> int:
        """
        Super Ugly Numbers
        
        Company: Google, Facebook
        Difficulty: Medium
        Time: O(n log k), Space: O(n + k)
        
        LeetCode 313: Generalization of ugly numbers with custom prime factors
        """
        print(f"Finding {n}th super ugly number with primes: {primes}")
        print()
        
        # Min heap to generate super ugly numbers
        min_heap = [1]
        seen = {1}
        
        print("Generating super ugly numbers:")
        
        for i in range(n):
            # Get next super ugly number
            ugly = heapq.heappop(min_heap)
            
            print(f"Super ugly #{i + 1}: {ugly}")
            
            if i == n - 1:
                print(f"The {n}th super ugly number is: {ugly}")
                return ugly
            
            # Generate next numbers by multiplying with each prime
            for prime in primes:
                new_ugly = ugly * prime
                if new_ugly not in seen:
                    seen.add(new_ugly)
                    heapq.heappush(min_heap, new_ugly)
            
            # Show progress
            if len(min_heap) > 0:
                next_few = sorted(min_heap)[:3]
                print(f"   Next in sequence: {next_few}...")
            print()
        
        return -1
    
    # ==========================================
    # 4. INTERVIEW STRATEGY AND TIPS
    # ==========================================
    
    def interview_approach_guide(self) -> None:
        """
        Comprehensive guide for tackling heap interview problems
        """
        print("=== HEAP INTERVIEW APPROACH GUIDE ===")
        print()
        
        print("🎯 STEP 1: PROBLEM RECOGNITION")
        print("Look for these keywords/patterns:")
        print("• 'Kth largest/smallest', 'top k', 'bottom k'")
        print("• 'Merge k sorted', 'combine multiple sequences'")
        print("• 'Median', 'middle value', 'streaming data'")
        print("• 'Priority', 'most/least frequent', 'schedule'")
        print("• 'Sliding window maximum/minimum'")
        print("• 'Shortest path', 'minimum cost', 'optimization'")
        print()
        
        print("🎯 STEP 2: CHOOSE HEAP TYPE")
        print("Decision framework:")
        print("• K largest elements → Min heap of size k")
        print("• K smallest elements → Max heap of size k")
        print("• Median finding → Two heaps (max + min)")
        print("• Merge operations → Min heap with pointers")
        print("• Priority scheduling → Max/min heap based on priority")
        print("• Optimization problems → Min heap for costs")
        print()
        
        print("🎯 STEP 3: IMPLEMENTATION PATTERNS")
        print()
        print("A) K-Element Pattern:")
        print("   heap = []")
        print("   for element in array:")
        print("       if len(heap) < k:")
        print("           heappush(heap, element)")
        print("       elif should_replace(element, heap[0]):")
        print("           heapreplace(heap, element)")
        print()
        print("B) Merge Pattern:")
        print("   heap = [(array[0], 0, 0) for array in arrays if array]")
        print("   while heap:")
        print("       value, array_idx, elem_idx = heappop(heap)")
        print("       result.append(value)")
        print("       # Add next element from same array")
        print()
        print("C) Two Heaps Pattern (Median):")
        print("   max_heap = []  # Smaller half")
        print("   min_heap = []  # Larger half")
        print("   # Maintain: |max_heap| - |min_heap| ≤ 1")
        print()
        
        print("🎯 STEP 4: OPTIMIZATION CONSIDERATIONS")
        print("• Space optimization: Use heap of size k instead of n")
        print("• Time optimization: Choose right heap type and operations")
        print("• Alternative approaches: Consider QuickSelect for kth element")
        print("• Memory efficiency: Use generators for large datasets")
        print()
        
        print("🎯 STEP 5: TESTING STRATEGY")
        print("Test cases to consider:")
        print("• Empty input or k = 0")
        print("• k larger than array size")
        print("• All elements equal")
        print("• Single element")
        print("• Already sorted vs reverse sorted")
        print("• Duplicate elements")
    
    def common_mistakes(self) -> None:
        """
        Common mistakes in heap interview problems
        """
        print("=== COMMON HEAP INTERVIEW MISTAKES ===")
        print()
        
        print("❌ MISTAKE 1: Wrong heap type for k-element problems")
        print("Problem: Using max heap for k largest elements")
        print("Solution: Use min heap of size k for k largest elements")
        print("Explanation: Min heap keeps k largest by ejecting smaller elements")
        print()
        
        print("❌ MISTAKE 2: Not handling Python's min heap correctly")
        print("Problem: Forgetting Python heapq is min heap only")
        print("Solution: Negate values for max heap behavior")
        print("Example: heappush(heap, -value) for max heap")
        print()
        
        print("❌ MISTAKE 3: Inefficient heap operations")
        print("Problem: Using heappop + heappush instead of heapreplace")
        print("Solution: Use heapreplace when replacing top element")
        print("Benefit: More efficient and atomic operation")
        print()
        
        print("❌ MISTAKE 4: Not balancing heaps in median problems")
        print("Problem: Letting heap sizes differ by more than 1")
        print("Solution: Always rebalance after insertion")
        print("Rule: |max_heap| - |min_heap| ≤ 1")
        print()
        
        print("❌ MISTAKE 5: Forgetting to handle duplicates")
        print("Problem: Not considering equal elements in ordering")
        print("Solution: Use tuples with tiebreakers or stable sorting")
        print("Example: (priority, timestamp, data)")
        print()
        
        print("❌ MISTAKE 6: Incorrect complexity analysis")
        print("Problem: Thinking heap operations are O(1)")
        print("Solution: Remember insert/delete are O(log n), only peek is O(1)")
        print("Impact: Affects overall algorithm complexity")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_heap_interview_problems():
    """Demonstrate key heap interview problems"""
    print("=== HEAP INTERVIEW PROBLEMS DEMONSTRATION ===\n")
    
    problems = HeapInterviewProblems()
    
    # 1. Classic problems
    print("=== CLASSIC HEAP PROBLEMS ===")
    
    print("1. Kth Largest Element in Array:")
    problems.kth_largest_element_in_array([3, 2, 1, 5, 6, 4], 2)
    print("\n" + "-" * 60 + "\n")
    
    print("2. Top K Frequent Elements:")
    problems.top_k_frequent_elements([1, 1, 1, 2, 2, 3], 2)
    print("\n" + "-" * 60 + "\n")
    
    print("3. Merge K Sorted Lists:")
    problems.merge_k_sorted_lists([[1, 4, 5], [1, 3, 4], [2, 6]])
    print("\n" + "-" * 60 + "\n")
    
    print("4. Find Median from Data Stream:")
    problems.find_median_from_data_stream()
    print("\n" + "=" * 60 + "\n")
    
    # 2. Medium problems
    print("=== MEDIUM DIFFICULTY PROBLEMS ===")
    
    print("1. Last Stone Weight:")
    problems.last_stone_weight([2, 7, 4, 1, 8, 1])
    print("\n" + "-" * 60 + "\n")
    
    print("2. Kth Smallest Element in Sorted Matrix:")
    matrix = [[1, 5, 9], [10, 11, 13], [12, 13, 15]]
    problems.kth_smallest_element_in_sorted_matrix(matrix, 8)
    print("\n" + "-" * 60 + "\n")
    
    print("3. Reorganize String:")
    problems.reorganize_string("aab")
    print("\n" + "-" * 60 + "\n")
    
    print("4. Ugly Number II:")
    problems.ugly_number_ii(10)
    print("\n" + "=" * 60 + "\n")
    
    # 3. Hard problems
    print("=== HARD DIFFICULTY PROBLEMS ===")
    
    print("1. Sliding Window Maximum:")
    problems.sliding_window_maximum([1, 3, -1, -3, 5, 3, 6, 7], 3)
    print("\n" + "-" * 60 + "\n")
    
    print("2. Smallest Range Covering Elements:")
    nums = [[4, 10, 15, 24, 26], [0, 9, 12, 20], [5, 18, 22, 30]]
    problems.smallest_range_covering_elements(nums)
    print("\n" + "-" * 60 + "\n")
    
    print("3. Super Ugly Numbers:")
    problems.super_ugly_numbers(12, [2, 7, 13, 19])
    print("\n" + "=" * 60 + "\n")
    
    # 4. Interview guidance
    problems.interview_approach_guide()
    print("\n" + "=" * 60 + "\n")
    
    problems.common_mistakes()


if __name__ == "__main__":
    demonstrate_heap_interview_problems()
    
    print("\n=== HEAP INTERVIEW SUCCESS STRATEGY ===")
    
    print("\n🎯 PREPARATION ROADMAP:")
    print("Week 1: Master heap operations and kth element problems")
    print("Week 2: Practice merge problems and two-heap techniques")
    print("Week 3: Tackle sliding window and matrix problems")
    print("Week 4: Solve advanced optimization and graph problems")
    
    print("\n📚 MUST-PRACTICE PROBLEMS:")
    print("• Kth Largest Element (Easy) - Foundation")
    print("• Top K Frequent Elements (Medium) - Hash + Heap")
    print("• Merge K Sorted Lists (Hard) - Classic merge")
    print("• Find Median from Data Stream (Hard) - Two heaps")
    print("• Sliding Window Maximum (Hard) - Alternative to deque")
    print("• Last Stone Weight (Easy) - Simulation")
    print("• Reorganize String (Medium) - Greedy with heap")
    print("• Smallest Range (Hard) - Advanced merge technique")
    
    print("\n⚡ QUICK PROBLEM IDENTIFICATION:")
    print("• 'Kth largest/smallest' → Min/Max heap of size k")
    print("• 'Top k frequent' → Hash map + Min heap")
    print("• 'Merge k sorted' → Min heap with pointers")
    print("• 'Median in stream' → Two heaps (max + min)")
    print("• 'Priority scheduling' → Heap as priority queue")
    print("• 'Optimization (shortest/minimum)' → Min heap")
    
    print("\n🏆 INTERVIEW DAY TIPS:")
    print("• Clarify if you need kth largest or kth smallest")
    print("• Ask about duplicate elements and how to handle them")
    print("• Consider space constraints (heap of size k vs n)")
    print("• Explain heap choice (min vs max) before coding")
    print("• Test with edge cases (empty, single element, k > n)")
    print("• Discuss alternative approaches (QuickSelect, sorting)")
    
    print("\n📊 COMPLEXITY GOALS:")
    print("• K-element problems: O(n log k) time, O(k) space")
    print("• Merge problems: O(n log k) where n is total elements")
    print("• Two heaps: O(log n) per operation")
    print("• Streaming problems: O(log n) per insertion")
    
    print("\n🎓 ADVANCED PREPARATION:")
    print("• Study heap implementation details")
    print("• Practice with custom comparators")
    print("• Learn when heap is optimal vs alternatives")
    print("• Understand amortized analysis for heap operations")
    print("• Practice explaining heap choice in interviews")
