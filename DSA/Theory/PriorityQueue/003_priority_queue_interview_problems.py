"""
Priority Queue Interview Problems - Most Asked Questions
=======================================================

Topics: Common priority queue interview questions with solutions and explanations
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber, LinkedIn
Difficulty: Easy to Hard
Time Complexity: Usually O(n log k) or O(n log n)
Space Complexity: O(k) or O(n) for priority queue storage
"""

from typing import List, Optional, Dict, Tuple, Any, Set
import heapq
from collections import Counter, defaultdict
import math

class PriorityQueueInterviewProblems:
    
    def __init__(self):
        """Initialize with problem tracking"""
        self.problem_count = 0
        self.solution_steps = []
    
    # ==========================================
    # 1. CLASSIC PRIORITY QUEUE PROBLEMS
    # ==========================================
    
    def merge_k_sorted_lists(self, lists: List[List[int]]) -> List[int]:
        """
        Merge k Sorted Lists using Priority Queue
        
        Company: Google, Facebook, Amazon, Microsoft
        Difficulty: Hard
        Time: O(n log k), Space: O(k)
        
        LeetCode 23: Most fundamental priority queue merging problem
        """
        if not lists:
            return []
        
        # Min heap: (value, list_index, element_index)
        min_heap = []
        result = []
        
        print(f"Merging {len(lists)} sorted lists using priority queue:")
        for i, lst in enumerate(lists):
            print(f"   List {i}: {lst}")
        print()
        
        # Initialize heap with first element from each non-empty list
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(min_heap, (lst[0], i, 0))
                print(f"Initial: Added {lst[0]} from list {i}")
        
        print(f"Initial heap: {min_heap}")
        print()
        
        step = 1
        while min_heap:
            value, list_idx, elem_idx = heapq.heappop(min_heap)
            result.append(value)
            
            print(f"Step {step}: Extracted minimum {value} from list {list_idx}")
            
            # Add next element from same list if available
            if elem_idx + 1 < len(lists[list_idx]):
                next_value = lists[list_idx][elem_idx + 1]
                heapq.heappush(min_heap, (next_value, list_idx, elem_idx + 1))
                print(f"   Added next element {next_value} from list {list_idx}")
            
            print(f"   Result: {result}")
            print(f"   Heap: {min_heap}")
            print()
            step += 1
        
        print(f"Final merged result: {result}")
        return result
    
    def kth_largest_element_in_stream(self):
        """
        Kth Largest Element in a Stream
        
        Company: Facebook, Amazon, Google
        Difficulty: Easy
        Time: O(log k) per add, Space: O(k)
        
        LeetCode 703: Design data structure for streaming kth largest
        """
        
        class KthLargest:
            def __init__(self, k: int, nums: List[int]):
                self.k = k
                self.heap = nums
                heapq.heapify(self.heap)
                
                # Keep only k largest elements
                while len(self.heap) > k:
                    heapq.heappop(self.heap)
                
                print(f"Initialized KthLargest with k={k}, nums={nums}")
                print(f"Initial heap (k largest): {self.heap}")
            
            def add(self, val: int) -> int:
                """Add value and return kth largest"""
                print(f"\nAdding value: {val}")
                
                heapq.heappush(self.heap, val)
                print(f"   After adding: {self.heap}")
                
                if len(self.heap) > self.k:
                    removed = heapq.heappop(self.heap)
                    print(f"   Removed smallest: {removed}")
                    print(f"   Heap after removal: {self.heap}")
                
                kth_largest = self.heap[0]
                print(f"   {self.k}th largest element: {kth_largest}")
                
                return kth_largest
        
        print("=== KTH LARGEST ELEMENT IN STREAM ===")
        
        # Example usage
        kth_largest = KthLargest(3, [4, 5, 8, 2])
        
        values_to_add = [3, 5, 10, 9, 4]
        for val in values_to_add:
            result = kth_largest.add(val)
        
        return kth_largest
    
    def top_k_frequent_words(self, words: List[str], k: int) -> List[str]:
        """
        Top K Frequent Words
        
        Company: Amazon, Facebook, Google, Uber
        Difficulty: Medium
        Time: O(n log k), Space: O(n)
        
        LeetCode 692: Priority queue with custom comparator
        """
        # Count word frequencies
        word_count = Counter(words)
        
        print(f"Finding top {k} frequent words from: {words}")
        print(f"Word frequencies: {dict(word_count)}")
        print()
        
        # Min heap with custom comparator
        # For equal frequencies, lexicographically larger word has lower priority
        min_heap = []
        
        for word, frequency in word_count.items():
            print(f"Processing word '{word}' with frequency {frequency}")
            
            if len(min_heap) < k:
                # Heap not full, add word
                # Use (-frequency, word) so that:
                # - Higher frequency words have higher priority (smaller -frequency)
                # - For same frequency, lexicographically smaller words have higher priority
                heapq.heappush(min_heap, (frequency, word))
                print(f"   Added to heap: {min_heap}")
            else:
                # Heap full, check if current word should replace minimum
                min_freq, min_word = min_heap[0]
                
                # Replace if current has higher frequency
                # Or same frequency but lexicographically smaller
                if (frequency > min_freq or 
                    (frequency == min_freq and word < min_word)):
                    
                    replaced = heapq.heapreplace(min_heap, (frequency, word))
                    print(f"   Replaced {replaced} with ({frequency}, '{word}')")
                else:
                    print(f"   Not added: frequency {frequency} or lexicographic order doesn't qualify")
            
            print(f"   Current top {len(min_heap)} frequent: {[(f, w) for f, w in sorted(min_heap, reverse=True)]}")
            print()
        
        # Extract words in descending order of frequency
        # For same frequency, lexicographically ascending order
        result_tuples = []
        while min_heap:
            freq, word = heapq.heappop(min_heap)
            result_tuples.append((freq, word))
        
        # Sort by frequency (descending), then by word (ascending)
        result_tuples.sort(key=lambda x: (-x[0], x[1]))
        result = [word for freq, word in result_tuples]
        
        print(f"Top {k} frequent words: {result}")
        return result
    
    def ugly_number_ii(self, n: int) -> int:
        """
        Ugly Number II
        
        Company: Facebook, Amazon, Google
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        LeetCode 264: Generate sequence using priority queue
        """
        print(f"Finding the {n}th ugly number")
        print("Ugly numbers: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, ...")
        print("(Only have prime factors 2, 3, and 5)")
        print()
        
        # Min heap to generate ugly numbers in ascending order
        min_heap = [1]
        seen = {1}
        factors = [2, 3, 5]
        
        print("Generating ugly numbers using priority queue:")
        
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
                    print(f"   Generated: {ugly} × {factor} = {new_ugly}")
            
            # Show next few in heap
            if len(min_heap) >= 3:
                next_three = sorted(min_heap)[:3]
                print(f"   Next in sequence: {next_three}...")
            print()
        
        return -1
    
    # ==========================================
    # 2. MEDIAN AND STATISTICS PROBLEMS
    # ==========================================
    
    def find_median_from_data_stream(self):
        """
        Find Median from Data Stream
        
        Company: Google, Facebook, Amazon
        Difficulty: Hard
        Time: O(log n) addNum, O(1) findMedian
        Space: O(n)
        
        LeetCode 295: Two heaps technique for streaming median
        """
        
        class MedianFinder:
            def __init__(self):
                # Max heap for smaller half (use negative values)
                self.max_heap = []  # smaller half
                # Min heap for larger half
                self.min_heap = []  # larger half
                
                print("MedianFinder initialized with two heaps")
                print("Max heap (smaller half) and Min heap (larger half)")
            
            def addNum(self, num: int) -> None:
                """Add number to data structure"""
                print(f"\nAdding number: {num}")
                
                # Always maintain: len(max_heap) - len(min_heap) ∈ {0, 1}
                
                # Add to appropriate heap
                if not self.max_heap or num <= -self.max_heap[0]:
                    heapq.heappush(self.max_heap, -num)
                    print(f"   Added to max_heap (smaller half)")
                else:
                    heapq.heappush(self.min_heap, num)
                    print(f"   Added to min_heap (larger half)")
                
                # Balance heaps
                self._balance()
                
                self._display_state()
                print(f"   Current median: {self.findMedian()}")
            
            def findMedian(self) -> float:
                """Find median of all numbers"""
                if len(self.max_heap) == len(self.min_heap):
                    if not self.max_heap:  # Both empty
                        return 0.0
                    return (-self.max_heap[0] + self.min_heap[0]) / 2.0
                else:
                    # max_heap has one more element
                    return float(-self.max_heap[0])
            
            def _balance(self) -> None:
                """Balance heap sizes"""
                if len(self.max_heap) > len(self.min_heap) + 1:
                    # Move from max_heap to min_heap
                    value = -heapq.heappop(self.max_heap)
                    heapq.heappush(self.min_heap, value)
                    print(f"   Rebalanced: moved {value} from max_heap to min_heap")
                elif len(self.min_heap) > len(self.max_heap):
                    # Move from min_heap to max_heap
                    value = heapq.heappop(self.min_heap)
                    heapq.heappush(self.max_heap, -value)
                    print(f"   Rebalanced: moved {value} from min_heap to max_heap")
            
            def _display_state(self) -> None:
                """Display current state of heaps"""
                smaller_half = [-x for x in self.max_heap]
                larger_half = list(self.min_heap)
                print(f"   Max heap (smaller): {smaller_half}")
                print(f"   Min heap (larger): {larger_half}")
                print(f"   Sizes: {len(self.max_heap)}, {len(self.min_heap)}")
        
        print("=== FIND MEDIAN FROM DATA STREAM ===")
        median_finder = MedianFinder()
        
        # Test with stream of numbers
        numbers = [1, 2, 3, 4, 5, 6]
        for num in numbers:
            median_finder.addNum(num)
        
        return median_finder
    
    def sliding_window_median(self, nums: List[int], k: int) -> List[float]:
        """
        Sliding Window Median
        
        Company: Google, Facebook
        Difficulty: Hard
        Time: O(n log k), Space: O(k)
        
        LeetCode 480: Moving window median using two heaps
        """
        if not nums or k <= 0:
            return []
        
        from collections import deque
        
        def find_median_of_window(window: List[int]) -> float:
            """Find median of current window using sorting"""
            sorted_window = sorted(window)
            n = len(sorted_window)
            if n % 2 == 1:
                return float(sorted_window[n // 2])
            else:
                return (sorted_window[n // 2 - 1] + sorted_window[n // 2]) / 2.0
        
        result = []
        window = deque()
        
        print(f"Sliding Window Median: nums={nums}, k={k}")
        print()
        
        for i in range(len(nums)):
            # Add current element to window
            window.append(nums[i])
            
            # Remove elements outside window
            if len(window) > k:
                removed = window.popleft()
                print(f"Step {i+1}: Removed {removed}, added {nums[i]}")
            else:
                print(f"Step {i+1}: Added {nums[i]}")
            
            # Calculate median if window is complete
            if len(window) == k:
                median = find_median_of_window(list(window))
                result.append(median)
                
                window_list = list(window)
                print(f"   Window: {window_list}")
                print(f"   Sorted: {sorted(window_list)}")
                print(f"   Median: {median}")
            
            print()
        
        print(f"Sliding window medians: {result}")
        return result
    
    # ==========================================
    # 3. SCHEDULING AND OPTIMIZATION PROBLEMS
    # ==========================================
    
    def task_scheduler(self, tasks: List[str], n: int) -> int:
        """
        Task Scheduler
        
        Company: Facebook, Amazon
        Difficulty: Medium
        Time: O(m), Space: O(1) where m is execution time
        
        LeetCode 621: Schedule tasks with cooling period
        """
        # Count task frequencies
        task_count = Counter(tasks)
        
        print(f"Task Scheduler: tasks={tasks}, cooling_period={n}")
        print(f"Task frequencies: {dict(task_count)}")
        print()
        
        # Max heap of frequencies (negate for max heap behavior)
        max_heap = [-count for count in task_count.values()]
        heapq.heapify(max_heap)
        
        time = 0
        execution_log = []
        
        while max_heap:
            temp_storage = []
            cycle_time = 0
            
            print(f"Time {time}: Starting new cycle")
            print(f"   Available task frequencies: {[-x for x in max_heap]}")
            
            # Execute tasks for n+1 time slots (or until no tasks left)
            for slot in range(n + 1):
                if max_heap:
                    # Execute most frequent task
                    freq = -heapq.heappop(max_heap)
                    task_type = f"Task_{freq}"  # Simplified task naming
                    execution_log.append(task_type)
                    
                    print(f"   Slot {slot}: Execute {task_type} (frequency was {freq})")
                    
                    # Decrease frequency and store for later
                    if freq > 1:
                        temp_storage.append(-(freq - 1))
                        print(f"     Task frequency reduced to {freq - 1}")
                    
                    cycle_time += 1
                else:
                    # No tasks available, must idle
                    if temp_storage:  # Only idle if there are more tasks to do
                        execution_log.append("idle")
                        print(f"   Slot {slot}: Idle (cooling period)")
                        cycle_time += 1
            
            # Add tasks back to heap
            for task_freq in temp_storage:
                heapq.heappush(max_heap, task_freq)
            
            time += cycle_time
            print(f"   Cycle completed, total time: {time}")
            print(f"   Remaining task frequencies: {[-x for x in max_heap]}")
            print()
        
        print(f"Task execution order: {execution_log}")
        print(f"Total execution time: {time}")
        
        return time
    
    def meeting_rooms_ii(self, intervals: List[List[int]]) -> int:
        """
        Meeting Rooms II
        
        Company: Google, Facebook, Amazon
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        LeetCode 253: Minimum meeting rooms using priority queue
        """
        if not intervals:
            return 0
        
        # Sort meetings by start time
        intervals.sort()
        
        # Min heap to track meeting end times
        min_heap = []
        
        print(f"Meeting Rooms II: Finding minimum rooms needed")
        print(f"Meetings (sorted by start time): {intervals}")
        print()
        
        for i, (start, end) in enumerate(intervals):
            print(f"Processing meeting {i+1}: [{start}, {end})")
            
            # Remove meetings that have ended
            while min_heap and min_heap[0] <= start:
                ended_meeting = heapq.heappop(min_heap)
                print(f"   Meeting ending at {ended_meeting} finished, room freed")
            
            # Add current meeting's end time
            heapq.heappush(min_heap, end)
            print(f"   Added meeting ending at {end}")
            
            print(f"   Current ongoing meetings (end times): {sorted(min_heap)}")
            print(f"   Rooms needed: {len(min_heap)}")
            print()
        
        max_rooms = len(min_heap)
        print(f"Maximum meeting rooms needed: {max_rooms}")
        
        return max_rooms
    
    def reorganize_string(self, s: str) -> str:
        """
        Reorganize String
        
        Company: Google, Facebook, Amazon
        Difficulty: Medium
        Time: O(n log k), Space: O(k) where k is unique characters
        
        LeetCode 767: Rearrange characters using priority queue
        """
        # Count character frequencies
        char_count = Counter(s)
        
        print(f"Reorganizing string: '{s}'")
        print(f"Character frequencies: {dict(char_count)}")
        
        # Check if reorganization is possible
        max_freq = max(char_count.values())
        if max_freq > (len(s) + 1) // 2:
            print(f"Impossible: max frequency {max_freq} > {(len(s) + 1) // 2}")
            return ""
        
        # Max heap with character frequencies
        max_heap = [(-count, char) for char, count in char_count.items()]
        heapq.heapify(max_heap)
        
        print(f"Initial max heap: {[(-count, char) for count, char in max_heap]}")
        print()
        
        result = []
        prev_count = 0
        prev_char = None
        
        step = 1
        while max_heap:
            print(f"Step {step}:")
            
            # Get most frequent character
            count, char = heapq.heappop(max_heap)
            count = -count  # Convert back to positive
            
            result.append(char)
            print(f"   Added '{char}' (frequency was {count})")
            print(f"   Result so far: '{''.join(result)}'")
            
            # Add back previous character if it still has uses
            if prev_count > 0:
                heapq.heappush(max_heap, (-prev_count, prev_char))
                print(f"   Added back '{prev_char}' with count {prev_count}")
            
            # Update previous character info
            prev_char = char
            prev_count = count - 1
            
            heap_display = [(-c, ch) for c, ch in max_heap]
            print(f"   Heap after step: {heap_display}")
            print()
            step += 1
        
        # Check if we used all characters
        if len(result) != len(s):
            print("Failed to use all characters")
            return ""
        
        final_result = ''.join(result)
        print(f"Successfully reorganized: '{final_result}'")
        
        # Verify no adjacent duplicates
        for i in range(len(final_result) - 1):
            if final_result[i] == final_result[i + 1]:
                print(f"Error: Adjacent duplicates at position {i}")
                return ""
        
        return final_result
    
    # ==========================================
    # 4. ADVANCED PRIORITY QUEUE PROBLEMS
    # ==========================================
    
    def smallest_range_covering_elements(self, nums: List[List[int]]) -> List[int]:
        """
        Smallest Range Covering Elements from K Lists
        
        Company: Google, Amazon
        Difficulty: Hard
        Time: O(n log k), Space: O(k)
        
        LeetCode 632: Find range covering at least one element from each list
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
        while len(min_heap) == len(nums):  # All lists must be represented
            min_val, list_idx, elem_idx = heapq.heappop(min_heap)
            
            current_range_size = max_val - min_val + 1
            print(f"Step {step}: Range [{min_val}, {max_val}], size = {current_range_size}")
            
            # Update smallest range if current is better
            if max_val - min_val < range_end - range_start:
                range_start, range_end = min_val, max_val
                print(f"   New best range: [{range_start}, {range_end}]")
            
            # Try to add next element from same list
            if elem_idx + 1 < len(nums[list_idx]):
                next_val = nums[list_idx][elem_idx + 1]
                heapq.heappush(min_heap, (next_val, list_idx, elem_idx + 1))
                max_val = max(max_val, next_val)
                print(f"   Added {next_val} from list {list_idx}, new max = {max_val}")
            else:
                print(f"   List {list_idx} exhausted, cannot improve further")
                break
            
            print()
            step += 1
        
        result = [range_start, range_end]
        print(f"Smallest range covering all lists: {result}")
        return result
    
    def ipo_maximize_capital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        """
        IPO - Maximize Capital
        
        Company: Google, Amazon
        Difficulty: Hard
        Time: O(n log n), Space: O(n)
        
        LeetCode 502: Choose projects to maximize capital
        """
        print(f"IPO Capital Maximization:")
        print(f"Max projects: {k}, Initial capital: {w}")
        print(f"Profits: {profits}")
        print(f"Capital required: {capital}")
        print()
        
        # Create projects list and sort by capital requirement
        projects = list(zip(capital, profits))
        projects.sort()  # Sort by capital requirement
        
        print("Projects sorted by capital requirement:")
        for i, (cap, prof) in enumerate(projects):
            print(f"   Project {i}: Capital={cap}, Profit={prof}")
        print()
        
        # Min heap for available projects (by capital requirement)
        available_heap = []
        
        # Max heap for profitable projects (by profit)
        profit_heap = []
        
        current_capital = w
        project_index = 0
        
        for project_num in range(k):
            print(f"Selecting project {project_num + 1}:")
            print(f"   Current capital: {current_capital}")
            
            # Add all affordable projects to profit heap
            while project_index < len(projects) and projects[project_index][0] <= current_capital:
                cap, prof = projects[project_index]
                heapq.heappush(profit_heap, -prof)  # Negative for max heap
                print(f"   Project with capital={cap}, profit={prof} now affordable")
                project_index += 1
            
            # If no projects available, break
            if not profit_heap:
                print(f"   No affordable projects available")
                break
            
            # Choose most profitable project
            max_profit = -heapq.heappop(profit_heap)
            current_capital += max_profit
            
            print(f"   Selected project with profit={max_profit}")
            print(f"   New capital: {current_capital}")
            print()
        
        print(f"Final maximized capital: {current_capital}")
        return current_capital
    
    # ==========================================
    # 5. INTERVIEW STRATEGY AND TIPS
    # ==========================================
    
    def interview_approach_guide(self) -> None:
        """
        Comprehensive guide for tackling priority queue interview problems
        """
        print("=== PRIORITY QUEUE INTERVIEW APPROACH GUIDE ===")
        print()
        
        print("🎯 STEP 1: PROBLEM RECOGNITION")
        print("Look for these keywords/patterns:")
        print("• 'K largest/smallest', 'top k', 'kth element'")
        print("• 'Merge sorted', 'combine sequences'")
        print("• 'Median', 'streaming data', 'running statistics'")
        print("• 'Schedule', 'priority', 'resource allocation'")
        print("• 'Minimum/maximum range', 'optimization'")
        print("• 'Meeting rooms', 'interval scheduling'")
        print()
        
        print("🎯 STEP 2: CHOOSE PRIORITY QUEUE TYPE")
        print("Decision framework:")
        print("• K largest → Min heap of size k")
        print("• K smallest → Max heap of size k")
        print("• Streaming median → Two heaps (max + min)")
        print("• Merge operations → Min heap with indices")
        print("• Scheduling → Heap ordered by time/priority")
        print("• Range problems → Min heap + tracking max")
        print()
        
        print("🎯 STEP 3: IMPLEMENTATION PATTERNS")
        print()
        print("A) K-Element Pattern:")
        print("   heap = []")
        print("   for element in stream:")
        print("       if len(heap) < k:")
        print("           heappush(heap, element)")
        print("       elif element > heap[0]:  # for k largest")
        print("           heapreplace(heap, element)")
        print()
        print("B) Merge Pattern:")
        print("   heap = [(arr[0], 0, 0) for arr in arrays if arr]")
        print("   while heap:")
        print("       val, arr_idx, elem_idx = heappop(heap)")
        print("       result.append(val)")
        print("       # Add next element from same array")
        print()
        print("C) Two Heaps (Median):")
        print("   max_heap, min_heap = [], []")
        print("   # Maintain balance: |max_heap| - |min_heap| ≤ 1")
        print("   # max_heap stores smaller half, min_heap stores larger half")
        print()
        
        print("🎯 STEP 4: OPTIMIZATION STRATEGIES")
        print("• Use heapreplace instead of heappop + heappush")
        print("• Consider custom comparators for complex objects")
        print("• Implement lazy deletion for sliding window problems")
        print("• Cache expensive calculations (distances, priorities)")
        print("• Use appropriate data structures for tie-breaking")
        print()
        
        print("🎯 STEP 5: TESTING APPROACH")
        print("Essential test cases:")
        print("• Empty input or k = 0")
        print("• k greater than input size")
        print("• All elements equal")
        print("• Single element")
        print("• Extreme values (very large/small)")
        print("• Duplicate elements and tie-breaking")
    
    def common_mistakes(self) -> None:
        """
        Common mistakes in priority queue interview problems
        """
        print("=== COMMON PRIORITY QUEUE INTERVIEW MISTAKES ===")
        print()
        
        print("❌ MISTAKE 1: Wrong heap type for k-element problems")
        print("Issue: Using max heap for k largest elements")
        print("Correct: Use min heap of size k for k largest elements")
        print("Why: Min heap keeps k largest by removing smaller elements")
        print()
        
        print("❌ MISTAKE 2: Forgetting Python heapq is min heap only")
        print("Issue: Expecting max heap behavior from heapq")
        print("Solution: Negate values for max heap: heappush(heap, -value)")
        print("Example: For max heap, push -priority instead of priority")
        print()
        
        print("❌ MISTAKE 3: Not handling tie-breaking correctly")
        print("Issue: Undefined behavior when priorities are equal")
        print("Solution: Use tuples with multiple criteria")
        print("Example: (priority, timestamp, data) for stable ordering")
        print()
        
        print("❌ MISTAKE 4: Inefficient heap operations")
        print("Issue: Using separate heappop and heappush operations")
        print("Solution: Use heapreplace for atomic replace operation")
        print("Benefit: More efficient and maintains heap invariant")
        print()
        
        print("❌ MISTAKE 5: Incorrect two-heap balance in median problems")
        print("Issue: Not maintaining proper size relationship")
        print("Rule: |max_heap_size - min_heap_size| ≤ 1")
        print("Solution: Always rebalance after insertion")
        print()
        
        print("❌ MISTAKE 6: Memory issues with large heaps")
        print("Issue: Not considering space complexity")
        print("Solution: Use bounded heaps when possible")
        print("Example: Heap of size k instead of size n for k-element problems")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_priority_queue_interview_problems():
    """Demonstrate key priority queue interview problems"""
    print("=== PRIORITY QUEUE INTERVIEW PROBLEMS DEMONSTRATION ===\n")
    
    problems = PriorityQueueInterviewProblems()
    
    # 1. Classic problems
    print("=== CLASSIC PRIORITY QUEUE PROBLEMS ===")
    
    print("1. Merge K Sorted Lists:")
    problems.merge_k_sorted_lists([[1, 4, 5], [1, 3, 4], [2, 6]])
    print("\n" + "-" * 60 + "\n")
    
    print("2. Kth Largest Element in Stream:")
    problems.kth_largest_element_in_stream()
    print("\n" + "-" * 60 + "\n")
    
    print("3. Top K Frequent Words:")
    problems.top_k_frequent_words(["i", "love", "leetcode", "i", "love", "coding"], 2)
    print("\n" + "-" * 60 + "\n")
    
    print("4. Ugly Number II:")
    problems.ugly_number_ii(10)
    print("\n" + "=" * 60 + "\n")
    
    # 2. Median problems
    print("=== MEDIAN AND STATISTICS PROBLEMS ===")
    
    print("1. Find Median from Data Stream:")
    problems.find_median_from_data_stream()
    print("\n" + "-" * 60 + "\n")
    
    print("2. Sliding Window Median:")
    problems.sliding_window_median([1, 3, -1, -3, 5, 3, 6, 7], 4)
    print("\n" + "=" * 60 + "\n")
    
    # 3. Scheduling problems
    print("=== SCHEDULING AND OPTIMIZATION PROBLEMS ===")
    
    print("1. Task Scheduler:")
    problems.task_scheduler(['A','A','A','B','B','B'], 2)
    print("\n" + "-" * 60 + "\n")
    
    print("2. Meeting Rooms II:")
    problems.meeting_rooms_ii([[0,30],[5,10],[15,20]])
    print("\n" + "-" * 60 + "\n")
    
    print("3. Reorganize String:")
    problems.reorganize_string("aab")
    print("\n" + "=" * 60 + "\n")
    
    # 4. Advanced problems
    print("=== ADVANCED PRIORITY QUEUE PROBLEMS ===")
    
    print("1. Smallest Range Covering Elements:")
    nums = [[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]
    problems.smallest_range_covering_elements(nums)
    print("\n" + "-" * 60 + "\n")
    
    print("2. IPO Maximize Capital:")
    problems.ipo_maximize_capital(2, 0, [1,2,3], [0,1,1])
    print("\n" + "=" * 60 + "\n")
    
    # 5. Interview guidance
    problems.interview_approach_guide()
    print("\n" + "=" * 60 + "\n")
    
    problems.common_mistakes()


if __name__ == "__main__":
    demonstrate_priority_queue_interview_problems()
    
    print("\n=== PRIORITY QUEUE INTERVIEW SUCCESS STRATEGY ===")
    
    print("\n🎯 PREPARATION ROADMAP:")
    print("Week 1: Master basic priority queue operations and k-element problems")
    print("Week 2: Practice merge problems and streaming data techniques")
    print("Week 3: Tackle scheduling and interval management problems")
    print("Week 4: Solve advanced optimization and multi-constraint problems")
    
    print("\n📚 MUST-PRACTICE PROBLEMS:")
    print("• Merge K Sorted Lists (Hard) - Foundation merging")
    print("• Kth Largest Element in Stream (Easy) - Basic streaming")
    print("• Find Median from Data Stream (Hard) - Two heaps technique")
    print("• Top K Frequent Elements (Medium) - Frequency + heap")
    print("• Task Scheduler (Medium) - Greedy scheduling")
    print("• Meeting Rooms II (Medium) - Interval scheduling")
    print("• Reorganize String (Medium) - Greedy character placement")
    print("• Smallest Range (Hard) - Advanced merge technique")
    
    print("\n⚡ QUICK PROBLEM IDENTIFICATION:")
    print("• 'K largest/smallest/frequent' → Heap of size k")
    print("• 'Merge k sorted' → Min heap with pointers")
    print("• 'Streaming median' → Two heaps")
    print("• 'Meeting rooms/intervals' → Heap by time")
    print("• 'Reorganize/schedule' → Max heap by frequency/priority")
    print("• 'Range covering' → Min heap + max tracking")
    
    print("\n🏆 INTERVIEW DAY TIPS:")
    print("• Clarify whether you need kth largest or kth smallest")
    print("• Ask about handling duplicates and tie-breaking")
    print("• Discuss space optimization (heap of size k vs n)")
    print("• Explain heap choice (min vs max) before implementing")
    print("• Consider edge cases: empty input, k > n, single element")
    print("• Mention alternative approaches when appropriate")
    
    print("\n📊 COMPLEXITY TARGETS:")
    print("• K-element problems: O(n log k) time, O(k) space")
    print("• Merge problems: O(n log k) where n = total elements")
    print("• Streaming problems: O(log n) per operation")
    print("• Scheduling problems: O(n log n) preprocessing + O(log n) per event")
    
    print("\n🎓 ADVANCED PREPARATION:")
    print("• Practice implementing heaps from scratch")
    print("• Study decrease-key operations and Fibonacci heaps")
    print("• Learn about persistent and concurrent priority queues")
    print("• Understand when priority queues are optimal vs alternatives")
    print("• Practice explaining trade-offs between different approaches")
