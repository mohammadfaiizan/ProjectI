"""
Searching & Sorting Interview Problems
=====================================

Topics: Real interview problems from top companies
Companies: Google, Facebook, Amazon, Microsoft, Apple, Netflix
Difficulty: Medium to Hard
LeetCode: Various popular problems
"""

from typing import List, Optional
import heapq
from collections import Counter

class SearchingSortingInterviewProblems:
    
    # ==========================================
    # GOOGLE PROBLEMS
    # ==========================================
    
    def find_first_bad_version(self, n: int, is_bad_version) -> int:
        """LC 278: First Bad Version (Google)
        Time: O(log n), Space: O(1)
        """
        left, right = 1, n
        
        while left < right:
            mid = left + (right - left) // 2
            
            if is_bad_version(mid):
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def search_suggestions_system(self, products: List[str], searchWord: str) -> List[List[str]]:
        """LC 1268: Search Suggestions System (Google)
        Time: O(n log n + m * log n), Space: O(1)
        """
        products.sort()
        result = []
        
        for i in range(len(searchWord)):
            prefix = searchWord[:i + 1]
            suggestions = []
            
            for product in products:
                if product.startswith(prefix):
                    suggestions.append(product)
                    if len(suggestions) == 3:
                        break
            
            result.append(suggestions)
        
        return result
    
    def meeting_rooms_ii(self, intervals: List[List[int]]) -> int:
        """LC 253: Meeting Rooms II (Google)
        Time: O(n log n), Space: O(n)
        """
        if not intervals:
            return 0
        
        intervals.sort(key=lambda x: x[0])
        heap = []
        
        for start, end in intervals:
            if heap and heap[0] <= start:
                heapq.heappop(heap)
            heapq.heappush(heap, end)
        
        return len(heap)
    
    # ==========================================
    # FACEBOOK/META PROBLEMS
    # ==========================================
    
    def merge_intervals(self, intervals: List[List[int]]) -> List[List[int]]:
        """LC 56: Merge Intervals (Facebook)
        Time: O(n log n), Space: O(1)
        """
        if not intervals:
            return []
        
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        
        return merged
    
    def insert_interval(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        """LC 57: Insert Interval (Facebook)
        Time: O(n), Space: O(1)
        """
        result = []
        i = 0
        n = len(intervals)
        
        # Add intervals before new interval
        while i < n and intervals[i][1] < newInterval[0]:
            result.append(intervals[i])
            i += 1
        
        # Merge overlapping intervals
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
        
        result.append(newInterval)
        
        # Add remaining intervals
        while i < n:
            result.append(intervals[i])
            i += 1
        
        return result
    
    def top_k_frequent_elements(self, nums: List[int], k: int) -> List[int]:
        """LC 347: Top K Frequent Elements (Facebook)
        Time: O(n log k), Space: O(n)
        """
        count = Counter(nums)
        return heapq.nlargest(k, count.keys(), key=count.get)
    
    # ==========================================
    # AMAZON PROBLEMS
    # ==========================================
    
    def two_sum_sorted(self, numbers: List[int], target: int) -> List[int]:
        """LC 167: Two Sum II - Input Array Is Sorted (Amazon)
        Time: O(n), Space: O(1)
        """
        left, right = 0, len(numbers) - 1
        
        while left < right:
            current_sum = numbers[left] + numbers[right]
            
            if current_sum == target:
                return [left + 1, right + 1]  # 1-indexed
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        
        return []
    
    def three_sum(self, nums: List[int]) -> List[List[int]]:
        """LC 15: 3Sum (Amazon)
        Time: O(n²), Space: O(1)
        """
        nums.sort()
        result = []
        n = len(nums)
        
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            left, right = i + 1, n - 1
            
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                
                if total < 0:
                    left += 1
                elif total > 0:
                    right -= 1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
        
        return result
    
    def search_rotated_sorted_array(self, nums: List[int], target: int) -> int:
        """LC 33: Search in Rotated Sorted Array (Amazon)
        Time: O(log n), Space: O(1)
        """
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                return mid
            
            if nums[left] <= nums[mid]:  # Left half is sorted
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:  # Right half is sorted
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        
        return -1
    
    # ==========================================
    # MICROSOFT PROBLEMS
    # ==========================================
    
    def sort_colors(self, nums: List[int]) -> List[int]:
        """LC 75: Sort Colors (Microsoft)
        Time: O(n), Space: O(1)
        Dutch National Flag Algorithm
        """
        nums = nums.copy()
        left = curr = 0
        right = len(nums) - 1
        
        while curr <= right:
            if nums[curr] == 0:
                nums[left], nums[curr] = nums[curr], nums[left]
                left += 1
                curr += 1
            elif nums[curr] == 2:
                nums[curr], nums[right] = nums[right], nums[curr]
                right -= 1
            else:
                curr += 1
        
        return nums
    
    def find_kth_largest(self, nums: List[int], k: int) -> int:
        """LC 215: Kth Largest Element (Microsoft)
        Time: O(n) average, O(n²) worst, Space: O(1)
        """
        def partition(left, right, pivot_index):
            pivot_value = nums[pivot_index]
            nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
            
            store_index = left
            for i in range(left, right):
                if nums[i] < pivot_value:
                    nums[store_index], nums[i] = nums[i], nums[store_index]
                    store_index += 1
            
            nums[right], nums[store_index] = nums[store_index], nums[right]
            return store_index
        
        def select(left, right, k_smallest):
            if left == right:
                return nums[left]
            
            pivot_index = left + (right - left) // 2
            pivot_index = partition(left, right, pivot_index)
            
            if k_smallest == pivot_index:
                return nums[k_smallest]
            elif k_smallest < pivot_index:
                return select(left, pivot_index - 1, k_smallest)
            else:
                return select(pivot_index + 1, right, k_smallest)
        
        return select(0, len(nums) - 1, len(nums) - k)
    
    # ==========================================
    # APPLE PROBLEMS
    # ==========================================
    
    def search_range(self, nums: List[int], target: int) -> List[int]:
        """LC 34: Find First and Last Position (Apple)
        Time: O(log n), Space: O(1)
        """
        def find_boundary(is_first):
            left, right = 0, len(nums) - 1
            boundary_index = -1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                if nums[mid] == target:
                    boundary_index = mid
                    if is_first:
                        right = mid - 1
                    else:
                        left = mid + 1
                elif nums[mid] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            
            return boundary_index
        
        first_pos = find_boundary(True)
        if first_pos == -1:
            return [-1, -1]
        
        last_pos = find_boundary(False)
        return [first_pos, last_pos]
    
    def longest_consecutive_sequence(self, nums: List[int]) -> int:
        """LC 128: Longest Consecutive Sequence (Apple)
        Time: O(n), Space: O(n)
        """
        if not nums:
            return 0
        
        num_set = set(nums)
        max_length = 0
        
        for num in num_set:
            if num - 1 not in num_set:  # Start of sequence
                current_num = num
                current_length = 1
                
                while current_num + 1 in num_set:
                    current_num += 1
                    current_length += 1
                
                max_length = max(max_length, current_length)
        
        return max_length

# Test Examples
def run_examples():
    problems = SearchingSortingInterviewProblems()
    
    print("=== SEARCHING & SORTING INTERVIEW PROBLEMS ===\n")
    
    print("1. GOOGLE PROBLEMS:")
    # Search suggestions
    products = ["mobile","mouse","moneypot","monitor","mousepad"]
    search_word = "mouse"
    suggestions = problems.search_suggestions_system(products, search_word)
    print(f"Search suggestions for '{search_word}': {suggestions}")
    
    print("\n2. FACEBOOK PROBLEMS:")
    # Merge intervals
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    merged = problems.merge_intervals(intervals)
    print(f"Merged intervals: {merged}")
    
    # Top K frequent
    nums = [1,1,1,2,2,3]
    k = 2
    top_k = problems.top_k_frequent_elements(nums, k)
    print(f"Top {k} frequent: {top_k}")
    
    print("\n3. AMAZON PROBLEMS:")
    # 3Sum
    nums = [-1,0,1,2,-1,-4]
    three_sum = problems.three_sum(nums)
    print(f"3Sum result: {three_sum}")
    
    print("\n4. MICROSOFT PROBLEMS:")
    # Sort colors
    colors = [2,0,2,1,1,0]
    sorted_colors = problems.sort_colors(colors)
    print(f"Sorted colors: {sorted_colors}")
    
    # Kth largest
    nums = [3,2,1,5,6,4]
    k = 2
    kth_largest = problems.find_kth_largest(nums, k)
    print(f"{k}th largest: {kth_largest}")
    
    print("\n5. APPLE PROBLEMS:")
    # Search range
    nums = [5,7,7,8,8,10]
    target = 8
    range_result = problems.search_range(nums, target)
    print(f"Range of {target}: {range_result}")

if __name__ == "__main__":
    run_examples() 