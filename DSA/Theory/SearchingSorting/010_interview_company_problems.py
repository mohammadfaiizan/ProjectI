"""
Company-Specific Searching & Sorting Interview Problems
======================================================

Topics: Real problems from top tech companies
Companies: Google, Facebook, Amazon, Microsoft, Apple, Netflix, Uber
Difficulty: Medium to Hard
LeetCode: Multiple problems from actual interviews
"""

from typing import List, Optional, Dict
import heapq
from collections import defaultdict, Counter

class CompanyInterviewProblems:
    
    # ==========================================
    # GOOGLE PROBLEMS
    # ==========================================
    
    def merge_k_sorted_lists(self, lists: List[List[int]]) -> List[int]:
        """LC 23: Merge k Sorted Lists (Google)
        Time: O(N log k), Space: O(k)
        """
        if not lists:
            return []
        
        heap = []
        
        # Initialize heap with first element from each list
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst[0], i, 0))
        
        result = []
        
        while heap:
            val, list_idx, elem_idx = heapq.heappop(heap)
            result.append(val)
            
            # Add next element from the same list
            if elem_idx + 1 < len(lists[list_idx]):
                heapq.heappush(heap, (
                    lists[list_idx][elem_idx + 1],
                    list_idx,
                    elem_idx + 1
                ))
        
        return result
    
    def find_median_data_stream(self) -> 'MedianFinder':
        """LC 295: Find Median from Data Stream (Google)
        Design: O(log n) insertion, O(1) median
        """
        return MedianFinder()
    
    def meeting_rooms_ii(self, intervals: List[List[int]]) -> int:
        """LC 253: Meeting Rooms II (Google)
        Time: O(n log n), Space: O(n)
        """
        if not intervals:
            return 0
        
        intervals.sort(key=lambda x: x[0])
        min_heap = []
        
        for start, end in intervals:
            if min_heap and min_heap[0] <= start:
                heapq.heappop(min_heap)
            heapq.heappush(min_heap, end)
        
        return len(min_heap)
    
    def longest_increasing_subsequence(self, nums: List[int]) -> int:
        """LC 300: Longest Increasing Subsequence (Google)
        Time: O(n log n), Space: O(n)
        """
        if not nums:
            return 0
        
        tails = []
        
        for num in nums:
            left, right = 0, len(tails)
            
            # Binary search for insertion position
            while left < right:
                mid = (left + right) // 2
                if tails[mid] < num:
                    left = mid + 1
                else:
                    right = mid
            
            # If num is larger than all elements, append it
            if left == len(tails):
                tails.append(num)
            else:
                tails[left] = num
        
        return len(tails)
    
    # ==========================================
    # FACEBOOK/META PROBLEMS
    # ==========================================
    
    def k_closest_points_to_origin(self, points: List[List[int]], k: int) -> List[List[int]]:
        """LC 973: K Closest Points to Origin (Facebook)
        Time: O(n log k), Space: O(k)
        """
        def distance_squared(point):
            return point[0] ** 2 + point[1] ** 2
        
        max_heap = []
        
        for point in points:
            dist = distance_squared(point)
            
            if len(max_heap) < k:
                heapq.heappush(max_heap, (-dist, point))
            elif -dist > max_heap[0][0]:
                heapq.heappop(max_heap)
                heapq.heappush(max_heap, (-dist, point))
        
        return [point for _, point in max_heap]
    
    def valid_palindrome_ii(self, s: str) -> bool:
        """LC 680: Valid Palindrome II (Facebook)
        Time: O(n), Space: O(1)
        """
        def is_palindrome(left, right):
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        left, right = 0, len(s) - 1
        
        while left < right:
            if s[left] != s[right]:
                # Try removing either left or right character
                return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
            left += 1
            right -= 1
        
        return True
    
    def subarray_sum_equals_k(self, nums: List[int], k: int) -> int:
        """LC 560: Subarray Sum Equals K (Facebook)
        Time: O(n), Space: O(n)
        """
        count = 0
        sum_count = defaultdict(int)
        sum_count[0] = 1
        
        current_sum = 0
        for num in nums:
            current_sum += num
            
            if current_sum - k in sum_count:
                count += sum_count[current_sum - k]
            
            sum_count[current_sum] += 1
        
        return count
    
    # ==========================================
    # AMAZON PROBLEMS
    # ==========================================
    
    def two_sum(self, nums: List[int], target: int) -> List[int]:
        """LC 1: Two Sum (Amazon)
        Time: O(n), Space: O(n)
        """
        num_map = {}
        
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_map:
                return [num_map[complement], i]
            num_map[num] = i
        
        return []
    
    def three_sum_closest(self, nums: List[int], target: int) -> int:
        """LC 16: 3Sum Closest (Amazon)
        Time: O(n²), Space: O(1)
        """
        nums.sort()
        n = len(nums)
        closest_sum = float('inf')
        
        for i in range(n - 2):
            left, right = i + 1, n - 1
            
            while left < right:
                current_sum = nums[i] + nums[left] + nums[right]
                
                if abs(current_sum - target) < abs(closest_sum - target):
                    closest_sum = current_sum
                
                if current_sum < target:
                    left += 1
                elif current_sum > target:
                    right -= 1
                else:
                    return current_sum
        
        return closest_sum
    
    def container_with_most_water(self, height: List[int]) -> int:
        """LC 11: Container With Most Water (Amazon)
        Time: O(n), Space: O(1)
        """
        left, right = 0, len(height) - 1
        max_area = 0
        
        while left < right:
            width = right - left
            current_area = width * min(height[left], height[right])
            max_area = max(max_area, current_area)
            
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_area
    
    # ==========================================
    # MICROSOFT PROBLEMS
    # ==========================================
    
    def reverse_pairs(self, nums: List[int]) -> int:
        """LC 493: Reverse Pairs (Microsoft)
        Time: O(n log n), Space: O(n)
        """
        def merge_sort_count(nums, temp, left, right):
            if left >= right:
                return 0
            
            mid = (left + right) // 2
            count = merge_sort_count(nums, temp, left, mid)
            count += merge_sort_count(nums, temp, mid + 1, right)
            count += merge_count(nums, temp, left, mid, right)
            
            return count
        
        def merge_count(nums, temp, left, mid, right):
            # Count reverse pairs
            count = 0
            j = mid + 1
            
            for i in range(left, mid + 1):
                while j <= right and nums[i] > 2 * nums[j]:
                    j += 1
                count += j - (mid + 1)
            
            # Merge step
            i, j, k = left, mid + 1, left
            
            while i <= mid and j <= right:
                if nums[i] <= nums[j]:
                    temp[k] = nums[i]
                    i += 1
                else:
                    temp[k] = nums[j]
                    j += 1
                k += 1
            
            while i <= mid:
                temp[k] = nums[i]
                i += 1
                k += 1
            
            while j <= right:
                temp[k] = nums[j]
                j += 1
                k += 1
            
            for i in range(left, right + 1):
                nums[i] = temp[i]
            
            return count
        
        temp = [0] * len(nums)
        return merge_sort_count(nums, temp, 0, len(nums) - 1)
    
    def find_k_pairs_smallest_sums(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        """LC 373: Find K Pairs with Smallest Sums (Microsoft)
        Time: O(k log k), Space: O(k)
        """
        if not nums1 or not nums2:
            return []
        
        heap = [(nums1[0] + nums2[0], 0, 0)]
        visited = {(0, 0)}
        result = []
        
        for _ in range(min(k, len(nums1) * len(nums2))):
            if not heap:
                break
            
            sum_val, i, j = heapq.heappop(heap)
            result.append([nums1[i], nums2[j]])
            
            # Add adjacent pairs
            if i + 1 < len(nums1) and (i + 1, j) not in visited:
                heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
                visited.add((i + 1, j))
            
            if j + 1 < len(nums2) and (i, j + 1) not in visited:
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
                visited.add((i, j + 1))
        
        return result
    
    # ==========================================
    # APPLE PROBLEMS
    # ==========================================
    
    def maximum_gap(self, nums: List[int]) -> int:
        """LC 164: Maximum Gap (Apple)
        Time: O(n), Space: O(n)
        """
        if len(nums) < 2:
            return 0
        
        # Use bucket sort approach
        min_val, max_val = min(nums), max(nums)
        if min_val == max_val:
            return 0
        
        n = len(nums)
        bucket_size = max(1, (max_val - min_val) // (n - 1))
        bucket_count = (max_val - min_val) // bucket_size + 1
        
        buckets = [[float('inf'), float('-inf')] for _ in range(bucket_count)]
        
        # Place numbers in buckets
        for num in nums:
            bucket_idx = (num - min_val) // bucket_size
            buckets[bucket_idx][0] = min(buckets[bucket_idx][0], num)
            buckets[bucket_idx][1] = max(buckets[bucket_idx][1], num)
        
        # Find maximum gap
        max_gap = 0
        prev_max = min_val
        
        for bucket_min, bucket_max in buckets:
            if bucket_min != float('inf'):
                max_gap = max(max_gap, bucket_min - prev_max)
                prev_max = bucket_max
        
        return max_gap

class MedianFinder:
    """Helper class for LC 295"""
    
    def __init__(self):
        self.small = []  # max heap (negated values)
        self.large = []  # min heap
    
    def addNum(self, num: int) -> None:
        heapq.heappush(self.small, -num)
        
        # Balance heaps
        if self.small and self.large and (-self.small[0] > self.large[0]):
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        
        # Maintain size constraint
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        
        if len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)
    
    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        elif len(self.large) > len(self.small):
            return self.large[0]
        else:
            return (-self.small[0] + self.large[0]) / 2

# Test Examples
def run_examples():
    problems = CompanyInterviewProblems()
    
    print("=== COMPANY INTERVIEW PROBLEMS ===\n")
    
    print("1. GOOGLE PROBLEMS:")
    # Merge k sorted lists
    lists = [[1,4,5],[1,3,4],[2,6]]
    merged = problems.merge_k_sorted_lists(lists)
    print(f"Merge k sorted lists: {merged}")
    
    # Meeting rooms
    intervals = [[0,30],[5,10],[15,20]]
    rooms = problems.meeting_rooms_ii(intervals)
    print(f"Minimum meeting rooms needed: {rooms}")
    
    print("\n2. FACEBOOK PROBLEMS:")
    # K closest points
    points = [[1,1],[1,3],[3,4]]
    closest = problems.k_closest_points_to_origin(points, 2)
    print(f"2 closest points to origin: {closest}")
    
    # Valid palindrome
    palindrome = problems.valid_palindrome_ii("raceacar")
    print(f"'raceacar' can be palindrome after removing one char: {palindrome}")
    
    print("\n3. AMAZON PROBLEMS:")
    # Two sum
    nums = [2,7,11,15]
    two_sum = problems.two_sum(nums, 9)
    print(f"Two sum indices for target 9: {two_sum}")
    
    # Container with most water
    height = [1,8,6,2,5,4,8,3,7]
    max_area = problems.container_with_most_water(height)
    print(f"Container with most water: {max_area}")
    
    print("\n4. MICROSOFT PROBLEMS:")
    # Find k pairs with smallest sums
    nums1, nums2 = [1,7,11], [2,4,6]
    k_pairs = problems.find_k_pairs_smallest_sums(nums1, nums2, 3)
    print(f"3 pairs with smallest sums: {k_pairs}")

if __name__ == "__main__":
    run_examples() 