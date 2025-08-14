"""
Array Heap and Priority Queue Operations
========================================

Topics: Heap implementation, priority queue, top K problems
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium to Hard
"""

from typing import List, Tuple
import heapq
from collections import Counter

class ArrayHeapPriorityQueue:
    
    # ==========================================
    # 1. HEAP IMPLEMENTATION USING ARRAY
    # ==========================================
    
    class MaxHeap:
        """Max heap implementation using array"""
        
        def __init__(self):
            self.heap = []
        
        def parent(self, i: int) -> int:
            return (i - 1) // 2
        
        def left_child(self, i: int) -> int:
            return 2 * i + 1
        
        def right_child(self, i: int) -> int:
            return 2 * i + 2
        
        def insert(self, val: int) -> None:
            self.heap.append(val)
            self._heapify_up(len(self.heap) - 1)
        
        def extract_max(self) -> int:
            if not self.heap:
                return None
            
            if len(self.heap) == 1:
                return self.heap.pop()
            
            max_val = self.heap[0]
            self.heap[0] = self.heap.pop()
            self._heapify_down(0)
            return max_val
        
        def peek(self) -> int:
            return self.heap[0] if self.heap else None
        
        def _heapify_up(self, i: int) -> None:
            while i > 0 and self.heap[self.parent(i)] < self.heap[i]:
                self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
                i = self.parent(i)
        
        def _heapify_down(self, i: int) -> None:
            while self.left_child(i) < len(self.heap):
                max_child_idx = self.left_child(i)
                
                if (self.right_child(i) < len(self.heap) and 
                    self.heap[self.right_child(i)] > self.heap[self.left_child(i)]):
                    max_child_idx = self.right_child(i)
                
                if self.heap[i] >= self.heap[max_child_idx]:
                    break
                
                self.heap[i], self.heap[max_child_idx] = self.heap[max_child_idx], self.heap[i]
                i = max_child_idx
    
    # ==========================================
    # 2. TOP K PROBLEMS
    # ==========================================
    
    def find_kth_largest(self, nums: List[int], k: int) -> int:
        """LC 215: Kth Largest Element using min heap
        Time: O(n log k), Space: O(k)
        """
        heap = []
        
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        
        return heap[0]
    
    def top_k_frequent(self, nums: List[int], k: int) -> List[int]:
        """LC 347: Top K Frequent Elements
        Time: O(n log k), Space: O(n)
        """
        count = Counter(nums)
        return heapq.nlargest(k, count.keys(), key=count.get)
    
    def k_closest_points(self, points: List[List[int]], k: int) -> List[List[int]]:
        """LC 973: K Closest Points to Origin
        Time: O(n log k), Space: O(k)
        """
        def distance_squared(point):
            return point[0] ** 2 + point[1] ** 2
        
        heap = []
        
        for point in points:
            dist = distance_squared(point)
            if len(heap) < k:
                heapq.heappush(heap, (-dist, point))
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, point))
        
        return [point for _, point in heap]
    
    def find_k_pairs_smallest_sums(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        """LC 373: Find K Pairs with Smallest Sums
        Time: O(k log k), Space: O(k)
        """
        if not nums1 or not nums2:
            return []
        
        heap = [(nums1[0] + nums2[0], 0, 0)]
        visited = set()
        result = []
        
        while heap and len(result) < k:
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
    # 3. MERGE PROBLEMS USING HEAP
    # ==========================================
    
    def merge_k_sorted_lists(self, lists: List[List[int]]) -> List[int]:
        """LC 23: Merge k Sorted Lists (adapted for arrays)
        Time: O(n log k), Space: O(k)
        """
        heap = []
        result = []
        
        # Initialize heap with first element of each list
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst[0], i, 0))
        
        while heap:
            val, list_idx, elem_idx = heapq.heappop(heap)
            result.append(val)
            
            # Add next element from same list
            if elem_idx + 1 < len(lists[list_idx]):
                next_val = lists[list_idx][elem_idx + 1]
                heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
        
        return result
    
    def smallest_range_k_lists(self, nums: List[List[int]]) -> List[int]:
        """LC 632: Smallest Range Covering Elements from K Lists
        Time: O(n log k), Space: O(k)
        """
        heap = []
        max_val = float('-inf')
        
        # Initialize heap with first element of each list
        for i, lst in enumerate(nums):
            heapq.heappush(heap, (lst[0], i, 0))
            max_val = max(max_val, lst[0])
        
        range_start, range_end = 0, float('inf')
        
        while heap:
            min_val, list_idx, elem_idx = heapq.heappop(heap)
            
            # Update range if current is smaller
            if max_val - min_val < range_end - range_start:
                range_start, range_end = min_val, max_val
            
            # Add next element from same list
            if elem_idx + 1 < len(nums[list_idx]):
                next_val = nums[list_idx][elem_idx + 1]
                heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
                max_val = max(max_val, next_val)
            else:
                break
        
        return [range_start, range_end]
    
    # ==========================================
    # 4. SLIDING WINDOW WITH HEAP
    # ==========================================
    
    def sliding_window_median(self, nums: List[int], k: int) -> List[float]:
        """LC 480: Sliding Window Median
        Time: O(n log k), Space: O(k)
        """
        from bisect import bisect_left, insort
        
        result = []
        window = []
        
        for i, num in enumerate(nums):
            insort(window, num)
            
            if len(window) > k:
                window.pop(bisect_left(window, nums[i - k]))
            
            if len(window) == k:
                if k % 2 == 1:
                    result.append(float(window[k // 2]))
                else:
                    result.append((window[k // 2 - 1] + window[k // 2]) / 2.0)
        
        return result
    
    # ==========================================
    # 5. ADVANCED HEAP PROBLEMS
    # ==========================================
    
    def reorganize_string(self, s: str) -> str:
        """LC 767: Reorganize String
        Time: O(n log k), Space: O(k)
        """
        count = Counter(s)
        
        # Check if reorganization is possible
        if max(count.values()) > (len(s) + 1) // 2:
            return ""
        
        # Max heap of (count, char)
        heap = [(-cnt, char) for char, cnt in count.items()]
        heapq.heapify(heap)
        
        result = []
        prev_count, prev_char = 0, ''
        
        while heap:
            count, char = heapq.heappop(heap)
            result.append(char)
            
            # Put back previous character if it still has count
            if prev_count < 0:
                heapq.heappush(heap, (prev_count, prev_char))
            
            # Update for next iteration
            prev_count, prev_char = count + 1, char
        
        return ''.join(result) if len(result) == len(s) else ""
    
    def meeting_rooms_ii(self, intervals: List[List[int]]) -> int:
        """LC 253: Meeting Rooms II using heap
        Time: O(n log n), Space: O(n)
        """
        if not intervals:
            return 0
        
        intervals.sort(key=lambda x: x[0])
        heap = []  # End times
        
        for start, end in intervals:
            if heap and heap[0] <= start:
                heapq.heappop(heap)
            heapq.heappush(heap, end)
        
        return len(heap)
    
    def find_median_data_stream(self):
        """LC 295: Find Median from Data Stream
        """
        class MedianFinder:
            def __init__(self):
                self.small = []  # Max heap (negated values)
                self.large = []  # Min heap
            
            def addNum(self, num: int) -> None:
                heapq.heappush(self.small, -num)
                
                # Ensure all elements in small <= all elements in large
                if self.small and self.large and (-self.small[0] > self.large[0]):
                    val = -heapq.heappop(self.small)
                    heapq.heappush(self.large, val)
                
                # Balance the heaps
                if len(self.small) > len(self.large) + 1:
                    val = -heapq.heappop(self.small)
                    heapq.heappush(self.large, val)
                elif len(self.large) > len(self.small) + 1:
                    val = heapq.heappop(self.large)
                    heapq.heappush(self.small, -val)
            
            def findMedian(self) -> float:
                if len(self.small) > len(self.large):
                    return -self.small[0]
                elif len(self.large) > len(self.small):
                    return self.large[0]
                else:
                    return (-self.small[0] + self.large[0]) / 2.0
        
        return MedianFinder()

# Test Examples
def run_examples():
    ahpq = ArrayHeapPriorityQueue()
    
    print("=== ARRAY HEAP AND PRIORITY QUEUE EXAMPLES ===\n")
    
    # Max heap operations
    print("1. MAX HEAP OPERATIONS:")
    max_heap = ahpq.MaxHeap()
    elements = [3, 1, 6, 5, 2, 4]
    
    for elem in elements:
        max_heap.insert(elem)
        print(f"Inserted {elem}, max: {max_heap.peek()}")
    
    print("Extracting elements:")
    while max_heap.heap:
        print(f"Extracted: {max_heap.extract_max()}")
    
    # Top K problems
    print("\n2. TOP K PROBLEMS:")
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    kth_largest = ahpq.find_kth_largest(nums, k)
    print(f"Kth largest in {nums}: {kth_largest}")
    
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    top_k = ahpq.top_k_frequent(nums, k)
    print(f"Top {k} frequent: {top_k}")
    
    # Merge problems
    print("\n3. MERGE PROBLEMS:")
    lists = [[1, 4, 5], [1, 3, 4], [2, 6]]
    merged = ahpq.merge_k_sorted_lists(lists)
    print(f"Merged k sorted lists: {merged}")
    
    # Advanced problems
    print("\n4. ADVANCED PROBLEMS:")
    s = "aab"
    reorganized = ahpq.reorganize_string(s)
    print(f"Reorganize string '{s}': '{reorganized}'")
    
    intervals = [[0, 30], [5, 10], [15, 20]]
    rooms = ahpq.meeting_rooms_ii(intervals)
    print(f"Meeting rooms needed: {rooms}")

if __name__ == "__main__":
    run_examples() 