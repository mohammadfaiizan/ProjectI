"""
Problem: Top K Frequent Numbers
URL: https://leetcode.com/problems/top-k-frequent-elements/description/

Problem Statement:
Given an integer array nums and an integer k, return the k most frequent elements.
You may return the answer in any order.

Sample Input/Output:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Explanation: 1 appears 3 times, 2 appears 2 times, 3 appears 1 time

Input: nums = [1], k = 1
Output: [1]
Explanation: Only one element with frequency 1
"""

import heapq
from collections import Counter, defaultdict
from typing import List

class Solution:
    def Top_K_Frequent_Brute_Force(self, nums: List[int], k: int) -> List[int]:
        """
        Brute Force Approach - Count and sort by frequency
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        freq_map = {}
        for num in nums:
            freq_map[num] = freq_map.get(num, 0) + 1
        
        sorted_items = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:k]]
    
    def Top_K_Frequent_Counter_Sort(self, nums: List[int], k: int) -> List[int]:
        """
        Counter with Sorting Approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        counter = Counter(nums)
        return [item[0] for item in counter.most_common(k)]
    
    def Top_K_Frequent_Min_Heap_Optimal(self, nums: List[int], k: int) -> List[int]:
        """
        Min Heap Approach - Maintain heap of size k
        Time Complexity: O(n log k)
        Space Complexity: O(n + k)
        """
        freq_map = Counter(nums)
        min_heap = []
        
        for num, freq in freq_map.items():
            if len(min_heap) < k:
                heapq.heappush(min_heap, (freq, num))
            elif freq > min_heap[0][0]:
                heapq.heappop(min_heap)
                heapq.heappush(min_heap, (freq, num))
        
        return [item[1] for item in min_heap]
    
    def Top_K_Frequent_Max_Heap(self, nums: List[int], k: int) -> List[int]:
        """
        Max Heap Approach - Extract k most frequent
        Time Complexity: O(n + m log m) where m is unique elements
        Space Complexity: O(m)
        """
        freq_map = Counter(nums)
        max_heap = [(-freq, num) for num, freq in freq_map.items()]
        heapq.heapify(max_heap)
        
        result = []
        for _ in range(k):
            freq, num = heapq.heappop(max_heap)
            result.append(num)
        
        return result
    
    def Top_K_Frequent_Bucket_Sort(self, nums: List[int], k: int) -> List[int]:
        """
        Bucket Sort Approach - Use frequency as bucket index
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        freq_map = Counter(nums)
        n = len(nums)
        buckets = [[] for _ in range(n + 1)]
        
        for num, freq in freq_map.items():
            buckets[freq].append(num)
        
        result = []
        for i in range(n, 0, -1):
            if buckets[i]:
                result.extend(buckets[i])
                if len(result) >= k:
                    break
        
        return result[:k]
    
    def Top_K_Frequent_Heap_Select(self, nums: List[int], k: int) -> List[int]:
        """
        Heap Select using nlargest
        Time Complexity: O(n + m log k) where m is unique elements
        Space Complexity: O(m)
        """
        freq_map = Counter(nums)
        return heapq.nlargest(k, freq_map.keys(), key=freq_map.get)

def Test_Top_K_Frequent():
    solution = Solution()
    
    test_cases = [
        ([1,1,1,2,2,3], 2),
        ([1], 1),
        ([1,2,3,4,5,6,7,8,9,10], 3),
        ([4,1,-1,2,-1,2,3], 2)
    ]
    
    for nums, k in test_cases:
        result1 = solution.Top_K_Frequent_Brute_Force(nums.copy(), k)
        result2 = solution.Top_K_Frequent_Counter_Sort(nums.copy(), k)
        result3 = solution.Top_K_Frequent_Min_Heap_Optimal(nums.copy(), k)
        result4 = solution.Top_K_Frequent_Max_Heap(nums.copy(), k)
        result5 = solution.Top_K_Frequent_Bucket_Sort(nums.copy(), k)
        result6 = solution.Top_K_Frequent_Heap_Select(nums.copy(), k)
        
        print(f"Array: {nums}, k: {k}")
        print(f"Brute Force: {result1}")
        print(f"Counter Sort: {result2}")
        print(f"Min Heap Optimal: {result3}")
        print(f"Max Heap: {result4}")
        print(f"Bucket Sort: {result5}")
        print(f"Heap Select: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Top_K_Frequent()
