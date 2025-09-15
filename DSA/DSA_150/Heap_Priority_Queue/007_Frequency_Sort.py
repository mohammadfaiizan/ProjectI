"""
Problem: Frequency Sort
URL: https://leetcode.com/problems/sort-array-by-increasing-frequency/description/

Problem Statement:
Given an array of integers nums, sort the array in increasing order based on the frequency of the values.
If multiple values have the same frequency, sort them in decreasing order.

Sample Input/Output:
Input: nums = [1,1,2,2,2,3]
Output: [3,1,1,2,2,2]
Explanation: '3' has frequency 1, '1' has frequency 2, '2' has frequency 3

Input: nums = [2,3,1,3,2]
Output: [1,3,3,2,2]
Explanation: '1' has frequency 1, '2' and '3' both have frequency 2
"""

import heapq
from collections import Counter
from typing import List

class Solution:
    def Frequency_Sort_Brute_Force(self, nums: List[int]) -> List[int]:
        """
        Brute Force Approach - Count frequencies and sort
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        freq_map = {}
        for num in nums:
            freq_map[num] = freq_map.get(num, 0) + 1
        
        nums.sort(key=lambda x: (freq_map[x], -x))
        return nums
    
    def Frequency_Sort_Counter_Sort(self, nums: List[int]) -> List[int]:
        """
        Counter with Custom Sort
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        counter = Counter(nums)
        return sorted(nums, key=lambda x: (counter[x], -x))
    
    def Frequency_Sort_Min_Heap_Optimal(self, nums: List[int]) -> List[int]:
        """
        Min Heap Approach - Sort by frequency using heap
        Time Complexity: O(n log k) where k is unique elements
        Space Complexity: O(k)
        """
        freq_map = Counter(nums)
        min_heap = []
        
        for num, freq in freq_map.items():
            heapq.heappush(min_heap, (freq, -num, num))
        
        result = []
        while min_heap:
            freq, neg_num, num = heapq.heappop(min_heap)
            result.extend([num] * freq)
        
        return result
    
    def Frequency_Sort_Bucket_Sort(self, nums: List[int]) -> List[int]:
        """
        Bucket Sort Approach - Group by frequency
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        freq_map = Counter(nums)
        n = len(nums)
        buckets = [[] for _ in range(n + 1)]
        
        for num, freq in freq_map.items():
            buckets[freq].append(num)
        
        result = []
        for freq in range(1, n + 1):
            if buckets[freq]:
                buckets[freq].sort(reverse=True)
                for num in buckets[freq]:
                    result.extend([num] * freq)
        
        return result
    
    def Frequency_Sort_Priority_Queue(self, nums: List[int]) -> List[int]:
        """
        Priority Queue with Custom Comparator
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        freq_map = Counter(nums)
        priority_queue = []
        
        for num, freq in freq_map.items():
            heapq.heappush(priority_queue, (freq, -num))
        
        result = []
        while priority_queue:
            freq, neg_num = heapq.heappop(priority_queue)
            result.extend([-neg_num] * freq)
        
        return result
    
    def Frequency_Sort_Stable_Sort(self, nums: List[int]) -> List[int]:
        """
        Stable Sort Implementation
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        freq_map = Counter(nums)
        
        def Compare_Key(x):
            return (freq_map[x], -x)
        
        return sorted(nums, key=Compare_Key)

def Test_Frequency_Sort():
    solution = Solution()
    
    test_cases = [
        ([1,1,2,2,2,3], [3,1,1,2,2,2]),
        ([2,3,1,3,2], [1,3,3,2,2]),
        ([-1,1,-6,4,5,-6,1,4,1], [5,-1,4,4,-6,-6,1,1,1]),
        ([1,2,3,4,5], [1,2,3,4,5])
    ]
    
    for nums, expected in test_cases:
        result1 = solution.Frequency_Sort_Brute_Force(nums.copy())
        result2 = solution.Frequency_Sort_Counter_Sort(nums.copy())
        result3 = solution.Frequency_Sort_Min_Heap_Optimal(nums.copy())
        result4 = solution.Frequency_Sort_Bucket_Sort(nums.copy())
        result5 = solution.Frequency_Sort_Priority_Queue(nums.copy())
        result6 = solution.Frequency_Sort_Stable_Sort(nums.copy())
        
        print(f"Array: {nums}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Counter Sort: {result2}")
        print(f"Min Heap Optimal: {result3}")
        print(f"Bucket Sort: {result4}")
        print(f"Priority Queue: {result5}")
        print(f"Stable Sort: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Frequency_Sort()
