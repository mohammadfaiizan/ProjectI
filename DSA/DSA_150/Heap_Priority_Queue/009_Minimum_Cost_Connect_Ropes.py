"""
Problem: Minimum Cost to Connect Ropes
URL: https://leetcode.com/problems/min-cost-to-connect-all-points/description/

Problem Statement:
You have some number of ropes. Each rope has some length. You need to connect all the ropes into one rope.
The cost of connecting two ropes is equal to the sum of their lengths.
Find the minimum cost to connect all the ropes.

Sample Input/Output:
Input: ropes = [8, 4, 6, 12]
Output: 58
Explanation: Connect 4 and 6 first (cost=10), then connect 8 and 12 (cost=20), then connect 10 and 20 (cost=30)
Total cost = 10 + 20 + 30 = 60. But optimal is: 4+6=10, 8+10=18, 12+18=30. Total = 10+18+30=58

Input: ropes = [20, 4, 8, 2]
Output: 54
Explanation: Connect 2 and 4 first (cost=6), then 6 and 8 (cost=14), then 14 and 20 (cost=34)
Total cost = 6 + 14 + 34 = 54
"""

import heapq
from typing import List

class Solution:
    def Connect_Ropes_Brute_Force(self, ropes: List[int]) -> int:
        """
        Brute Force Approach - Always pick two smallest ropes
        Time Complexity: O(n³)
        Space Complexity: O(1)
        """
        if len(ropes) <= 1:
            return 0
        
        total_cost = 0
        
        while len(ropes) > 1:
            ropes.sort()
            first = ropes.pop(0)
            second = ropes.pop(0)
            cost = first + second
            total_cost += cost
            ropes.append(cost)
        
        return total_cost
    
    def Connect_Ropes_Selection_Sort(self, ropes: List[int]) -> int:
        """
        Selection Sort Approach - Manually find two smallest
        Time Complexity: O(n³)
        Space Complexity: O(1)
        """
        if len(ropes) <= 1:
            return 0
        
        total_cost = 0
        
        while len(ropes) > 1:
            min1_idx = 0
            for i in range(1, len(ropes)):
                if ropes[i] < ropes[min1_idx]:
                    min1_idx = i
            
            first = ropes.pop(min1_idx)
            
            min2_idx = 0
            for i in range(1, len(ropes)):
                if ropes[i] < ropes[min2_idx]:
                    min2_idx = i
            
            second = ropes.pop(min2_idx)
            
            cost = first + second
            total_cost += cost
            ropes.append(cost)
        
        return total_cost
    
    def Connect_Ropes_Min_Heap_Optimal(self, ropes: List[int]) -> int:
        """
        Min Heap Approach - Always extract two smallest efficiently
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if len(ropes) <= 1:
            return 0
        
        heapq.heapify(ropes)
        total_cost = 0
        
        while len(ropes) > 1:
            first = heapq.heappop(ropes)
            second = heapq.heappop(ropes)
            cost = first + second
            total_cost += cost
            heapq.heappush(ropes, cost)
        
        return total_cost
    
    def Connect_Ropes_Priority_Queue(self, ropes: List[int]) -> int:
        """
        Priority Queue Approach - Using heapq operations
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if len(ropes) <= 1:
            return 0
        
        priority_queue = ropes.copy()
        heapq.heapify(priority_queue)
        total_cost = 0
        
        while len(priority_queue) > 1:
            rope1 = heapq.heappop(priority_queue)
            rope2 = heapq.heappop(priority_queue)
            
            merged_cost = rope1 + rope2
            total_cost += merged_cost
            
            heapq.heappush(priority_queue, merged_cost)
        
        return total_cost
    
    def Connect_Ropes_Greedy_Heap(self, ropes: List[int]) -> int:
        """
        Greedy Algorithm with Heap - Always combine smallest ropes
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if not ropes or len(ropes) <= 1:
            return 0
        
        min_heap = ropes[:]
        heapq.heapify(min_heap)
        
        result = 0
        
        while len(min_heap) > 1:
            cost1 = heapq.heappop(min_heap)
            cost2 = heapq.heappop(min_heap)
            
            current_cost = cost1 + cost2
            result += current_cost
            
            heapq.heappush(min_heap, current_cost)
        
        return result
    
    def Connect_Ropes_Huffman_Coding(self, ropes: List[int]) -> int:
        """
        Huffman Coding Approach - Similar to Huffman tree construction
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if len(ropes) <= 1:
            return 0
        
        heap = ropes[:]
        heapq.heapify(heap)
        total_cost = 0
        
        while len(heap) > 1:
            first_min = heapq.heappop(heap)
            second_min = heapq.heappop(heap)
            
            merge_cost = first_min + second_min
            total_cost += merge_cost
            
            heapq.heappush(heap, merge_cost)
        
        return total_cost

def Test_Connect_Ropes():
    solution = Solution()
    
    test_cases = [
        ([8, 4, 6, 12], 58),
        ([20, 4, 8, 2], 54),
        ([1, 2, 3, 4, 5], 33),
        ([5], 0),
        ([3, 7], 10)
    ]
    
    for ropes, expected in test_cases:
        result1 = solution.Connect_Ropes_Brute_Force(ropes.copy())
        result2 = solution.Connect_Ropes_Selection_Sort(ropes.copy())
        result3 = solution.Connect_Ropes_Min_Heap_Optimal(ropes.copy())
        result4 = solution.Connect_Ropes_Priority_Queue(ropes.copy())
        result5 = solution.Connect_Ropes_Greedy_Heap(ropes.copy())
        result6 = solution.Connect_Ropes_Huffman_Coding(ropes.copy())
        
        print(f"Ropes: {ropes}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Selection Sort: {result2}")
        print(f"Min Heap Optimal: {result3}")
        print(f"Priority Queue: {result4}")
        print(f"Greedy Heap: {result5}")
        print(f"Huffman Coding: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Connect_Ropes()
