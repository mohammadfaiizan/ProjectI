"""
Problem: Minimum Cost of Ropes
URL: https://practice.geeksforgeeks.org/problems/minimum-cost-of-ropes-1587115620/1

Problem Statement:
Connect N ropes with minimum cost. Cost of connecting two ropes = sum of their lengths.

Sample Input/Output:
Input: [4,3,2,6]
Output: 29
Explanation: Min-heap approach minimizes total connection cost.
"""

import heapq


class Solution:
    def Min_Cost_To_Connect_Ropes(self, ropes):
        """
        Min-heap approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        min_heap = ropes[:]
        heapq.heapify(min_heap)
        
        total_cost = 0
        
        while len(min_heap) > 1:
            first = heapq.heappop(min_heap)
            second = heapq.heappop(min_heap)
            
            cost = first + second
            total_cost += cost
            heapq.heappush(min_heap, cost)
        
        return total_cost


def Test_Minimum_Cost_Ropes():
    solution = Solution()
    
    ropes1 = [4, 3, 2, 6]
    print(f"Test 1: {solution.Min_Cost_To_Connect_Ropes(ropes1)}")
    
    ropes2 = [4, 2, 7, 6, 9]
    print(f"Test 2: {solution.Min_Cost_To_Connect_Ropes(ropes2)}")
    
    ropes3 = [1, 2, 3]
    print(f"Test 3: {solution.Min_Cost_To_Connect_Ropes(ropes3)}")


if __name__ == "__main__":
    Test_Minimum_Cost_Ropes()
