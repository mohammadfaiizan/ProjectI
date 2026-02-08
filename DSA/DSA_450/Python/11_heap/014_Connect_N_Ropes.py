"""
Problem: Connect N Ropes with Minimum Cost
URL: https://practice.geeksforgeeks.org/problems/minimum-cost-of-ropes-1587115620/1

Problem Statement:
Given N ropes of different lengths, connect them into one rope with minimum total cost. Cost to connect two ropes = sum of their lengths.

Sample Input/Output:
Input: [4,3,2,6]
Output: 29
"""

import heapq


class Solution:
    def Connect_Ropes_Min_Heap(self, ropes):
        """
        Min Heap Approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        pq = []
        for rope in ropes:
            heapq.heappush(pq, rope)

        total_cost = 0
        while len(pq) > 1:
            first = heapq.heappop(pq)
            second = heapq.heappop(pq)
            cost = first + second
            total_cost += cost
            heapq.heappush(pq, cost)

        return total_cost

    def Connect_Ropes_Sort(self, ropes):
        """
        Sort Approach
        Time Complexity: O(n^2 log n)
        Space Complexity: O(1)
        """
        arr = ropes.copy()
        total_cost = 0

        while len(arr) > 1:
            arr.sort()
            first = arr.pop(0)
            second = arr.pop(0)
            cost = first + second
            total_cost += cost
            arr.append(cost)

        return total_cost


def Test_Connect_Ropes():
    solution = Solution()

    ropes1 = [4, 3, 2, 6]
    print("Test 1 Min Heap:", solution.Connect_Ropes_Min_Heap(ropes1))
    ropes1b = [4, 3, 2, 6]
    print("Test 1 Sort:", solution.Connect_Ropes_Sort(ropes1b))

    ropes2 = [1, 2, 3, 4, 5]
    print("Test 2 Min Heap:", solution.Connect_Ropes_Min_Heap(ropes2))
    ropes2b = [1, 2, 3, 4, 5]
    print("Test 2 Sort:", solution.Connect_Ropes_Sort(ropes2b))

    ropes3 = [5, 4, 2, 8]
    print("Test 3 Min Heap:", solution.Connect_Ropes_Min_Heap(ropes3))
    ropes3b = [5, 4, 2, 8]
    print("Test 3 Sort:", solution.Connect_Ropes_Sort(ropes3b))


if __name__ == "__main__":
    Test_Connect_Ropes()
