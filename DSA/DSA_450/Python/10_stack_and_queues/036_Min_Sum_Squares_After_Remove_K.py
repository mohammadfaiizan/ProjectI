"""
Problem: Minimum Sum of Squares of Character Counts After Removing K Characters
URL: https://practice.geeksforgeeks.org/problems/game-with-string4100/1

Problem Statement:
Given a string s and an integer k, remove k characters from the string such that the sum of squares of the count of each distinct character remaining in the string is minimized.

Sample Input/Output:
Input: s = "aabcbcbcac", k = 3
Output: 27
Explanation: Remove 3 'c' characters. Remaining: a=3, b=3, c=1. Sum = 3^2 + 3^2 + 1^2 = 19
Actually, optimal: Remove 3 characters to minimize sum of squares.
"""

import heapq


class Solution:
    def Min_Sum_Squares_After_Remove_K_Max_Heap(self, s, k):
        """
        Minimize sum using max-heap greedy approach.
        Time Complexity: O(n + k log n)
        Space Complexity: O(n)
        """
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        
        pq = []
        for count in freq.values():
            heapq.heappush(pq, -count)
        
        while k > 0 and pq:
            top = -heapq.heappop(pq)
            top -= 1
            if top > 0:
                heapq.heappush(pq, -top)
            k -= 1
        
        total_sum = 0
        while pq:
            val = -heapq.heappop(pq)
            total_sum += val * val
        
        return total_sum


def Test_Min_Sum_Squares_After_Remove_K():
    solution = Solution()
    
    s1 = "aabcbcbcac"
    k1 = 3
    print(f"Test 1 - Max Heap: {solution.Min_Sum_Squares_After_Remove_K_Max_Heap(s1, k1)}")
    
    s2 = "abccc"
    k2 = 1
    print(f"Test 2 - Max Heap: {solution.Min_Sum_Squares_After_Remove_K_Max_Heap(s2, k2)}")
    
    s3 = "aaab"
    k3 = 2
    print(f"Test 3 - Max Heap: {solution.Min_Sum_Squares_After_Remove_K_Max_Heap(s3, k3)}")
    
    s4 = "abbccc"
    k4 = 3
    print(f"Test 4 - Max Heap: {solution.Min_Sum_Squares_After_Remove_K_Max_Heap(s4, k4)}")


if __name__ == "__main__":
    Test_Min_Sum_Squares_After_Remove_K()
