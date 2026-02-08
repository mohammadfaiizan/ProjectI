"""
Problem: Maximize Sum After K Negations
URL: https://practice.geeksforgeeks.org/problems/maximize-sum-after-k-negations1149/1

Problem Statement:
Given array and K, negate K elements to maximize sum.

Sample Input/Output:
Input: arr[] = {-2, 0, 5, -1, 2}, K = 4
Output: 10
Explanation: Negate -2, -1, 0, 5. Array becomes {2, 0, -5, 1, 2}. Sum = 0.
"""

import heapq


class Solution:
    def Maximize_Sum_After_K_Negations_Sort(self, arr, K):
        """
        Sort + negate negatives, handle remaining K greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr.sort()
        n = len(arr)
        
        for i in range(n):
            if K > 0 and arr[i] < 0:
                arr[i] = -arr[i]
                K -= 1
            else:
                break
        
        if K > 0 and K % 2 == 1:
            arr.sort()
            arr[0] = -arr[0]
        
        return sum(arr)
    
    def Maximize_Sum_After_K_Negations_Min_Heap(self, arr, K):
        """
        Min-heap approach: Always negate smallest element
        Time Complexity: O(n + k log n)
        Space Complexity: O(n)
        """
        pq = arr[:]
        heapq.heapify(pq)
        
        for i in range(K):
            min_val = heapq.heappop(pq)
            heapq.heappush(pq, -min_val)
        
        return sum(pq)


def Test_Maximize_Sum_After_K_Negations():
    solution = Solution()
    
    arr1 = [-2, 0, 5, -1, 2]
    print(f"Test 1 (Sort): {solution.Maximize_Sum_After_K_Negations_Sort(arr1[:], 4)}")
    
    arr2 = [-2, 0, 5, -1, 2]
    print(f"Test 1 (Heap): {solution.Maximize_Sum_After_K_Negations_Min_Heap(arr2, 4)}")
    
    arr3 = [9, 8, 8, 5]
    print(f"Test 2 (Sort): {solution.Maximize_Sum_After_K_Negations_Sort(arr3[:], 3)}")


if __name__ == "__main__":
    Test_Maximize_Sum_After_K_Negations()
