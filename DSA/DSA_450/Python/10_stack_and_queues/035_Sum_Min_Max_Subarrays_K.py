"""
Problem: Sum of Minimum and Maximum Elements of All Subarrays of Size K
URL: https://www.geeksforgeeks.org/sum-minimum-maximum-elements-subarrays-size-k/

Problem Statement:
Given an array of size N and an integer K, find the sum of minimum and maximum elements of all contiguous subarrays of size K.

Sample Input/Output:
Input: arr[] = {2,5,-1,7,-3,-1,-2}, k = 3
Output: 18
Explanation: Subarrays of size 3: [2,5,-1] -> min=-1, max=5, sum=4
            [5,-1,7] -> min=-1, max=7, sum=6
            [-1,7,-3] -> min=-3, max=7, sum=4
            [7,-3,-1] -> min=-3, max=7, sum=4
            Total = 4+6+4+4 = 18
"""

from collections import deque


class Solution:
    def Sum_Min_Max_Subarrays_K_Deque(self, arr, k):
        """
        Find sum using deque-based sliding window with two deques.
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        n = len(arr)
        minDeque = deque()
        maxDeque = deque()
        total_sum = 0
        
        for i in range(k):
            while minDeque and arr[minDeque[-1]] >= arr[i]:
                minDeque.pop()
            while maxDeque and arr[maxDeque[-1]] <= arr[i]:
                maxDeque.pop()
            minDeque.append(i)
            maxDeque.append(i)
        
        total_sum += arr[minDeque[0]] + arr[maxDeque[0]]
        
        for i in range(k, n):
            while minDeque and minDeque[0] <= i - k:
                minDeque.popleft()
            while maxDeque and maxDeque[0] <= i - k:
                maxDeque.popleft()
            
            while minDeque and arr[minDeque[-1]] >= arr[i]:
                minDeque.pop()
            while maxDeque and arr[maxDeque[-1]] <= arr[i]:
                maxDeque.pop()
            
            minDeque.append(i)
            maxDeque.append(i)
            
            total_sum += arr[minDeque[0]] + arr[maxDeque[0]]
        
        return total_sum
    
    def Sum_Min_Max_Subarrays_K_Brute_Force(self, arr, k):
        """
        Find sum using brute force.
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        n = len(arr)
        total_sum = 0
        
        for i in range(n - k + 1):
            minVal = arr[i]
            maxVal = arr[i]
            
            for j in range(i, i + k):
                minVal = min(minVal, arr[j])
                maxVal = max(maxVal, arr[j])
            
            total_sum += minVal + maxVal
        
        return total_sum


def Test_Sum_Min_Max_Subarrays_K():
    solution = Solution()
    
    arr1 = [2, 5, -1, 7, -3, -1, -2]
    k1 = 3
    print(f"Test 1 - Deque: {solution.Sum_Min_Max_Subarrays_K_Deque(arr1, k1)}")
    print(f"Test 1 - Brute Force: {solution.Sum_Min_Max_Subarrays_K_Brute_Force(arr1, k1)}")
    
    arr2 = [1, 2, 3, 4, 5]
    k2 = 3
    print(f"Test 2 - Deque: {solution.Sum_Min_Max_Subarrays_K_Deque(arr2, k2)}")
    print(f"Test 2 - Brute Force: {solution.Sum_Min_Max_Subarrays_K_Brute_Force(arr2, k2)}")
    
    arr3 = [5, 4, 3, 2, 1]
    k3 = 2
    print(f"Test 3 - Deque: {solution.Sum_Min_Max_Subarrays_K_Deque(arr3, k3)}")
    print(f"Test 3 - Brute Force: {solution.Sum_Min_Max_Subarrays_K_Brute_Force(arr3, k3)}")


if __name__ == "__main__":
    Test_Sum_Min_Max_Subarrays_K()
