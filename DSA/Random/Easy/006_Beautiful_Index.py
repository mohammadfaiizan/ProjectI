"""
Problem: Beautiful Index (Equilibrium Index)
URL: https://www.naukri.com/code360/problems/beautiful-index

Problem Statement:
You are given an array 'A' of length 'N'. You say an index 'i' is beautiful if the sum 
of the first 'i - 1' elements of the array 'A' equals the sum of the last 'N - i' elements 
of the array 'A', where 'i' is in 1-based indexing.

Find the leftmost beautiful index. If no beautiful index exists, return -1.

Note: If you select the first index, then the sum of the prefix will be '0', and if you 
select the last index, then the sum of the suffix will be '0'.

Sample Input/Output:
Input: A = [1, 3, 1, 5]
Output: -1
Explanation: No index satisfies the condition

Input: A = [1, 2, 3]
Output: 2
Explanation: At index 2 (1-based): prefix sum = 1, suffix sum = 3. Not equal.
             At index 2 (correct): prefix = 1, suffix = 3 (wrong example)

Input: A = [2, 1, -1]
Output: 1
Explanation: At index 1: prefix = 0, suffix = 1 + (-1) = 0. Equal!

Input: A = [1, 1, 1, 1]
Output: -1
"""

from typing import List

class Solution:
    def Beautiful_Index_Brute_Force(self, A: List[int]) -> int:
        """
        Brute Force Approach - Calculate prefix and suffix for each index
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(A)
        
        for i in range(n):
            prefix_sum = sum(A[:i])
            suffix_sum = sum(A[i + 1:])
            
            if prefix_sum == suffix_sum:
                return i + 1
        
        return -1
    
    def Beautiful_Index_Prefix_Array(self, A: List[int]) -> int:
        """
        Prefix Sum Array Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(A)
        prefix = [0] * (n + 1)
        
        for i in range(n):
            prefix[i + 1] = prefix[i] + A[i]
        
        total = prefix[n]
        
        for i in range(n):
            left_sum = prefix[i]
            right_sum = total - prefix[i + 1]
            
            if left_sum == right_sum:
                return i + 1
        
        return -1
    
    def Beautiful_Index_Total_Sum(self, A: List[int]) -> int:
        """
        Total Sum Approach - Optimal solution
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        total_sum = sum(A)
        left_sum = 0
        
        for i in range(len(A)):
            right_sum = total_sum - left_sum - A[i]
            
            if left_sum == right_sum:
                return i + 1
            
            left_sum += A[i]
        
        return -1
    
    def Beautiful_Index_Running_Sum(self, A: List[int]) -> int:
        """
        Running Sum Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(A)
        total = sum(A)
        prefix_sum = 0
        
        for i in range(n):
            suffix_sum = total - prefix_sum - A[i]
            
            if prefix_sum == suffix_sum:
                return i + 1
            
            prefix_sum += A[i]
        
        return -1
    
    def Beautiful_Index_Enumerate(self, A: List[int]) -> int:
        """
        Enumerate Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        total = sum(A)
        left = 0
        
        for i, val in enumerate(A):
            right = total - left - val
            
            if left == right:
                return i + 1
            
            left += val
        
        return -1
    
    def Beautiful_Index_Two_Pass(self, A: List[int]) -> int:
        """
        Two Pass Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(A)
        total_sum = 0
        
        for num in A:
            total_sum += num
        
        left_sum = 0
        
        for i in range(n):
            if left_sum == (total_sum - left_sum - A[i]):
                return i + 1
            left_sum += A[i]
        
        return -1

def Test_Beautiful_Index():
    solution = Solution()
    
    test_cases = [
        ([1, 3, 1, 5], -1),
        ([2, 1, -1], 1),
        ([1, 2, 3], -1),
        ([1, 1, 1], 2),
        ([7, 1, 5, 2, -4, 3, 0], 4),
        ([1], 1),
        ([0, 0, 0], 1),
        ([1, 2, 3, 4, 6], 4)
    ]
    
    for A, expected in test_cases:
        result1 = solution.Beautiful_Index_Brute_Force(A.copy())
        result2 = solution.Beautiful_Index_Prefix_Array(A.copy())
        result3 = solution.Beautiful_Index_Total_Sum(A.copy())
        result4 = solution.Beautiful_Index_Running_Sum(A.copy())
        result5 = solution.Beautiful_Index_Enumerate(A.copy())
        result6 = solution.Beautiful_Index_Two_Pass(A.copy())
        
        print(f"Array: {A}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Prefix Array: {result2}")
        print(f"Total Sum: {result3}")
        print(f"Running Sum: {result4}")
        print(f"Enumerate: {result5}")
        print(f"Two Pass: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Beautiful_Index()

