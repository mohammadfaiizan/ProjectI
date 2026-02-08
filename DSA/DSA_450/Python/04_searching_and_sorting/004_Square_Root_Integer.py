"""
Problem: Count Squares / Square Root of Integer
URL: https://practice.geeksforgeeks.org/problems/count-squares3649/1

Problem Statement:
Consider a sample space S consisting of all perfect squares starting from 1, 4, 9 and so on. You are given a number N, you have to output the number of integers less than N in the sample space S. This is equivalent to finding floor(sqrt(N-1)).

Sample Input/Output:
Input: N = 9
Output: 2

Input: N = 3
Output: 1
"""

import math


class Solution:
    def Count_Squares_Math_Sqrt(self, N):
        """
        Using built-in sqrt function
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return int(math.sqrt(N - 1))

    def Count_Squares_Linear(self, N):
        """
        Linear iteration checking perfect squares
        Time Complexity: O(sqrt(n))
        Space Complexity: O(1)
        """
        count = 0
        i = 1
        while i * i < N:
            count += 1
            i += 1
        return count

    def Count_Squares_Binary_Search(self, N):
        """
        Binary search to find floor of sqrt(N-1)
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if N <= 1:
            return 0
        
        left = 1
        right = N - 1
        result = 0
        
        while left <= right:
            mid = left + (right - left) // 2
            if mid * mid < N:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result


def Test_Square_Root_Integer():
    sol = Solution()
    tests = [9, 3, 1, 16, 25, 100]

    for N in tests:
        print(f"N = {N}")
        
        res1 = sol.Count_Squares_Math_Sqrt(N)
        print(f"Math sqrt: {res1}")
        
        res2 = sol.Count_Squares_Linear(N)
        print(f"Linear: {res2}")
        
        res3 = sol.Count_Squares_Binary_Search(N)
        print(f"Binary Search: {res3}")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Square_Root_Integer()
