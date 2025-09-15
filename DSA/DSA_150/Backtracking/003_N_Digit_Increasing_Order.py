"""
Problem: N Digit Number with the Digits in Increasing Order
URL: https://www.geeksforgeeks.org/problems/n-digit-numbers-with-digits-in-increasing-order5903/1

Problem Statement:
Given an integer N, count all N digit numbers such that the digits are in non-decreasing order.

Sample Input/Output:
Input: N = 1
Output: 9
Explanation: All single digit numbers 1,2,3,4,5,6,7,8,9 are in non-decreasing order

Input: N = 2
Output: 45
Explanation: Numbers like 11,12,13,...,99 where digits are in non-decreasing order
"""

from typing import List

class Solution:
    def Count_Numbers_Brute_Force(self, n: int) -> int:
        """
        Brute Force - Generate all n-digit numbers and count valid ones
        Time Complexity: O(9^n)
        Space Complexity: O(n)
        """
        def Is_Non_Decreasing(num_str: str) -> bool:
            for i in range(1, len(num_str)):
                if num_str[i] < num_str[i-1]:
                    return False
            return True
        
        count = 0
        start = 10**(n-1) if n > 1 else 1
        end = 10**n
        
        for num in range(start, end):
            if Is_Non_Decreasing(str(num)):
                count += 1
        
        return count
    
    def Count_Numbers_Backtracking_Optimal(self, n: int) -> int:
        """
        Backtracking Approach - Generate only valid numbers
        Time Complexity: O(C(n+8, 8))
        Space Complexity: O(n)
        """
        def Backtrack(position: int, last_digit: int) -> int:
            if position == n:
                return 1
            
            count = 0
            for digit in range(last_digit, 10):
                count += Backtrack(position + 1, digit)
            
            return count
        
        return Backtrack(0, 1)
    
    def Count_Numbers_Dynamic_Programming(self, n: int) -> int:
        """
        Dynamic Programming - Build solution bottom-up
        Time Complexity: O(n * 10)
        Space Complexity: O(n * 10)
        """
        dp = [[0] * 10 for _ in range(n + 1)]
        
        for digit in range(1, 10):
            dp[1][digit] = 1
        
        for length in range(2, n + 1):
            for last_digit in range(1, 10):
                for prev_digit in range(1, last_digit + 1):
                    dp[length][last_digit] += dp[length - 1][prev_digit]
        
        total = 0
        for digit in range(1, 10):
            total += dp[n][digit]
        
        return total
    
    def Count_Numbers_Combinatorial(self, n: int) -> int:
        """
        Combinatorial Approach - Stars and bars method
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        from math import comb
        return comb(n + 8, 8)
    
    def Count_Numbers_Memoized(self, n: int) -> int:
        """
        Memoized Backtracking - Cache intermediate results
        Time Complexity: O(n * 10)
        Space Complexity: O(n * 10)
        """
        memo = {}
        
        def Count_With_Memo(position: int, last_digit: int) -> int:
            if position == n:
                return 1
            
            if (position, last_digit) in memo:
                return memo[(position, last_digit)]
            
            count = 0
            for digit in range(last_digit, 10):
                count += Count_With_Memo(position + 1, digit)
            
            memo[(position, last_digit)] = count
            return count
        
        return Count_With_Memo(0, 1)

def Test_Count_Numbers():
    solution = Solution()
    
    test_cases = [1, 2, 3, 4, 5]
    
    for n in test_cases:
        if n <= 3:  # Brute force is slow for large n
            result1 = solution.Count_Numbers_Brute_Force(n)
        else:
            result1 = "Skipped (too slow)"
            
        result2 = solution.Count_Numbers_Backtracking_Optimal(n)
        result3 = solution.Count_Numbers_Dynamic_Programming(n)
        result4 = solution.Count_Numbers_Combinatorial(n)
        result5 = solution.Count_Numbers_Memoized(n)
        
        print(f"N = {n}")
        print(f"Brute Force: {result1}")
        print(f"Backtracking Optimal: {result2}")
        print(f"Dynamic Programming: {result3}")
        print(f"Combinatorial: {result4}")
        print(f"Memoized: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Count_Numbers()
