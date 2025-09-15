"""
Problem: Print 1 to n/ n to 1
URL: https://www.geeksforgeeks.org/problems/print-1-to-n-without-using-loops-1587115620/1&selectedLang=python3

Problem Statement:
Print numbers from 1 to N and from N to 1 without using loops.

Sample Input/Output:
Input: n = 5
Output: 1 2 3 4 5 5 4 3 2 1
Explanation: Print 1 to 5, then 5 to 1

Input: n = 3
Output: 1 2 3 3 2 1
Explanation: Print 1 to 3, then 3 to 1
"""

from typing import List

class Solution:
    def Print_1_To_N_Iterative(self, n: int) -> List[int]:
        """
        Iterative Approach - Using loops
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        result = []
        for i in range(1, n + 1):
            result.append(i)
        return result
    
    def Print_N_To_1_Iterative(self, n: int) -> List[int]:
        """
        Iterative Approach - Using loops for n to 1
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        result = []
        for i in range(n, 0, -1):
            result.append(i)
        return result
    
    def Print_1_To_N_Recursive_Simple(self, n: int, result: List[int] = None) -> List[int]:
        """
        Simple Recursive Approach - Print 1 to n
        Time Complexity: O(n)
        Space Complexity: O(n) - recursion stack
        """
        if result is None:
            result = []
        
        if n <= 0:
            return result
        
        self.Print_1_To_N_Recursive_Simple(n - 1, result)
        result.append(n)
        return result
    
    def Print_N_To_1_Recursive_Simple(self, n: int, result: List[int] = None) -> List[int]:
        """
        Simple Recursive Approach - Print n to 1
        Time Complexity: O(n)
        Space Complexity: O(n) - recursion stack
        """
        if result is None:
            result = []
        
        if n <= 0:
            return result
        
        result.append(n)
        self.Print_N_To_1_Recursive_Simple(n - 1, result)
        return result
    
    def Print_1_To_N_Recursive_Optimal(self, n: int, current: int = 1, result: List[int] = None) -> List[int]:
        """
        Tail Recursive Approach - Print 1 to n
        Time Complexity: O(n)
        Space Complexity: O(n) - recursion stack
        """
        if result is None:
            result = []
        
        if current > n:
            return result
        
        result.append(current)
        return self.Print_1_To_N_Recursive_Optimal(n, current + 1, result)
    
    def Print_Both_Recursive_Combined(self, n: int, current: int = 1, result: List[int] = None) -> List[int]:
        """
        Combined Recursive Approach - Print 1 to n, then n to 1
        Time Complexity: O(n)
        Space Complexity: O(n) - recursion stack
        """
        if result is None:
            result = []
        
        if current > n:
            return result
        
        result.append(current)
        self.Print_Both_Recursive_Combined(n, current + 1, result)
        result.append(current)
        return result

def Test_Print_Numbers():
    solution = Solution()
    
    test_cases = [1, 3, 5, 7]
    
    for n in test_cases:
        result1 = solution.Print_1_To_N_Iterative(n)
        result2 = solution.Print_N_To_1_Iterative(n)
        result3 = solution.Print_1_To_N_Recursive_Simple(n)
        result4 = solution.Print_N_To_1_Recursive_Simple(n)
        result5 = solution.Print_1_To_N_Recursive_Optimal(n)
        result6 = solution.Print_Both_Recursive_Combined(n)
        
        print(f"n = {n}")
        print(f"1 to n Iterative: {result1}")
        print(f"n to 1 Iterative: {result2}")
        print(f"1 to n Recursive Simple: {result3}")
        print(f"n to 1 Recursive Simple: {result4}")
        print(f"1 to n Recursive Optimal: {result5}")
        print(f"Both Combined: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Print_Numbers()
