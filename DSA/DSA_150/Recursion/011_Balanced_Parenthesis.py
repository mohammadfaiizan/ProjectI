"""
Problem: Balanced Parenthesis
URL: https://leetcode.com/problems/generate-parentheses/description/

Problem Statement:
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

Sample Input/Output:
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
Explanation: All valid combinations of 3 pairs of parentheses

Input: n = 1
Output: ["()"]
Explanation: Only one valid combination for 1 pair
"""

from typing import List

class Solution:
    def Generate_Parenthesis_Brute_Force(self, n: int) -> List[str]:
        """
        Brute Force - Generate all combinations and filter valid ones
        Time Complexity: O(4^n * n)
        Space Complexity: O(4^n * n)
        """
        def Is_Valid(combination: str) -> bool:
            count = 0
            for char in combination:
                if char == '(':
                    count += 1
                elif char == ')':
                    count -= 1
                    if count < 0:
                        return False
            return count == 0
        
        def Generate_All(current: str, length: int) -> None:
            if length == 2 * n:
                if Is_Valid(current):
                    result.append(current)
                return
            
            Generate_All(current + '(', length + 1)
            Generate_All(current + ')', length + 1)
        
        result = []
        Generate_All("", 0)
        return result
    
    def Generate_Parenthesis_Backtracking_Recursive(self, n: int) -> List[str]:
        """
        Backtracking Recursive - Optimal solution with pruning
        Time Complexity: O(4^n / √n) - Catalan number
        Space Complexity: O(4^n / √n)
        """
        result = []
        
        def Backtrack(current: str, open_count: int, close_count: int) -> None:
            if len(current) == 2 * n:
                result.append(current)
                return
            
            if open_count < n:
                Backtrack(current + '(', open_count + 1, close_count)
            
            if close_count < open_count:
                Backtrack(current + ')', open_count, close_count + 1)
        
        Backtrack("", 0, 0)
        return result
    
    def Generate_Parenthesis_Dynamic_Programming(self, n: int) -> List[str]:
        """
        Dynamic Programming - Build solutions from smaller problems
        Time Complexity: O(4^n / √n)
        Space Complexity: O(4^n / √n)
        """
        if n == 0:
            return [""]
        
        dp = [[] for _ in range(n + 1)]
        dp[0] = [""]
        
        for i in range(1, n + 1):
            for j in range(i):
                for left in dp[j]:
                    for right in dp[i - 1 - j]:
                        dp[i].append(f"({left}){right}")
        
        return dp[n]
    
    def Generate_Parenthesis_Stack_Based_Recursive(self, n: int) -> List[str]:
        """
        Stack Based Recursive - Using stack to track state
        Time Complexity: O(4^n / √n)
        Space Complexity: O(4^n / √n)
        """
        result = []
        
        def Generate(current: List[str], open_needed: int, close_needed: int) -> None:
            if open_needed == 0 and close_needed == 0:
                result.append(''.join(current))
                return
            
            if open_needed > 0:
                current.append('(')
                Generate(current, open_needed - 1, close_needed)
                current.pop()
            
            if close_needed > open_needed:
                current.append(')')
                Generate(current, open_needed, close_needed - 1)
                current.pop()
        
        Generate([], n, n)
        return result
    
    def Generate_Parenthesis_Closure_Number(self, n: int) -> List[str]:
        """
        Closure Number - Mathematical approach using closure numbers
        Time Complexity: O(4^n / √n)
        Space Complexity: O(4^n / √n)
        """
        def Generate_Recursive(num: int) -> List[str]:
            if num == 0:
                return ['']
            
            combinations = []
            for i in range(num):
                for left in Generate_Recursive(i):
                    for right in Generate_Recursive(num - 1 - i):
                        combinations.append(f'({left}){right}')
            
            return combinations
        
        return Generate_Recursive(n)
    
    def Generate_Parenthesis_Memoized_Recursive(self, n: int) -> List[str]:
        """
        Memoized Recursive - Cache results for efficiency
        Time Complexity: O(4^n / √n)
        Space Complexity: O(4^n / √n)
        """
        memo = {}
        
        def Generate(num: int) -> List[str]:
            if num in memo:
                return memo[num]
            
            if num == 0:
                return ['']
            
            result = []
            for i in range(num):
                for left in Generate(i):
                    for right in Generate(num - 1 - i):
                        result.append(f'({left}){right}')
            
            memo[num] = result
            return result
        
        return Generate(n)

def Test_Generate_Parenthesis():
    solution = Solution()
    
    test_cases = [1, 2, 3, 4]
    
    for n in test_cases:
        result1 = solution.Generate_Parenthesis_Brute_Force(n)
        result2 = solution.Generate_Parenthesis_Backtracking_Recursive(n)
        result3 = solution.Generate_Parenthesis_Dynamic_Programming(n)
        result4 = solution.Generate_Parenthesis_Stack_Based_Recursive(n)
        result5 = solution.Generate_Parenthesis_Closure_Number(n)
        result6 = solution.Generate_Parenthesis_Memoized_Recursive(n)
        
        print(f"n = {n}")
        print(f"Brute Force count: {len(result1)}")
        print(f"Backtracking count: {len(result2)}")
        print(f"Dynamic Programming count: {len(result3)}")
        print(f"Stack Based count: {len(result4)}")
        print(f"Closure Number count: {len(result5)}")
        print(f"Memoized count: {len(result6)}")
        
        if n <= 3:
            print(f"Backtracking result: {result2}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Generate_Parenthesis()
