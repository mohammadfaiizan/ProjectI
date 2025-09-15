"""
Problem: Celebrity Problem
URL: https://www.geeksforgeeks.org/problems/the-celebrity-problem/1

Problem Statement:
A celebrity is a person who is known by everyone but does not know anyone at a party.
Given a square matrix M[][] where M[i][j] = 1 means person i knows person j, find the celebrity.
If there is no celebrity, return -1.

Sample Input/Output:
Input: M = [[0, 1, 1, 0],
            [0, 0, 1, 0], 
            [0, 0, 0, 0],
            [1, 1, 1, 0]]
Output: 2
Explanation: Person 2 is known by everyone but knows no one.

Input: M = [[0, 1],
            [1, 0]]
Output: -1
Explanation: No celebrity exists.
"""

from typing import List

class Solution:
    def Find_Celebrity_Brute_Force(self, M: List[List[int]]) -> int:
        """
        Brute Force Approach - Check each person if they are celebrity
        Time Complexity: O(n²)
        Space Complexity: O(1)
        """
        n = len(M)
        
        for i in range(n):
            is_celebrity = True
            
            for j in range(n):
                if i != j and M[i][j] == 1:
                    is_celebrity = False
                    break
            
            if is_celebrity:
                for j in range(n):
                    if i != j and M[j][i] == 0:
                        is_celebrity = False
                        break
                
                if is_celebrity:
                    return i
        
        return -1
    
    def Find_Celebrity_Two_Pass(self, M: List[List[int]]) -> int:
        """
        Two Pass Approach - Count knows and known_by for each person
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        n = len(M)
        knows = [0] * n
        known_by = [0] * n
        
        for i in range(n):
            for j in range(n):
                if M[i][j] == 1:
                    knows[i] += 1
                    known_by[j] += 1
        
        for i in range(n):
            if knows[i] == 0 and known_by[i] == n - 1:
                return i
        
        return -1
    
    def Find_Celebrity_Stack_Optimal(self, M: List[List[int]]) -> int:
        """
        Stack Approach - Eliminate candidates using stack
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(M)
        stack = list(range(n))
        
        while len(stack) > 1:
            a = stack.pop()
            b = stack.pop()
            
            if M[a][b] == 1:
                stack.append(b)
            else:
                stack.append(a)
        
        candidate = stack[0]
        
        for i in range(n):
            if i != candidate:
                if M[candidate][i] == 1 or M[i][candidate] == 0:
                    return -1
        
        return candidate
    
    def Find_Celebrity_Two_Pointers(self, M: List[List[int]]) -> int:
        """
        Two Pointers Approach - Eliminate candidates with two pointers
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(M)
        left, right = 0, n - 1
        
        while left < right:
            if M[left][right] == 1:
                left += 1
            else:
                right -= 1
        
        candidate = left
        
        for i in range(n):
            if i != candidate:
                if M[candidate][i] == 1 or M[i][candidate] == 0:
                    return -1
        
        return candidate
    
    def Find_Celebrity_Elimination(self, M: List[List[int]]) -> int:
        """
        Elimination Approach - Sequential elimination
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(M)
        candidate = 0
        
        for i in range(1, n):
            if M[candidate][i] == 1:
                candidate = i
        
        for i in range(n):
            if i != candidate:
                if M[candidate][i] == 1 or M[i][candidate] == 0:
                    return -1
        
        return candidate
    
    def Find_Celebrity_Graph_Theory(self, M: List[List[int]]) -> int:
        """
        Graph Theory Approach - Find node with in-degree n-1 and out-degree 0
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        n = len(M)
        in_degree = [0] * n
        out_degree = [0] * n
        
        for i in range(n):
            for j in range(n):
                if M[i][j] == 1:
                    out_degree[i] += 1
                    in_degree[j] += 1
        
        for i in range(n):
            if in_degree[i] == n - 1 and out_degree[i] == 0:
                return i
        
        return -1

def Test_Celebrity_Problem():
    solution = Solution()
    
    test_cases = [
        ([[0, 1, 1, 0],
          [0, 0, 1, 0], 
          [0, 0, 0, 0],
          [1, 1, 1, 0]], 2),
        ([[0, 1],
          [1, 0]], -1),
        ([[0, 0, 1, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 0],
          [0, 0, 1, 0]], 2),
        ([[0, 1, 1],
          [0, 0, 0],
          [0, 1, 0]], -1)
    ]
    
    for M, expected in test_cases:
        M_copy1 = [row.copy() for row in M]
        M_copy2 = [row.copy() for row in M]
        M_copy3 = [row.copy() for row in M]
        M_copy4 = [row.copy() for row in M]
        M_copy5 = [row.copy() for row in M]
        M_copy6 = [row.copy() for row in M]
        
        result1 = solution.Find_Celebrity_Brute_Force(M_copy1)
        result2 = solution.Find_Celebrity_Two_Pass(M_copy2)
        result3 = solution.Find_Celebrity_Stack_Optimal(M_copy3)
        result4 = solution.Find_Celebrity_Two_Pointers(M_copy4)
        result5 = solution.Find_Celebrity_Elimination(M_copy5)
        result6 = solution.Find_Celebrity_Graph_Theory(M_copy6)
        
        print(f"Matrix: {M}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Two Pass: {result2}")
        print(f"Stack Optimal: {result3}")
        print(f"Two Pointers: {result4}")
        print(f"Elimination: {result5}")
        print(f"Graph Theory: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Celebrity_Problem()
