"""
Problem: All Paths Top Left to Bottom Right
URL: https://www.geeksforgeeks.org/print-all-possible-paths-from-top-left-to-bottom-right-of-a-mxn-matrix/

Problem Statement:
Print all possible paths from top-left to bottom-right in MxN matrix. Can only move right or down.

Sample Input/Output:
Input: Matrix 2x2
Output: 
DR
RD
Explanation: D=Down, R=Right
"""

from collections import deque


class Solution:
    def All_Paths_Backtracking(self, m, n):
        """
        Backtracking
        Time Complexity: O(2^(m+n))
        Space Complexity: O(m+n)
        """
        result = []
        path = []
        
        def backtrack(x, y):
            if x == m - 1 and y == n - 1:
                result.append(''.join(path))
                return
            
            if x < m - 1:
                path.append('D')
                backtrack(x + 1, y)
                path.pop()
            
            if y < n - 1:
                path.append('R')
                backtrack(x, y + 1)
                path.pop()
        
        backtrack(0, 0)
        return result
    
    def All_Paths_Iterative(self, m, n):
        """
        Iterative using queue
        Time Complexity: O(2^(m+n))
        Space Complexity: O(2^(m+n))
        """
        result = []
        q = deque([((0, 0), "")])
        
        while q:
            (x, y), path = q.popleft()
            
            if x == m - 1 and y == n - 1:
                result.append(path)
                continue
            
            if x < m - 1:
                q.append(((x + 1, y), path + "D"))
            
            if y < n - 1:
                q.append(((x, y + 1), path + "R"))
        
        return result


def Test_All_Paths_Top_Left_Bottom_Right():
    solution = Solution()
    m = 2
    n = 2
    result1 = solution.All_Paths_Backtracking(m, n)
    result2 = solution.All_Paths_Iterative(m, n)
    
    print("Backtracking Approach:")
    for path in result1:
        print(path)
    
    print("Iterative Approach:")
    for path in result2:
        print(path)


if __name__ == "__main__":
    Test_All_Paths_Top_Left_Bottom_Right()
