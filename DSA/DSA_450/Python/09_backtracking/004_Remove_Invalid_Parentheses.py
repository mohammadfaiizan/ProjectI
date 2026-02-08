"""
Problem: Remove Invalid Parentheses
URL: https://leetcode.com/problems/remove-invalid-parentheses/

Problem Statement:
Remove minimum number of invalid parentheses to make string valid. Return all unique results.

Sample Input/Output:
Input: s="()())()"
Output: ["(())()","()()()"]
Explanation: Remove one ')' to make valid
"""

from collections import deque


class Solution:
    def Remove_Invalid_Parentheses_BFS(self, s):
        """
        BFS level-by-level
        Time Complexity: O(2^n)
        Space Complexity: O(2^n)
        """
        result = []
        visited = set()
        q = deque([s])
        visited.add(s)
        found = False
        
        while q:
            current = q.popleft()
            
            if self.Is_Valid(current):
                result.append(current)
                found = True
            
            if found:
                continue
            
            for i in range(len(current)):
                if current[i] not in '()':
                    continue
                
                next_str = current[:i] + current[i+1:]
                if next_str not in visited:
                    visited.add(next_str)
                    q.append(next_str)
        
        return result
    
    def Remove_Invalid_Parentheses_Backtracking(self, s):
        """
        Backtracking with min removal count
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        result = []
        min_removal = self.Get_Min_Removal(s)
        result_set = set()
        
        def backtrack(current, index, left_count, right_count, removed):
            if index == len(s):
                if left_count == right_count and removed == min_removal:
                    result_set.add(current)
                return
            
            if s[index] not in '()':
                backtrack(current + s[index], index + 1, left_count, right_count, removed)
                return
            
            backtrack(current, index + 1, left_count, right_count, removed + 1)
            
            if s[index] == '(':
                backtrack(current + '(', index + 1, left_count + 1, right_count, removed)
            elif right_count < left_count:
                backtrack(current + ')', index + 1, left_count, right_count + 1, removed)
        
        backtrack("", 0, 0, 0, 0)
        return list(result_set)
    
    def Is_Valid(self, s):
        count = 0
        for c in s:
            if c == '(':
                count += 1
            elif c == ')':
                count -= 1
                if count < 0:
                    return False
        return count == 0
    
    def Get_Min_Removal(self, s):
        left = 0
        right = 0
        for c in s:
            if c == '(':
                left += 1
            elif c == ')':
                if left > 0:
                    left -= 1
                else:
                    right += 1
        return left + right


def Test_Remove_Invalid_Parentheses():
    solution = Solution()
    
    s = "()())()"
    result1 = solution.Remove_Invalid_Parentheses_BFS(s)
    print("BFS Results:")
    for str_val in result1:
        print(str_val)
    
    result2 = solution.Remove_Invalid_Parentheses_Backtracking(s)
    print("Backtracking Results:")
    for str_val in result2:
        print(str_val)


if __name__ == "__main__":
    Test_Remove_Invalid_Parentheses()
