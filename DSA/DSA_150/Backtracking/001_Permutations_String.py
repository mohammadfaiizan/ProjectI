"""
Problem: Permutations of a String
URL: https://www.geeksforgeeks.org/problems/bfs-traversal-of-graph/1

Problem Statement:
Given a string S. The task is to print all unique permutations of the given string in lexicographically sorted order.

Sample Input/Output:
Input: S = "ABC"
Output: ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]
Explanation: All permutations of "ABC" in lexicographical order

Input: S = "ABSG"
Output: ["ABGS", "ABSG", "AGBS", "AGSB", "ASBG", "ASGB", "BAGS", "BASG", "BGAS", "BGSA", "BSAG", "BSGA", "GABS", "GASB", "GBAS", "GBSA", "GSAB", "GSBA", "SABG", "SAGB", "SBAG", "SBGA", "SGAB", "SGBA"]
"""

from typing import List
import itertools

class Solution:
    def String_Permutations_Built_In(self, s: str) -> List[str]:
        """
        Built-in Permutations - Using itertools
        Time Complexity: O(n! * n)
        Space Complexity: O(n! * n)
        """
        perms = set(itertools.permutations(s))
        return sorted([''.join(p) for p in perms])
    
    def String_Permutations_Backtracking_Optimal(self, s: str) -> List[str]:
        """
        Backtracking Approach - Generate permutations recursively
        Time Complexity: O(n! * n)
        Space Complexity: O(n! * n)
        """
        result = []
        chars = list(s)
        
        def Backtrack(current_permutation: List[str]) -> None:
            if len(current_permutation) == len(chars):
                result.append(''.join(current_permutation))
                return
            
            used = set()
            for i, char in enumerate(chars):
                if i not in used_indices and char not in used:
                    used.add(char)
                    used_indices.add(i)
                    current_permutation.append(char)
                    Backtrack(current_permutation)
                    current_permutation.pop()
                    used_indices.remove(i)
        
        used_indices = set()
        Backtrack([])
        return sorted(list(set(result)))
    
    def String_Permutations_Swap_Based(self, s: str) -> List[str]:
        """
        Swap Based Backtracking - Generate by swapping characters
        Time Complexity: O(n! * n)
        Space Complexity: O(n! * n)
        """
        result = set()
        chars = list(s)
        
        def Generate_Permutations(start: int) -> None:
            if start == len(chars):
                result.add(''.join(chars))
                return
            
            for i in range(start, len(chars)):
                chars[start], chars[i] = chars[i], chars[start]
                Generate_Permutations(start + 1)
                chars[start], chars[i] = chars[i], chars[start]
        
        Generate_Permutations(0)
        return sorted(list(result))
    
    def String_Permutations_Used_Array(self, s: str) -> List[str]:
        """
        Used Array Backtracking - Track used characters
        Time Complexity: O(n! * n)
        Space Complexity: O(n! * n)
        """
        result = set()
        chars = list(s)
        used = [False] * len(chars)
        
        def Generate(current: List[str]) -> None:
            if len(current) == len(chars):
                result.add(''.join(current))
                return
            
            for i in range(len(chars)):
                if not used[i]:
                    used[i] = True
                    current.append(chars[i])
                    Generate(current)
                    current.pop()
                    used[i] = False
        
        Generate([])
        return sorted(list(result))
    
    def String_Permutations_Lexicographic(self, s: str) -> List[str]:
        """
        Lexicographic Generation - Generate in sorted order
        Time Complexity: O(n! * n)
        Space Complexity: O(n! * n)
        """
        chars = sorted(list(s))
        result = []
        
        def Next_Permutation() -> bool:
            i = len(chars) - 2
            while i >= 0 and chars[i] >= chars[i + 1]:
                i -= 1
            
            if i == -1:
                return False
            
            j = len(chars) - 1
            while chars[j] <= chars[i]:
                j -= 1
            
            chars[i], chars[j] = chars[j], chars[i]
            chars[i + 1:] = reversed(chars[i + 1:])
            return True
        
        result.append(''.join(chars))
        while Next_Permutation():
            result.append(''.join(chars))
        
        return list(set(result))
    
    def String_Permutations_Recursive_Helper(self, s: str) -> List[str]:
        """
        Recursive Helper - Using helper function for recursion
        Time Complexity: O(n! * n)
        Space Complexity: O(n! * n)
        """
        if len(s) <= 1:
            return [s]
        
        result = set()
        for i, char in enumerate(s):
            remaining = s[:i] + s[i+1:]
            for perm in self.String_Permutations_Recursive_Helper(remaining):
                result.add(char + perm)
        
        return sorted(list(result))

def Test_String_Permutations():
    solution = Solution()
    
    test_cases = [
        "ABC",
        "AAB", 
        "AB",
        "A",
        "ABCD"
    ]
    
    for s in test_cases:
        result1 = solution.String_Permutations_Built_In(s)
        result2 = solution.String_Permutations_Backtracking_Optimal(s)
        result3 = solution.String_Permutations_Swap_Based(s)
        result4 = solution.String_Permutations_Used_Array(s)
        result5 = solution.String_Permutations_Lexicographic(s)
        result6 = solution.String_Permutations_Recursive_Helper(s)
        
        print(f"String: '{s}'")
        print(f"Built-in count: {len(result1)}")
        print(f"Backtracking count: {len(result2)}")
        print(f"Swap-based count: {len(result3)}")
        print(f"Used array count: {len(result4)}")
        print(f"Lexicographic count: {len(result5)}")
        print(f"Recursive helper count: {len(result6)}")
        
        if len(s) <= 3:
            print(f"Backtracking result: {result2}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_String_Permutations()
