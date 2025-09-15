"""
Problem: Combination Sum
URL: https://leetcode.com/problems/combination-sum/

Problem Statement:
Given an array of distinct integers candidates and a target integer target, 
return a list of all unique combinations of candidates where the chosen numbers sum to target. 
You may choose the same number from candidates an unlimited number of times.

Sample Input/Output:
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation: 2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7. These are the only two combinations.

Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
Explanation: Various combinations that sum to 8
"""

from typing import List

class Solution:
    def Combination_Sum_Brute_Force(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        Brute Force - Generate all possible combinations
        Time Complexity: O(N^(target/min))
        Space Complexity: O(target/min)
        """
        result = []
        
        def Generate_All_Combinations(current: List[int], remaining: int) -> None:
            if remaining == 0:
                result.append(sorted(current[:]))
                return
            
            if remaining < 0:
                return
            
            for candidate in candidates:
                current.append(candidate)
                Generate_All_Combinations(current, remaining - candidate)
                current.pop()
        
        Generate_All_Combinations([], target)
        
        unique_results = []
        seen = set()
        for combo in result:
            combo_tuple = tuple(combo)
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                unique_results.append(combo)
        
        return unique_results
    
    def Combination_Sum_Backtracking_Optimal(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        Backtracking Optimal - Avoid duplicates by maintaining order
        Time Complexity: O(N^(target/min))
        Space Complexity: O(target/min)
        """
        result = []
        
        def Backtrack(start: int, current: List[int], remaining: int) -> None:
            if remaining == 0:
                result.append(current[:])
                return
            
            for i in range(start, len(candidates)):
                if candidates[i] <= remaining:
                    current.append(candidates[i])
                    Backtrack(i, current, remaining - candidates[i])
                    current.pop()
        
        Backtrack(0, [], target)
        return result
    
    def Combination_Sum_DFS_Recursive(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        DFS Recursive - Depth-first search approach
        Time Complexity: O(N^(target/min))
        Space Complexity: O(target/min)
        """
        result = []
        candidates.sort()
        
        def DFS(index: int, path: List[int], remaining: int) -> None:
            if remaining == 0:
                result.append(path)
                return
            
            for i in range(index, len(candidates)):
                if candidates[i] > remaining:
                    break
                DFS(i, path + [candidates[i]], remaining - candidates[i])
        
        DFS(0, [], target)
        return result
    
    def Combination_Sum_Memoized(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        Memoized Backtracking - Cache intermediate results
        Time Complexity: O(N^(target/min))
        Space Complexity: O(N * target)
        """
        memo = {}
        
        def Get_Combinations(start_index: int, remaining: int) -> List[List[int]]:
            if remaining == 0:
                return [[]]
            
            if (start_index, remaining) in memo:
                return memo[(start_index, remaining)]
            
            combinations = []
            for i in range(start_index, len(candidates)):
                if candidates[i] <= remaining:
                    sub_combinations = Get_Combinations(i, remaining - candidates[i])
                    for combo in sub_combinations:
                        combinations.append([candidates[i]] + combo)
            
            memo[(start_index, remaining)] = combinations
            return combinations
        
        return Get_Combinations(0, target)
    
    def Combination_Sum_Dynamic_Programming(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        Dynamic Programming - Build combinations bottom-up
        Time Complexity: O(N * target * result_size)
        Space Complexity: O(target * result_size)
        """
        dp = [[] for _ in range(target + 1)]
        dp[0] = [[]]
        
        for candidate in candidates:
            for amount in range(candidate, target + 1):
                for combination in dp[amount - candidate]:
                    dp[amount].append(combination + [candidate])
        
        return dp[target]

def Test_Combination_Sum():
    solution = Solution()
    
    test_cases = [
        ([2,3,6,7], 7, [[2,2,3],[7]]),
        ([2,3,5], 8, [[2,2,2,2],[2,3,3],[3,5]]),
        ([2], 1, []),
        ([1], 1, [[1]]),
        ([1], 2, [[1,1]])
    ]
    
    for candidates, target, expected in test_cases:
        result1 = solution.Combination_Sum_Brute_Force(candidates.copy(), target)
        result2 = solution.Combination_Sum_Backtracking_Optimal(candidates.copy(), target)
        result3 = solution.Combination_Sum_DFS_Recursive(candidates.copy(), target)
        result4 = solution.Combination_Sum_Memoized(candidates.copy(), target)
        result5 = solution.Combination_Sum_Dynamic_Programming(candidates.copy(), target)
        
        print(f"Candidates: {candidates}, Target: {target}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Backtracking Optimal: {result2}")
        print(f"DFS Recursive: {result3}")
        print(f"Memoized: {result4}")
        print(f"Dynamic Programming: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Combination_Sum()
