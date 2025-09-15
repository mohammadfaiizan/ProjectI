"""
Problem: Dungeon Game
URL: https://leetcode.com/problems/dungeon-game/

Problem Statement:
The demons had captured the princess and imprisoned her in the bottom-right corner of a dungeon. 
The dungeon consists of m x n rooms laid out in a 2D grid. Our valiant knight starts in the top-left corner and must reach the princess.
The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately.
Some of the rooms are guarded by demons (represented by negative integers), so the knight loses health upon entering these rooms; 
other rooms are empty (represented as 0) or contain magic orbs that increase the knight's health (represented by positive integers).
To reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.
Return the knight's minimum initial health so that he can rescue the princess.

Sample Input/Output:
Input: dungeon = [[-3,5]]
Output: 4
Explanation: The optimal path is right -> right -> down -> down.

Input: dungeon = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
Output: 7
Explanation: The initial health of 7 at (0,0) allows the knight to reach the princess.
"""

from typing import List

class Solution:
    def Calculate_Minimum_HP_Brute_Force(self, dungeon: List[List[int]]) -> int:
        """
        Brute Force - Try all possible paths and track minimum health
        Time Complexity: O(2^(m+n))
        Space Complexity: O(m+n)
        """
        m, n = len(dungeon), len(dungeon[0])
        
        def Find_Min_Health(row: int, col: int, current_health: int, min_health_needed: int) -> int:
            if row >= m or col >= n:
                return float('inf')
            
            current_health += dungeon[row][col]
            min_health_needed = max(min_health_needed, 1 - current_health)
            
            if row == m - 1 and col == n - 1:
                return min_health_needed
            
            right_health = Find_Min_Health(row, col + 1, current_health, min_health_needed)
            down_health = Find_Min_Health(row + 1, col, current_health, min_health_needed)
            
            return min(right_health, down_health)
        
        return Find_Min_Health(0, 0, 0, 1)
    
    def Calculate_Minimum_HP_Recursive(self, dungeon: List[List[int]]) -> int:
        """
        Recursive - Calculate minimum health needed from each cell
        Time Complexity: O(2^(m+n))
        Space Complexity: O(m+n)
        """
        m, n = len(dungeon), len(dungeon[0])
        
        def Min_Health_From(row: int, col: int) -> int:
            if row >= m or col >= n:
                return float('inf')
            
            if row == m - 1 and col == n - 1:
                return max(1, 1 - dungeon[row][col])
            
            right_health = Min_Health_From(row, col + 1)
            down_health = Min_Health_From(row + 1, col)
            
            min_health_needed = min(right_health, down_health)
            
            return max(1, min_health_needed - dungeon[row][col])
        
        return Min_Health_From(0, 0)
    
    def Calculate_Minimum_HP_Memoized(self, dungeon: List[List[int]]) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        m, n = len(dungeon), len(dungeon[0])
        memo = {}
        
        def Min_Health_Memo(row: int, col: int) -> int:
            if row >= m or col >= n:
                return float('inf')
            
            if (row, col) in memo:
                return memo[(row, col)]
            
            if row == m - 1 and col == n - 1:
                result = max(1, 1 - dungeon[row][col])
            else:
                right_health = Min_Health_Memo(row, col + 1)
                down_health = Min_Health_Memo(row + 1, col)
                
                min_health_needed = min(right_health, down_health)
                result = max(1, min_health_needed - dungeon[row][col])
            
            memo[(row, col)] = result
            return result
        
        return Min_Health_Memo(0, 0)
    
    def Calculate_Minimum_HP_Bottom_Up_Optimal(self, dungeon: List[List[int]]) -> int:
        """
        Bottom Up Optimal - Bottom-up DP from princess to knight
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        m, n = len(dungeon), len(dungeon[0])
        
        dp = [[0] * n for _ in range(m)]
        
        dp[m-1][n-1] = max(1, 1 - dungeon[m-1][n-1])
        
        for i in range(m - 2, -1, -1):
            dp[i][n-1] = max(1, dp[i+1][n-1] - dungeon[i][n-1])
        
        for j in range(n - 2, -1, -1):
            dp[m-1][j] = max(1, dp[m-1][j+1] - dungeon[m-1][j])
        
        for i in range(m - 2, -1, -1):
            for j in range(n - 2, -1, -1):
                min_health_needed = min(dp[i+1][j], dp[i][j+1])
                dp[i][j] = max(1, min_health_needed - dungeon[i][j])
        
        return dp[0][0]
    
    def Calculate_Minimum_HP_Space_Optimized(self, dungeon: List[List[int]]) -> int:
        """
        Space Optimized - Use 1D array
        Time Complexity: O(m*n)
        Space Complexity: O(n)
        """
        m, n = len(dungeon), len(dungeon[0])
        
        dp = [0] * n
        
        dp[n-1] = max(1, 1 - dungeon[m-1][n-1])
        
        for j in range(n - 2, -1, -1):
            dp[j] = max(1, dp[j+1] - dungeon[m-1][j])
        
        for i in range(m - 2, -1, -1):
            dp[n-1] = max(1, dp[n-1] - dungeon[i][n-1])
            
            for j in range(n - 2, -1, -1):
                min_health_needed = min(dp[j], dp[j+1])
                dp[j] = max(1, min_health_needed - dungeon[i][j])
        
        return dp[0]
    
    def Calculate_Minimum_HP_With_Path(self, dungeon: List[List[int]]) -> tuple:
        """
        With Path - Return minimum health and optimal path
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        m, n = len(dungeon), len(dungeon[0])
        
        dp = [[0] * n for _ in range(m)]
        path_choice = [['' for _ in range(n)] for _ in range(m)]
        
        dp[m-1][n-1] = max(1, 1 - dungeon[m-1][n-1])
        
        for i in range(m - 2, -1, -1):
            dp[i][n-1] = max(1, dp[i+1][n-1] - dungeon[i][n-1])
            path_choice[i][n-1] = 'D'
        
        for j in range(n - 2, -1, -1):
            dp[m-1][j] = max(1, dp[m-1][j+1] - dungeon[m-1][j])
            path_choice[m-1][j] = 'R'
        
        for i in range(m - 2, -1, -1):
            for j in range(n - 2, -1, -1):
                if dp[i+1][j] < dp[i][j+1]:
                    min_health_needed = dp[i+1][j]
                    path_choice[i][j] = 'D'
                else:
                    min_health_needed = dp[i][j+1]
                    path_choice[i][j] = 'R'
                
                dp[i][j] = max(1, min_health_needed - dungeon[i][j])
        
        path = []
        i, j = 0, 0
        
        while i < m - 1 or j < n - 1:
            if path_choice[i][j] == 'D':
                path.append('D')
                i += 1
            else:
                path.append('R')
                j += 1
        
        return dp[0][0], path
    
    def Calculate_Minimum_HP_Forward_DP(self, dungeon: List[List[int]]) -> int:
        """
        Forward DP - Alternative forward approach (less efficient)
        Time Complexity: O(m*n*max_health)
        Space Complexity: O(m*n*max_health)
        """
        m, n = len(dungeon), len(dungeon[0])
        
        max_possible_health = sum(max(0, cell) for row in dungeon for cell in row) + 1
        
        dp = [[[False] * (max_possible_health + 1) for _ in range(n)] for _ in range(m)]
        
        initial_health = max(1, 1 - dungeon[0][0])
        if initial_health <= max_possible_health:
            final_health = initial_health + dungeon[0][0]
            if final_health > 0:
                dp[0][0][final_health] = True
        
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                
                for health in range(1, max_possible_health + 1):
                    if i > 0 and dp[i-1][j][health - dungeon[i][j]] and health - dungeon[i][j] > 0:
                        dp[i][j][health] = True
                    
                    if j > 0 and dp[i][j-1][health - dungeon[i][j]] and health - dungeon[i][j] > 0:
                        dp[i][j][health] = True
        
        for health in range(1, max_possible_health + 1):
            if dp[m-1][n-1][health]:
                return health - dungeon[0][0]
        
        return self.Calculate_Minimum_HP_Bottom_Up_Optimal(dungeon)
    
    def Calculate_Minimum_HP_Simulation(self, dungeon: List[List[int]]) -> int:
        """
        Simulation - Binary search on initial health
        Time Complexity: O(log(sum) * m * n)
        Space Complexity: O(1)
        """
        def Can_Survive_With_Health(initial_health: int) -> bool:
            m, n = len(dungeon), len(dungeon[0])
            health = initial_health
            
            for i in range(m):
                for j in range(n):
                    if (i == 0 and j == 0) or (i > 0 and j == 0) or (j > 0 and i == 0) or (i > 0 and j > 0):
                        health += dungeon[i][j]
                        if health <= 0:
                            return False
            
            return True
        
        left, right = 1, abs(sum(min(0, cell) for row in dungeon for cell in row)) + 1
        
        while left < right:
            mid = (left + right) // 2
            
            if Can_Survive_With_Health(mid):
                right = mid
            else:
                left = mid + 1
        
        return left

def Test_Calculate_Minimum_HP():
    solution = Solution()
    
    test_cases = [
        ([[-3,5]], 4),
        ([[-2,-3,3],[-5,-10,1],[10,30,-5]], 7),
        ([[1,-3,3],[0,-2,0],[-3,-3,-3]], 3),
        ([[0]], 1),
        ([[1,2,3],[4,5,6]], 1),
        ([[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]], 4)
    ]
    
    methods = [
        ("Recursive", solution.Calculate_Minimum_HP_Recursive),
        ("Memoized", solution.Calculate_Minimum_HP_Memoized),
        ("Bottom Up Optimal", solution.Calculate_Minimum_HP_Bottom_Up_Optimal),
        ("Space Optimized", solution.Calculate_Minimum_HP_Space_Optimized),
        ("Simulation", solution.Calculate_Minimum_HP_Simulation)
    ]
    
    for dungeon, expected in test_cases:
        print(f"Dungeon: {dungeon}")
        print(f"Expected: {expected}")
        
        if len(dungeon) <= 3 and len(dungeon[0]) <= 3:
            result_bf = solution.Calculate_Minimum_HP_Brute_Force([row[:] for row in dungeon])
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method([row[:] for row in dungeon])
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        min_health, path = solution.Calculate_Minimum_HP_With_Path([row[:] for row in dungeon])
        print(f"With Path: Health={min_health}, Path={''.join(path)}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Calculate_Minimum_HP()
