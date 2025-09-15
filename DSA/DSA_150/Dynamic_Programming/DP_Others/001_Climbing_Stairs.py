"""
Problem: Climbing Stairs
URL: https://leetcode.com/problems/climbing-stairs/

Problem Statement:
You are climbing a staircase. It takes n steps to reach the top.
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Sample Input/Output:
Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
"""

from typing import List

class Solution:
    def Climb_Stairs_Recursive(self, n: int) -> int:
        """
        Recursive - Try both 1 and 2 steps at each position
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        if n <= 1:
            return 1
        
        return self.Climb_Stairs_Recursive(n - 1) + self.Climb_Stairs_Recursive(n - 2)
    
    def Climb_Stairs_Memoized(self, n: int) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        memo = {}
        
        def Climb_Memo(steps: int) -> int:
            if steps <= 1:
                return 1
            
            if steps in memo:
                return memo[steps]
            
            memo[steps] = Climb_Memo(steps - 1) + Climb_Memo(steps - 2)
            return memo[steps]
        
        return Climb_Memo(n)
    
    def Climb_Stairs_Tabulation_Optimal(self, n: int) -> int:
        """
        Tabulation Optimal - Bottom-up DP
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if n <= 1:
            return 1
        
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        
        return dp[n]
    
    def Climb_Stairs_Space_Optimized(self, n: int) -> int:
        """
        Space Optimized - Use only two variables
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n <= 1:
            return 1
        
        prev2 = 1
        prev1 = 1
        
        for i in range(2, n + 1):
            current = prev1 + prev2
            prev2 = prev1
            prev1 = current
        
        return prev1
    
    def Climb_Stairs_Matrix_Exponentiation(self, n: int) -> int:
        """
        Matrix Exponentiation - Use matrix multiplication
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if n <= 1:
            return 1
        
        def Matrix_Multiply(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
            return [[A[0][0] * B[0][0] + A[0][1] * B[1][0],
                     A[0][0] * B[0][1] + A[0][1] * B[1][1]],
                    [A[1][0] * B[0][0] + A[1][1] * B[1][0],
                     A[1][0] * B[0][1] + A[1][1] * B[1][1]]]
        
        def Matrix_Power(matrix: List[List[int]], power: int) -> List[List[int]]:
            if power == 1:
                return matrix
            
            if power % 2 == 0:
                half_power = Matrix_Power(matrix, power // 2)
                return Matrix_Multiply(half_power, half_power)
            else:
                return Matrix_Multiply(matrix, Matrix_Power(matrix, power - 1))
        
        base_matrix = [[1, 1], [1, 0]]
        result_matrix = Matrix_Power(base_matrix, n)
        
        return result_matrix[0][0]
    
    def Climb_Stairs_Fibonacci_Formula(self, n: int) -> int:
        """
        Fibonacci Formula - Use Binet's formula
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        import math
        
        sqrt5 = math.sqrt(5)
        phi = (1 + sqrt5) / 2
        psi = (1 - sqrt5) / 2
        
        return int((phi**(n+1) - psi**(n+1)) / sqrt5)
    
    def Climb_Stairs_Generalized_K_Steps(self, n: int, k: int = 2) -> int:
        """
        Generalized K Steps - Allow 1 to k steps at each position
        Time Complexity: O(n*k)
        Space Complexity: O(n)
        """
        if n <= 1:
            return 1
        
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i] += dp[i - j]
        
        return dp[n]
    
    def Climb_Stairs_With_Path_Count(self, n: int) -> tuple:
        """
        With Path Count - Return count and all possible paths
        Time Complexity: O(2^n)
        Space Complexity: O(2^n)
        """
        def Generate_Paths(steps_left: int, current_path: List[int]) -> List[List[int]]:
            if steps_left == 0:
                return [current_path[:]]
            
            paths = []
            
            if steps_left >= 1:
                current_path.append(1)
                paths.extend(Generate_Paths(steps_left - 1, current_path))
                current_path.pop()
            
            if steps_left >= 2:
                current_path.append(2)
                paths.extend(Generate_Paths(steps_left - 2, current_path))
                current_path.pop()
            
            return paths
        
        all_paths = Generate_Paths(n, [])
        return len(all_paths), all_paths
    
    def Climb_Stairs_Bottom_Up_Alternative(self, n: int) -> int:
        """
        Bottom Up Alternative - Different initialization
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n == 0:
            return 1
        if n == 1:
            return 1
        
        first = 1
        second = 1
        
        for i in range(2, n + 1):
            third = first + second
            first = second
            second = third
        
        return second

def Test_Climb_Stairs():
    solution = Solution()
    
    test_cases = [
        (2, 2),
        (3, 3),
        (4, 5),
        (5, 8),
        (10, 89),
        (1, 1),
        (0, 1)
    ]
    
    methods = [
        ("Memoized", solution.Climb_Stairs_Memoized),
        ("Tabulation Optimal", solution.Climb_Stairs_Tabulation_Optimal),
        ("Space Optimized", solution.Climb_Stairs_Space_Optimized),
        ("Matrix Exponentiation", solution.Climb_Stairs_Matrix_Exponentiation),
        ("Fibonacci Formula", solution.Climb_Stairs_Fibonacci_Formula),
        ("Bottom Up Alternative", solution.Climb_Stairs_Bottom_Up_Alternative)
    ]
    
    for n, expected in test_cases:
        print(f"Steps: {n}")
        print(f"Expected: {expected}")
        
        if n <= 10:
            result_rec = solution.Climb_Stairs_Recursive(n)
            print(f"Recursive: {result_rec}")
        
        for method_name, method in methods:
            try:
                result = method(n)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        if n <= 6:
            count, paths = solution.Climb_Stairs_With_Path_Count(n)
            print(f"With Paths: Count={count}")
            for path in paths:
                print(f"  {path}")
        
        generalized_result = solution.Climb_Stairs_Generalized_K_Steps(n, 3)
        print(f"Generalized (k=3): {generalized_result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Climb_Stairs()
