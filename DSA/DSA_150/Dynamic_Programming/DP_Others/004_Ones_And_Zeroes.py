"""
Problem: Ones and Zeroes
URL: https://leetcode.com/problems/ones-and-zeroes/

Problem Statement:
You are given an array of binary strings strs and two integers m and n.
Return the size of the largest subset of strs such that there are at most m 0's and n 1's in the subset.
A set x is a subset of a set y if all elements of x are also elements of y.

Sample Input/Output:
Input: strs = ["10","0001","111001","1","0"], m = 5, n = 3
Output: 4
Explanation: The largest subset with at most 5 0's and 3 1's is {"10", "0001", "1", "0"}, so the answer is 4.

Input: strs = ["10","0","1"], m = 1, n = 1
Output: 2
Explanation: The largest subset is {"0", "1"}, so the answer is 2.
"""

from typing import List, Tuple

class Solution:
    def Find_Max_Form_Brute_Force(self, strs: List[str], m: int, n: int) -> int:
        """
        Brute Force - Try all possible subsets
        Time Complexity: O(2^len(strs))
        Space Complexity: O(len(strs))
        """
        def Count_Zeros_Ones(s: str) -> Tuple[int, int]:
            zeros = s.count('0')
            ones = s.count('1')
            return zeros, ones
        
        def Generate_All_Subsets(index: int, current_subset: List[str]) -> int:
            if index >= len(strs):
                total_zeros = sum(Count_Zeros_Ones(s)[0] for s in current_subset)
                total_ones = sum(Count_Zeros_Ones(s)[1] for s in current_subset)
                
                if total_zeros <= m and total_ones <= n:
                    return len(current_subset)
                return 0
            
            exclude = Generate_All_Subsets(index + 1, current_subset)
            
            current_subset.append(strs[index])
            include = Generate_All_Subsets(index + 1, current_subset)
            current_subset.pop()
            
            return max(exclude, include)
        
        return Generate_All_Subsets(0, [])
    
    def Find_Max_Form_Recursive(self, strs: List[str], m: int, n: int) -> int:
        """
        Recursive - Try include/exclude for each string
        Time Complexity: O(2^len(strs))
        Space Complexity: O(len(strs))
        """
        def Count_Zeros_Ones(s: str) -> Tuple[int, int]:
            zeros = s.count('0')
            ones = s.count('1')
            return zeros, ones
        
        def Max_Form_Rec(index: int, zeros_left: int, ones_left: int) -> int:
            if index >= len(strs):
                return 0
            
            zeros, ones = Count_Zeros_Ones(strs[index])
            
            exclude = Max_Form_Rec(index + 1, zeros_left, ones_left)
            
            if zeros <= zeros_left and ones <= ones_left:
                include = 1 + Max_Form_Rec(index + 1, zeros_left - zeros, ones_left - ones)
                return max(include, exclude)
            
            return exclude
        
        return Max_Form_Rec(0, m, n)
    
    def Find_Max_Form_Memoized(self, strs: List[str], m: int, n: int) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(len(strs) * m * n)
        Space Complexity: O(len(strs) * m * n)
        """
        def Count_Zeros_Ones(s: str) -> Tuple[int, int]:
            zeros = s.count('0')
            ones = s.count('1')
            return zeros, ones
        
        memo = {}
        
        def Max_Form_Memo(index: int, zeros_left: int, ones_left: int) -> int:
            if index >= len(strs):
                return 0
            
            if (index, zeros_left, ones_left) in memo:
                return memo[(index, zeros_left, ones_left)]
            
            zeros, ones = Count_Zeros_Ones(strs[index])
            
            exclude = Max_Form_Memo(index + 1, zeros_left, ones_left)
            
            if zeros <= zeros_left and ones <= ones_left:
                include = 1 + Max_Form_Memo(index + 1, zeros_left - zeros, ones_left - ones)
                result = max(include, exclude)
            else:
                result = exclude
            
            memo[(index, zeros_left, ones_left)] = result
            return result
        
        return Max_Form_Memo(0, m, n)
    
    def Find_Max_Form_3D_DP_Optimal(self, strs: List[str], m: int, n: int) -> int:
        """
        3D DP Optimal - Bottom-up DP with 3D table
        Time Complexity: O(len(strs) * m * n)
        Space Complexity: O(len(strs) * m * n)
        """
        def Count_Zeros_Ones(s: str) -> Tuple[int, int]:
            zeros = s.count('0')
            ones = s.count('1')
            return zeros, ones
        
        k = len(strs)
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(k + 1)]
        
        for i in range(1, k + 1):
            zeros, ones = Count_Zeros_Ones(strs[i - 1])
            
            for j in range(m + 1):
                for l in range(n + 1):
                    dp[i][j][l] = dp[i - 1][j][l]
                    
                    if j >= zeros and l >= ones:
                        dp[i][j][l] = max(dp[i][j][l], dp[i - 1][j - zeros][l - ones] + 1)
        
        return dp[k][m][n]
    
    def Find_Max_Form_2D_DP_Space_Optimized(self, strs: List[str], m: int, n: int) -> int:
        """
        2D DP Space Optimized - Optimize space to 2D
        Time Complexity: O(len(strs) * m * n)
        Space Complexity: O(m * n)
        """
        def Count_Zeros_Ones(s: str) -> Tuple[int, int]:
            zeros = s.count('0')
            ones = s.count('1')
            return zeros, ones
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for s in strs:
            zeros, ones = Count_Zeros_Ones(s)
            
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
        
        return dp[m][n]
    
    def Find_Max_Form_With_Subset(self, strs: List[str], m: int, n: int) -> Tuple[int, List[str]]:
        """
        With Subset - Return max count and actual subset
        Time Complexity: O(len(strs) * m * n)
        Space Complexity: O(len(strs) * m * n)
        """
        def Count_Zeros_Ones(s: str) -> Tuple[int, int]:
            zeros = s.count('0')
            ones = s.count('1')
            return zeros, ones
        
        k = len(strs)
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(k + 1)]
        choice = [[[False] * (n + 1) for _ in range(m + 1)] for _ in range(k + 1)]
        
        for i in range(1, k + 1):
            zeros, ones = Count_Zeros_Ones(strs[i - 1])
            
            for j in range(m + 1):
                for l in range(n + 1):
                    dp[i][j][l] = dp[i - 1][j][l]
                    choice[i][j][l] = False
                    
                    if j >= zeros and l >= ones:
                        include_value = dp[i - 1][j - zeros][l - ones] + 1
                        if include_value > dp[i][j][l]:
                            dp[i][j][l] = include_value
                            choice[i][j][l] = True
        
        subset = []
        i, j, l = k, m, n
        
        while i > 0:
            if choice[i][j][l]:
                subset.append(strs[i - 1])
                zeros, ones = Count_Zeros_Ones(strs[i - 1])
                j -= zeros
                l -= ones
            i -= 1
        
        return dp[k][m][n], subset[::-1]
    
    def Find_Max_Form_Greedy_Comparison(self, strs: List[str], m: int, n: int) -> Tuple[int, int]:
        """
        Greedy Comparison - Compare with greedy approach
        Time Complexity: O(len(strs) * log(len(strs)))
        Space Complexity: O(len(strs))
        """
        def Count_Zeros_Ones(s: str) -> Tuple[int, int]:
            zeros = s.count('0')
            ones = s.count('1')
            return zeros, ones
        
        def Greedy_Approach() -> int:
            string_costs = [(Count_Zeros_Ones(s), i) for i, s in enumerate(strs)]
            string_costs.sort(key=lambda x: x[0][0] + x[0][1])
            
            count = 0
            zeros_used = 0
            ones_used = 0
            
            for (zeros, ones), _ in string_costs:
                if zeros_used + zeros <= m and ones_used + ones <= n:
                    zeros_used += zeros
                    ones_used += ones
                    count += 1
            
            return count
        
        greedy_result = Greedy_Approach()
        optimal_result = self.Find_Max_Form_2D_DP_Space_Optimized(strs, m, n)
        
        return optimal_result, greedy_result
    
    def Find_Max_Form_Bottom_Up_Alternative(self, strs: List[str], m: int, n: int) -> int:
        """
        Bottom Up Alternative - Different iteration order
        Time Complexity: O(len(strs) * m * n)
        Space Complexity: O(m * n)
        """
        def Count_Zeros_Ones(s: str) -> Tuple[int, int]:
            return s.count('0'), s.count('1')
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for string in strs:
            zeros, ones = Count_Zeros_Ones(string)
            
            for zeros_budget in range(m, zeros - 1, -1):
                for ones_budget in range(n, ones - 1, -1):
                    dp[zeros_budget][ones_budget] = max(
                        dp[zeros_budget][ones_budget],
                        dp[zeros_budget - zeros][ones_budget - ones] + 1
                    )
        
        return dp[m][n]

def Test_Find_Max_Form():
    solution = Solution()
    
    test_cases = [
        (["10","0001","111001","1","0"], 5, 3, 4),
        (["10","0","1"], 1, 1, 2),
        (["10","0001","111001","1","0"], 3, 4, 3),
        (["0","1","10","101"], 2, 2, 3),
        (["111","1000","1000","1000"], 9, 3, 3)
    ]
    
    methods = [
        ("Recursive", solution.Find_Max_Form_Recursive),
        ("Memoized", solution.Find_Max_Form_Memoized),
        ("3D DP Optimal", solution.Find_Max_Form_3D_DP_Optimal),
        ("2D DP Space Optimized", solution.Find_Max_Form_2D_DP_Space_Optimized),
        ("Bottom Up Alternative", solution.Find_Max_Form_Bottom_Up_Alternative)
    ]
    
    for strs, m, n, expected in test_cases:
        print(f"Strings: {strs}, m: {m}, n: {n}")
        print(f"Expected: {expected}")
        
        if len(strs) <= 8:
            result_bf = solution.Find_Max_Form_Brute_Force(strs.copy(), m, n)
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(strs.copy(), m, n)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        max_count, subset = solution.Find_Max_Form_With_Subset(strs.copy(), m, n)
        print(f"With Subset: Count={max_count}, Subset={subset}")
        
        optimal, greedy = solution.Find_Max_Form_Greedy_Comparison(strs.copy(), m, n)
        print(f"Comparison: Optimal={optimal}, Greedy={greedy}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Find_Max_Form()
