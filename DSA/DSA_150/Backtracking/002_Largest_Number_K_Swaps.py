"""
Problem: Largest Number in K Swaps
URL: https://www.geeksforgeeks.org/problems/largest-number-in-k-swaps-1587115620/1

Problem Statement:
Given a number K and string str of digits denoting a positive integer, 
build the largest number possible by performing swap operations on the digits of str at most K times.

Sample Input/Output:
Input: K = 4, str = "1234567"
Output: "7654321"
Explanation: Three swaps can make the input 1234567 to 7654321, swapping 1 with 7, 2 with 6 and 3 with 5

Input: K = 3, str = "3435335"
Output: "5543333"
Explanation: Three swaps can make the input 3435335 to 5543333, swapping 3 with 5, 4 with 5 and 3 with 3
"""

from typing import List

class Solution:
    def Find_Maximum_Num_Brute_Force(self, num: str, k: int) -> str:
        """
        Brute Force - Try all possible swaps
        Time Complexity: O(n^(2k))
        Space Complexity: O(k)
        """
        def Generate_All_Swaps(current_num: str, swaps_left: int, results: List[str]) -> None:
            results.append(current_num)
            
            if swaps_left == 0:
                return
            
            for i in range(len(current_num)):
                for j in range(i + 1, len(current_num)):
                    num_list = list(current_num)
                    num_list[i], num_list[j] = num_list[j], num_list[i]
                    new_num = ''.join(num_list)
                    Generate_All_Swaps(new_num, swaps_left - 1, results)
        
        all_results = []
        Generate_All_Swaps(num, k, all_results)
        return max(all_results)
    
    def Find_Maximum_Num_Backtracking_Optimal(self, num: str, k: int) -> str:
        """
        Backtracking Approach - Optimal solution with pruning
        Time Complexity: O(n^(2k))
        Space Complexity: O(k)
        """
        max_num = [num]
        
        def Backtrack(current_num: List[str], swaps_left: int) -> None:
            max_num[0] = max(max_num[0], ''.join(current_num))
            
            if swaps_left == 0:
                return
            
            for i in range(len(current_num)):
                for j in range(i + 1, len(current_num)):
                    current_num[i], current_num[j] = current_num[j], current_num[i]
                    Backtrack(current_num, swaps_left - 1)
                    current_num[i], current_num[j] = current_num[j], current_num[i]
        
        Backtrack(list(num), k)
        return max_num[0]
    
    def Find_Maximum_Num_Greedy_Backtrack(self, num: str, k: int) -> str:
        """
        Greedy with Backtracking - Choose best swaps first
        Time Complexity: O(n^2 * k)
        Space Complexity: O(k)
        """
        max_num = [num]
        
        def Backtrack(current_num: List[str], pos: int, swaps_left: int) -> None:
            max_num[0] = max(max_num[0], ''.join(current_num))
            
            if swaps_left == 0 or pos >= len(current_num):
                return
            
            max_digit = current_num[pos]
            for i in range(pos + 1, len(current_num)):
                max_digit = max(max_digit, current_num[i])
            
            if max_digit != current_num[pos]:
                for i in range(len(current_num) - 1, pos, -1):
                    if current_num[i] == max_digit:
                        current_num[pos], current_num[i] = current_num[i], current_num[pos]
                        Backtrack(current_num, pos + 1, swaps_left - 1)
                        current_num[pos], current_num[i] = current_num[i], current_num[pos]
            
            Backtrack(current_num, pos + 1, swaps_left)
        
        Backtrack(list(num), 0, k)
        return max_num[0]
    
    def Find_Maximum_Num_Memoized(self, num: str, k: int) -> str:
        """
        Memoized Backtracking - Cache results for efficiency
        Time Complexity: O(n^(2k))
        Space Complexity: O(n^k)
        """
        memo = {}
        
        def Backtrack(current_num: str, swaps_left: int) -> str:
            if (current_num, swaps_left) in memo:
                return memo[(current_num, swaps_left)]
            
            if swaps_left == 0:
                return current_num
            
            max_result = current_num
            current_list = list(current_num)
            
            for i in range(len(current_list)):
                for j in range(i + 1, len(current_list)):
                    current_list[i], current_list[j] = current_list[j], current_list[i]
                    new_num = ''.join(current_list)
                    result = Backtrack(new_num, swaps_left - 1)
                    max_result = max(max_result, result)
                    current_list[i], current_list[j] = current_list[j], current_list[i]
            
            memo[(current_num, swaps_left)] = max_result
            return max_result
        
        return Backtrack(num, k)

def Test_Find_Maximum_Num():
    solution = Solution()
    
    test_cases = [
        ("1234567", 4, "7654321"),
        ("3435335", 3, "5543333"),
        ("254", 1, "524"),
        ("254", 2, "542"),
        ("68543", 1, "86543")
    ]
    
    for num, k, expected in test_cases:
        result1 = solution.Find_Maximum_Num_Brute_Force(num, k)
        result2 = solution.Find_Maximum_Num_Backtracking_Optimal(num, k)
        result3 = solution.Find_Maximum_Num_Greedy_Backtrack(num, k)
        result4 = solution.Find_Maximum_Num_Memoized(num, k)
        
        print(f"Number: {num}, K: {k}")
        print(f"Expected: {expected}")
        print(f"Brute Force: {result1}")
        print(f"Backtracking Optimal: {result2}")
        print(f"Greedy Backtrack: {result3}")
        print(f"Memoized: {result4}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Find_Maximum_Num()
