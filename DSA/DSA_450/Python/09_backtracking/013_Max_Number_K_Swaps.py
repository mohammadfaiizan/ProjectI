"""
Problem: Maximum Number in K Swaps
URL: https://practice.geeksforgeeks.org/problems/largest-number-in-k-swaps-1587115620/1

Problem Statement:
Given a number as a string and K swaps allowed, find the maximum possible number.

Sample Input/Output:
Input: str = "1234567", K = 4
Output: "7654321"
Explanation: After 4 swaps, we get maximum number
"""


class Solution:
    def Max_Number_K_Swaps_Backtracking(self, str_val, k):
        """
        Backtracking all pairs
        Time Complexity: O((n^2)^k)
        Space Complexity: O(n)
        """
        max_str = str_val
        
        def backtrack(s, swaps_left, start):
            nonlocal max_str
            
            if swaps_left == 0 or start >= len(s):
                if s > max_str:
                    max_str = s
                return
            
            s_list = list(s)
            for i in range(start, len(s_list)):
                for j in range(i + 1, len(s_list)):
                    if s_list[j] > s_list[i]:
                        s_list[i], s_list[j] = s_list[j], s_list[i]
                        backtrack(''.join(s_list), swaps_left - 1, start + 1)
                        s_list[i], s_list[j] = s_list[j], s_list[i]
            
            if s > max_str:
                max_str = s
        
        backtrack(str_val, k, 0)
        return max_str
    
    def Max_Number_K_Swaps_Optimized(self, str_val, k):
        """
        Optimized find max digit first
        Time Complexity: O(n^k)
        Space Complexity: O(n)
        """
        max_str = str_val
        
        def backtrack(s, swaps_left, idx):
            nonlocal max_str
            
            if swaps_left == 0 or idx >= len(s):
                if s > max_str:
                    max_str = s
                return
            
            s_list = list(s)
            max_char = s_list[idx]
            for i in range(idx + 1, len(s_list)):
                if s_list[i] > max_char:
                    max_char = s_list[i]
            
            if max_char == s_list[idx]:
                backtrack(s, swaps_left, idx + 1)
            else:
                for i in range(idx + 1, len(s_list)):
                    if s_list[i] == max_char:
                        s_list[idx], s_list[i] = s_list[i], s_list[idx]
                        backtrack(''.join(s_list), swaps_left - 1, idx + 1)
                        s_list[idx], s_list[i] = s_list[i], s_list[idx]
        
        backtrack(str_val, k, 0)
        return max_str


def Test_Max_Number_K_Swaps():
    solution = Solution()
    str_val = "1234567"
    k = 4
    print("Original:", str_val)
    print("Backtracking Approach:", solution.Max_Number_K_Swaps_Backtracking(str_val, k))
    print("Optimized Approach:", solution.Max_Number_K_Swaps_Optimized(str_val, k))


if __name__ == "__main__":
    Test_Max_Number_K_Swaps()
