"""
Problem: Combination Sum
URL: https://practice.geeksforgeeks.org/problems/combination-sum-1587115620/1

Problem Statement:
Given an array of distinct integers and a target sum, find all unique combinations that sum to the target. The same number can be used unlimited times.

Sample Input/Output:
Input: arr = [2, 3, 6, 7], target = 7
Output: [[2, 2, 3], [7]]
Explanation: 2+2+3=7 and 7=7
"""


class Solution:
    def Combination_Sum_Backtracking(self, arr, target):
        """
        Backtracking with index tracking
        Time Complexity: O(2^t * k) where t=target, k=avg combo length
        Space Complexity: O(k)
        """
        arr.sort()
        result = []
        current = []
        
        def backtrack(idx, remaining):
            if remaining == 0:
                result.append(current[:])
                return
            
            for i in range(idx, len(arr)):
                if arr[i] > remaining:
                    break
                current.append(arr[i])
                backtrack(i, remaining - arr[i])
                current.pop()
        
        backtrack(0, target)
        return result


def Test_Combination_Sum():
    solution = Solution()
    arr = [2, 3, 6, 7]
    target = 7
    result = solution.Combination_Sum_Backtracking(arr, target)
    print(f"Combinations for target {target}:")
    for combo in result:
        print(" ".join(str(num) for num in combo))


if __name__ == "__main__":
    Test_Combination_Sum()
