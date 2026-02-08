"""
Problem: Tug Of War
URL: https://www.geeksforgeeks.org/tug-of-war/

Problem Statement:
Divide set of integers into two subsets of sizes n/2 and ceil(n/2) such that difference of their sums is minimized.

Sample Input/Output:
Input: arr[]={23,45,-34,12,0,98,-99,4,189,-1,4}
Output: Min difference: 1
Explanation: Subsets with minimum difference
"""


class Solution:
    def Tug_Of_War_Backtracking(self, arr):
        """
        Backtracking with subset size constraint
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        n = len(arr)
        total_sum = sum(arr)
        min_diff = float('inf')
        selected = [False] * n
        best_selection = [False] * n
        
        def backtrack(index, count, current_sum):
            nonlocal min_diff, best_selection
            
            if count == n // 2:
                diff = abs(total_sum - 2 * current_sum)
                if diff < min_diff:
                    min_diff = diff
                    best_selection = selected[:]
                return
            
            if index >= n:
                return
            
            selected[index] = True
            backtrack(index + 1, count + 1, current_sum + arr[index])
            
            selected[index] = False
            backtrack(index + 1, count, current_sum)
        
        backtrack(0, 0, 0)
        return min_diff


def Test_Tug_Of_War():
    solution = Solution()
    
    arr = [23, 45, -34, 12, 0, 98, -99, 4, 189, -1, 4]
    min_diff = solution.Tug_Of_War_Backtracking(arr)
    
    print("Minimum difference:", min_diff)


if __name__ == "__main__":
    Test_Tug_Of_War()
