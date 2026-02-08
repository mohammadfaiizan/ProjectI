"""
Problem: Maximum Sum Consecutive Difference Circular
URL: https://practice.geeksforgeeks.org/problems/swap-and-maximize5859/1

Problem Statement:
Rearrange circular array to maximize sum of |arr[i]-arr[i+1]|.

Sample Input/Output:
Input: arr[] = {4, 2, 1, 8}
Output: 18
Explanation: Rearrange to {1, 8, 2, 4}. Sum = |1-8| + |8-2| + |2-4| + |4-1| = 18
"""


class Solution:
    def Max_Sum_Consecutive_Diff_Circular_Sort(self, arr):
        """
        Sort + sum 2*(large-small) greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr.sort()
        n = len(arr)
        sum_val = 0
        
        for i in range(n // 2):
            sum_val += 2 * (arr[n - 1 - i] - arr[i])
        
        return sum_val


def Test_Max_Sum_Consecutive_Diff_Circular():
    solution = Solution()
    
    arr1 = [4, 2, 1, 8]
    print(f"Test 1: {solution.Max_Sum_Consecutive_Diff_Circular_Sort(arr1)}")
    
    arr2 = [1, 2, 3, 4, 5]
    print(f"Test 2: {solution.Max_Sum_Consecutive_Diff_Circular_Sort(arr2)}")
    
    arr3 = [10, 12]
    print(f"Test 3: {solution.Max_Sum_Consecutive_Diff_Circular_Sort(arr3)}")


if __name__ == "__main__":
    Test_Max_Sum_Consecutive_Diff_Circular()
