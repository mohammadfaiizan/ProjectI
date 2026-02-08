"""
Problem: Minimum Sum Absolute Difference Pairs
URL: https://www.geeksforgeeks.org/minimum-sum-absolute-difference-pairs-two-arrays/

Problem Statement:
Given two arrays, pair elements to minimize sum of absolute differences.

Sample Input/Output:
Input: a[] = {4, 1, 8, 7}, b[] = {2, 3, 6, 5}
Output: 6
Explanation: Pair (1,2), (4,3), (7,5), (8,6). Sum = |1-2| + |4-3| + |7-5| + |8-6| = 1 + 1 + 2 + 2 = 6
"""


class Solution:
    def Min_Sum_Absolute_Diff_Pairs_Sort_Both(self, a, b):
        """
        Sort both + pair corresponding greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        a.sort()
        b.sort()
        
        sum_val = 0
        n = len(a)
        
        for i in range(n):
            sum_val += abs(a[i] - b[i])
        
        return sum_val


def Test_Min_Sum_Absolute_Diff_Pairs():
    solution = Solution()
    
    a1 = [4, 1, 8, 7]
    b1 = [2, 3, 6, 5]
    print(f"Test 1: {solution.Min_Sum_Absolute_Diff_Pairs_Sort_Both(a1, b1)}")
    
    a2 = [4, 1, 2]
    b2 = [2, 4, 1]
    print(f"Test 2: {solution.Min_Sum_Absolute_Diff_Pairs_Sort_Both(a2, b2)}")
    
    a3 = [1, 2, 3]
    b3 = [3, 2, 1]
    print(f"Test 3: {solution.Min_Sum_Absolute_Diff_Pairs_Sort_Both(a3, b3)}")


if __name__ == "__main__":
    Test_Min_Sum_Absolute_Diff_Pairs()
