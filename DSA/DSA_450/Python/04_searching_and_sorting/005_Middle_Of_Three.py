"""
Problem: Middle of Three
URL: https://practice.geeksforgeeks.org/problems/middle-of-three2926/1

Problem Statement:
Given three distinct numbers A, B and C. Find the number with value in middle (Try to do it with minimum comparisons).

Sample Input/Output:
Input: A = 978, B = 518, C = 300
Output: 518

Input: A = 162, B = 934, C = 200
Output: 200
"""


class Solution:
    def Middle_Of_Three_Sum_Method(self, A, B, C):
        """
        Using sum minus min minus max to find middle
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return A + B + C - min(A, B, C) - max(A, B, C)

    def Middle_Of_Three_Comparisons(self, A, B, C):
        """
        Using comparisons to find middle element
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if (A > B and A < C) or (A < B and A > C):
            return A
        elif (B > A and B < C) or (B < A and B > C):
            return B
        else:
            return C


def Test_Middle_Of_Three():
    sol = Solution()
    tests = [
        [978, 518, 300],
        [162, 934, 200],
        [1, 2, 3],
        [10, 5, 8],
        [100, 50, 75]
    ]

    for test in tests:
        A, B, C = test[0], test[1], test[2]
        print(f"A = {A}, B = {B}, C = {C}")
        
        res1 = sol.Middle_Of_Three_Sum_Method(A, B, C)
        print(f"Sum Method: {res1}")
        
        res2 = sol.Middle_Of_Three_Comparisons(A, B, C)
        print(f"Comparisons: {res2}")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Middle_Of_Three()
