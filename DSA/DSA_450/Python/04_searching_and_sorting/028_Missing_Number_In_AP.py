"""
Problem: Arithmetic Number / Missing Number in AP
URL: https://practice.geeksforgeeks.org/problems/arithmetic-number2815/1

Problem Statement:
Given first term A, last term B, and common difference C, check if B exists in the AP.

Sample Input:
A = 1, B = 3, C = 2

Sample Output:
1
"""


class Solution:
    def InSequence_Math(self, A, B, C):
        """
        Approach: Use mathematical formula to check if B exists in AP
        Formula: B = A + n*C where n >= 0
        Rearranging: n = (B - A) / C
        B exists if (B - A) is divisible by C and n >= 0
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if C == 0:
            return 1 if A == B else 0
        diff = B - A
        if (diff > 0 and C < 0) or (diff < 0 and C > 0):
            return 0
        return 1 if diff % C == 0 else 0

    def InSequence_Iterative(self, A, B, C):
        """
        Approach: Iteratively check each term in the AP until we reach or exceed B
        Start from A and keep adding C until we reach B or exceed it
        
        Time Complexity: O(n) where n is the number of terms
        Space Complexity: O(1)
        """
        if C == 0:
            return 1 if A == B else 0
        current = A
        if C > 0:
            while current < B:
                current += C
        else:
            while current > B:
                current += C
        return 1 if current == B else 0


def Test_Missing_Number_In_AP():
    sol = Solution()
    
    assert sol.InSequence_Math(1, 3, 2) == 1
    assert sol.InSequence_Math(1, 2, 2) == 1
    assert sol.InSequence_Math(1, 5, 2) == 1
    assert sol.InSequence_Math(1, 4, 2) == 0
    assert sol.InSequence_Math(5, 1, -2) == 1
    assert sol.InSequence_Math(1, 1, 0) == 1
    assert sol.InSequence_Math(1, 2, 0) == 0
    
    assert sol.InSequence_Iterative(1, 3, 2) == 1
    assert sol.InSequence_Iterative(1, 2, 2) == 1
    assert sol.InSequence_Iterative(1, 5, 2) == 1
    assert sol.InSequence_Iterative(1, 4, 2) == 0
    assert sol.InSequence_Iterative(5, 1, -2) == 1
    assert sol.InSequence_Iterative(1, 1, 0) == 1
    assert sol.InSequence_Iterative(1, 2, 0) == 0
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Missing_Number_In_AP()
