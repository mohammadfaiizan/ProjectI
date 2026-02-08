"""
Problem: Maximum Difference of Zeros and Ones in Binary String
URL: https://practice.geeksforgeeks.org/problems/maximum-difference-of-zeros-and-ones-in-binary-string4111/1

Problem Statement:
Given a binary string S of 0s and 1s. The task is to find the maximum difference of the number of 0s and the number of 1s (number of 0s - number of 1s) in the substrings of a string.

Sample Input/Output:
Input: "11000010001"
Output: 6
"""

class Solution:
    def Max_Diff_Kadane(self, s):
        """
        Kadane's algorithm approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(s)
        max_diff = -1
        curr_diff = 0
        for i in range(n):
            val = 1 if s[i] == '0' else -1
            curr_diff += val
            if curr_diff < 0:
                curr_diff = 0
            max_diff = max(max_diff, curr_diff)
        return max_diff

def Test_Max_Diff():
    solution = Solution()
    s = "11000010001"
    print("Max Difference:", solution.Max_Diff_Kadane(s))

if __name__ == "__main__":
    Test_Max_Diff()
