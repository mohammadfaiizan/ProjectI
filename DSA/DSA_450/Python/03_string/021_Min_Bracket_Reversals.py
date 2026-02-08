"""
Problem: Minimum Number of Bracket Reversals
URL: https://practice.geeksforgeeks.org/problems/count-the-reversals0401/1

Problem Statement:
Given a string consisting of only '{' and '}', find the minimum number of
reversals required to make the expression balanced. Return -1 if not possible.

Sample Input/Output:
Input: "}}{{"
Output: 2

Input: "{{{"
Output: -1

Input: "{{}{{{}}{{"
Output: 3
"""

import math


class Solution:
    def Min_Reversals_Stack(self, str_val):
        """
        Remove balanced pairs using stack, then compute from remaining
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        length = len(str_val)
        if length % 2:
            return -1

        stack = []
        for i in range(length):
            if str_val[i] == '}' and stack and stack[-1] == '{':
                stack.pop()
            else:
                stack.append(str_val[i])

        stack_len = len(stack)
        left = 0
        while stack and stack[-1] == '{':
            stack.pop()
            left += 1
        right = stack_len - left
        return math.ceil(right / 2) + math.ceil(left / 2)

    def Min_Reversals_Counter(self, s):
        """
        Counter approach - no extra space for stack
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        length = len(s)
        if length % 2:
            return -1

        left = right = 0
        for i in range(length):
            if s[i] == '{':
                left += 1
            else:
                if left == 0:
                    right += 1
                else:
                    left -= 1

        return math.ceil(right / 2) + math.ceil(left / 2)


def Test_Min_Bracket_Reversals():
    sol = Solution()
    tests = ["}{", "{{{{", "}{{}}{{{", "}}{{", "{{{", "{{}{{{}}{{"]

    for s in tests:
        print(f"Input: {s}")
        print(f"Stack: {sol.Min_Reversals_Stack(s)}")
        print(f"Counter: {sol.Min_Reversals_Counter(s)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Min_Bracket_Reversals()
