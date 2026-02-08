"""
Problem: Recursively Remove All Adjacent Duplicates / Consecutive Characters
URL: https://practice.geeksforgeeks.org/problems/consecutive-elements2306/1

Problem Statement:
Given a string, remove all consecutive duplicate characters and return the result.

Sample Input/Output:
Input: "aabb"
Output: "ab"

Input: "aabaa"
Output: "aba"
"""


class Solution:
    def Remove_Consecutive_Iterative(self, s):
        """
        Skip consecutive duplicates using iteration
        Time Complexity: O(n)
        Space Complexity: O(n) for result
        """
        ans = ""
        n = len(s)
        i = 0
        while i < n:
            ans += s[i]
            temp = s[i]
            while i < n and s[i] == temp:
                i += 1
        return ans

    def Remove_Consecutive_Stack(self, s):
        """
        Using stack to track unique consecutive chars
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        stack = []
        n = len(s)
        for i in range(n):
            if not stack or stack[-1] != s[i]:
                stack.append(s[i])
        return ''.join(stack)

    def Remove_Consecutive_Recursive(self, s, i):
        """
        Recursive approach
        Time Complexity: O(n)
        Space Complexity: O(n) recursion stack
        """
        if i >= len(s):
            return ""
        rest = self.Remove_Consecutive_Recursive(s, i + 1)
        if rest and rest[0] == s[i]:
            return rest
        return s[i] + rest

    def Remove_Consecutive_Two_Pointer(self, s):
        """
        In-place two pointer approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not s:
            return s
        s_list = list(s)
        j = 0
        for i in range(1, len(s_list)):
            if s_list[i] != s_list[j]:
                j += 1
                s_list[j] = s_list[i]
        return ''.join(s_list[:j + 1])


def Test_Remove_Consecutive():
    sol = Solution()
    tests = ["aabb", "aabaa", "geeksforgeeks", "aabccba", "a", "aaaa"]

    for s in tests:
        print(f"Input: {s}")
        print(f"Iterative: {sol.Remove_Consecutive_Iterative(s)}")
        print(f"Stack: {sol.Remove_Consecutive_Stack(s)}")
        print(f"Recursive: {sol.Remove_Consecutive_Recursive(s, 0)}")
        print(f"Two Pointer: {sol.Remove_Consecutive_Two_Pointer(s)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Remove_Consecutive()
