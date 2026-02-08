"""
Problem: Check if Strings are Rotations of Each Other
URL: https://www.geeksforgeeks.org/a-program-to-check-if-strings-are-rotations-of-each-other/

Problem Statement:
Given two strings s1 and s2, check whether s2 is a rotation of s1.

Sample Input/Output:
Input: s1 = "AACD", s2 = "ACDA"
Output: YES

Input: s1 = "ABCD", s2 = "ACBD"
Output: NO
"""

from collections import deque


class Solution:
    def Check_Rotation_Concatenation(self, s1, s2):
        """
        Concatenate s1 with itself and search for s2
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if len(s1) != len(s2):
            return False
        temp = s1 + s1
        return s2 in temp

    def Check_Rotation_One_By_One(self, s1, s2):
        """
        Try all rotations one by one
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        if len(s1) != len(s2):
            return False
        n = len(s1)
        for i in range(n):
            rotated = s1[i:] + s1[:i]
            if rotated == s2:
                return True
        return False

    def Check_Rotation_Queue(self, s1, s2):
        """
        Using queue - dequeue from front and enqueue to back
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        if len(s1) != len(s2):
            return False
        q1 = deque(s1)
        q2 = deque(s2)
        n = len(s1)
        for _ in range(n):
            if q1 == q2:
                return True
            front = q1.popleft()
            q1.append(front)
        return False


def Test_Check_String_Rotation():
    sol = Solution()
    tests = [
        ("AACD", "ACDA"),
        ("ABCD", "ACBD"),
        ("abcde", "cdeab"),
        ("abc", "abc"),
        ("abc", "ab")
    ]

    for s1, s2 in tests:
        print(f"s1: {s1}, s2: {s2}")
        print(f"Concatenation: {'YES' if sol.Check_Rotation_Concatenation(s1, s2) else 'NO'}")
        print(f"One By One: {'YES' if sol.Check_Rotation_One_By_One(s1, s2) else 'NO'}")
        print(f"Queue: {'YES' if sol.Check_Rotation_Queue(s1, s2) else 'NO'}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Check_String_Rotation()
