"""
Problem: Transform One String to Another
URL: https://www.geeksforgeeks.org/transform-one-string-to-another-using-minimum-number-of-given-operation/

Problem Statement:
Given two strings A and B, find the minimum number of operations required to
transform A to B. The only allowed operation is to pick a character from A and
insert it at the front.

Sample Input/Output:
Input: A = "EACBD", B = "EABCD"
Output: 3

Input: A = "ABC", B = "BCA"
Output: 2
"""

from collections import deque


class Solution:
    def Min_Ops_Greedy(self, A, B):
        """
        Greedy - count mismatches from the end
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        m, n = len(A), len(B)
        if m != n:
            return -1

        count = [0] * 256
        for i in range(n):
            count[ord(B[i])] += 1
        for i in range(n):
            count[ord(A[i])] -= 1
        for i in range(256):
            if count[i]:
                return -1

        res = 0
        i = j = n - 1
        while i >= 0:
            while i >= 0 and A[i] != B[j]:
                i -= 1
                res += 1
            if i >= 0:
                i -= 1
                j -= 1

        return res

    def Min_Ops_Simulation(self, A, B):
        """
        Simulate the process using deque
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        n = len(A)
        if n != len(B):
            return -1

        countA = [0] * 256
        countB = [0] * 256
        for c in A:
            countA[ord(c)] += 1
        for c in B:
            countB[ord(c)] += 1
        for i in range(256):
            if countA[i] != countB[i]:
                return -1

        dq = deque(A)
        ops = 0
        j = n - 1

        while j >= 0:
            if dq[-1] == B[j]:
                dq.pop()
                j -= 1
            else:
                back = dq.pop()
                dq.appendleft(back)
                ops += 1

        return ops


def Test_Transform_String():
    sol = Solution()
    tests = [
        ("EACBD", "EABCD"),
        ("ABC", "BCA"),
        ("ABCD", "ABCD"),
        ("ABC", "DEF")
    ]

    for A, B in tests:
        print(f"A: {A}, B: {B}")
        print(f"Greedy: {sol.Min_Ops_Greedy(A, B)}")
        print(f"Simulation: {sol.Min_Ops_Simulation(A, B)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Transform_String()
