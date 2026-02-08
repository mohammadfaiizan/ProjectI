"""
Problem: Permutations of a Given String
URL: https://practice.geeksforgeeks.org/problems/permutations-of-a-given-string2041/1

Problem Statement:
Given a string S, find all permutations of the string and return them sorted.

Sample Input/Output:
Input: S = "ABC"
Output: ABC ACB BAC BCA CAB CBA

Input: S = "AB"
Output: AB BA
"""

import itertools


class Solution:
    def Permutations_Swap(self, s, l, r, result):
        """
        Swap-based recursion
        Time Complexity: O(n! * n)
        Space Complexity: O(n) recursion stack
        """
        s_list = list(s)
        if l == r:
            result.append(''.join(s_list))
            return
        for i in range(l, r + 1):
            s_list[l], s_list[i] = s_list[i], s_list[l]
            self.Permutations_Swap(''.join(s_list), l + 1, r, result)
            s_list[l], s_list[i] = s_list[i], s_list[l]

    def Permutations_STL(self, s):
        """
        Using itertools.permutations
        Time Complexity: O(n! * n)
        Space Complexity: O(n!)
        """
        result = []
        sorted_s = sorted(s)
        for perm in itertools.permutations(sorted_s):
            result.append(''.join(perm))
        return result

    def Permutations_Backtrack(self, s, used, curr, result):
        """
        Backtracking with visited array
        Time Complexity: O(n! * n)
        Space Complexity: O(n)
        """
        if len(curr) == len(s):
            result.append(curr)
            return
        for i in range(len(s)):
            if used[i]:
                continue
            used[i] = True
            curr += s[i]
            self.Permutations_Backtrack(s, used, curr, result)
            curr = curr[:-1]
            used[i] = False


def Test_Print_All_Permutations():
    sol = Solution()
    tests = ["ABC", "AB", "A"]

    for s in tests:
        print(f"Input: {s}")

        temp = s
        r1 = []
        sol.Permutations_Swap(temp, 0, len(s) - 1, r1)
        r1.sort()
        print(f"Swap: {' '.join(r1)}")

        r2 = sol.Permutations_STL(s)
        print(f"STL: {' '.join(r2)}")

        used = [False] * len(s)
        curr = ""
        r3 = []
        sorted_s = ''.join(sorted(s))
        sol.Permutations_Backtrack(sorted_s, used, curr, r3)
        print(f"Backtrack: {' '.join(r3)}")

        print('-' * 50)


if __name__ == "__main__":
    Test_Print_All_Permutations()
