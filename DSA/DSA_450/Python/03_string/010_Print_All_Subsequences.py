"""
Problem: Print All Subsequences of a String
URL: https://www.geeksforgeeks.org/print-subsequences-string/

Problem Statement:
Given a string, print all possible subsequences of the string.
A subsequence is a sequence derived from another sequence by deleting some or no
elements without changing the order of the remaining elements.

Sample Input/Output:
Input: "abc"
Output: "", "a", "b", "c", "ab", "ac", "bc", "abc"
"""


class Solution:
    def Subsequences_Include_Exclude(self, s, curr, i, result):
        """
        Include/Exclude recursion
        Time Complexity: O(2^n)
        Space Complexity: O(n) recursion depth
        """
        if i == len(s):
            result.append(curr)
            return
        self.Subsequences_Include_Exclude(s, curr, i + 1, result)
        self.Subsequences_Include_Exclude(s, curr + s[i], i + 1, result)

    def Subsequences_Backtracking(self, s, n, idx, curr, result):
        """
        Backtracking - pick characters from index onwards
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        if curr:
            result.append(curr)
        for i in range(idx, n):
            curr += s[i]
            self.Subsequences_Backtracking(s, n, i + 1, curr, result)
            curr = curr[:-1]

    def Subsequences_Bitmask(self, s):
        """
        Bitmask - iterate over all 2^n subsets
        Time Complexity: O(n * 2^n)
        Space Complexity: O(2^n)
        """
        n = len(s)
        total = 1 << n
        result = []
        for mask in range(total):
            sub = ""
            for j in range(n):
                if mask & (1 << j):
                    sub += s[j]
            result.append(sub)
        return result


def Test_Print_All_Subsequences():
    sol = Solution()
    tests = ["abc", "ab", "a"]

    for s in tests:
        print(f"Input: {s}")

        r1 = []
        sol.Subsequences_Include_Exclude(s, "", 0, r1)
        print(f"Include/Exclude ({len(r1)}): ", end="")
        for x in r1:
            print(f'"{x}"', end=" ")
        print()

        r2 = []
        sol.Subsequences_Backtracking(s, len(s), 0, "", r2)
        print(f"Backtracking ({len(r2)}): ", end="")
        for x in r2:
            print(f'"{x}"', end=" ")
        print()

        r3 = sol.Subsequences_Bitmask(s)
        print(f"Bitmask ({len(r3)}): ", end="")
        for x in r3:
            print(f'"{x}"', end=" ")
        print()

        print('-' * 50)


if __name__ == "__main__":
    Test_Print_All_Subsequences()
