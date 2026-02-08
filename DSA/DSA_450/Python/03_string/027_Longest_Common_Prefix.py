"""
Problem: Longest Common Prefix
URL: https://leetcode.com/problems/longest-common-prefix/

Problem Statement:
Write a function to find the longest common prefix string amongst an array of strings.
If there is no common prefix, return an empty string.

Sample Input/Output:
Input: strs = ["flower","flow","flight"]
Output: "fl"

Input: strs = ["dog","racecar","car"]
Output: ""
"""


class Solution:
    def LCP_Horizontal_Scan(self, strs):
        """
        Compare prefix with each string and shrink
        Time Complexity: O(S) where S = sum of all chars
        Space Complexity: O(1)
        """
        if not strs:
            return ""
        ans = strs[0]
        for i in range(1, len(strs)):
            j = 0
            while j < min(len(ans), len(strs[i])) and ans[j] == strs[i][j]:
                j += 1
            ans = ans[:j]
            if not ans:
                return ""
        return ans

    def LCP_Vertical_Scan(self, strs):
        """
        Compare characters column by column
        Time Complexity: O(S)
        Space Complexity: O(1)
        """
        if not strs:
            return ""
        for i in range(len(strs[0])):
            c = strs[0][i]
            for j in range(1, len(strs)):
                if i >= len(strs[j]) or strs[j][i] != c:
                    return strs[0][:i]
        return strs[0]

    def LCP_Sorting(self, strs):
        """
        Sort and compare only first and last strings
        Time Complexity: O(n * m * log n) for sorting
        Space Complexity: O(1)
        """
        if not strs:
            return ""
        sorted_strs = sorted(strs)
        first = sorted_strs[0]
        last = sorted_strs[-1]
        i = 0
        while i < min(len(first), len(last)) and first[i] == last[i]:
            i += 1
        return first[:i]


def Test_Longest_Common_Prefix():
    sol = Solution()
    tests = [
        ["flower", "flow", "flight"],
        ["dog", "racecar", "car"],
        ["interspecies", "interstellar", "interstate"],
        ["a"],
        ["", "abc"]
    ]

    for strs in tests:
        print(f"Input: ", end="")
        for s in strs:
            print(f'"{s}"', end=" ")
        print()

        copy1, copy2, copy3 = strs[:], strs[:], strs[:]
        print(f'Horizontal: "{sol.LCP_Horizontal_Scan(copy1)}"')
        print(f'Vertical: "{sol.LCP_Vertical_Scan(copy2)}"')
        print(f'Sorting: "{sol.LCP_Sorting(copy3)}"')
        print('-' * 50)


if __name__ == "__main__":
    Test_Longest_Common_Prefix()
