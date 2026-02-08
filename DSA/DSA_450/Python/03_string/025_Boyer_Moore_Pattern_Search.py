"""
Problem: Boyer Moore Algorithm for Pattern Searching
URL: https://www.geeksforgeeks.org/boyer-moore-algorithm-for-pattern-searching/

Problem Statement:
Given a text and a pattern, find all occurrences of the pattern in the text
using Boyer Moore's Bad Character Heuristic.

Sample Input/Output:
Input: txt = "ABAAABCD", pat = "ABC"
Output: Pattern found at shift 4
"""


class Solution:
    def Boyer_Moore_Bad_Char(self, txt, pat):
        """
        Boyer Moore Bad Character Heuristic
        Time Complexity: O(n/m) best, O(n*m) worst
        Space Complexity: O(256) = O(1)
        """
        m, n = len(pat), len(txt)
        result = []

        badchar = [-1] * 256
        for i in range(m):
            badchar[ord(pat[i])] = i

        s = 0
        while s <= n - m:
            j = m - 1
            while j >= 0 and pat[j] == txt[s + j]:
                j -= 1

            if j < 0:
                result.append(s)
                s += (m - badchar[ord(txt[s + m])]) if (s + m < n) else 1
            else:
                s += max(1, j - badchar[ord(txt[s + j])])

        return result

    def Boyer_Moore_Simplified(self, txt, pat):
        """
        Simplified Boyer Moore with only bad character rule
        Time Complexity: O(n*m) worst case
        Space Complexity: O(256)
        """
        n, m = len(txt), len(pat)
        result = []
        lastOccurrence = {}
        for i in range(m):
            lastOccurrence[pat[i]] = i

        i = 0
        while i <= n - m:
            j = m - 1
            while j >= 0 and pat[j] == txt[i + j]:
                j -= 1

            if j < 0:
                result.append(i)
                i += 1
            else:
                lo = lastOccurrence.get(txt[i + j], -1)
                i += max(1, j - lo)

        return result


def Test_Boyer_Moore():
    sol = Solution()
    tests = [
        ("ABAAABCD", "ABC"),
        ("AABAACAADAABAABA", "AABA"),
        ("GEEKS FOR GEEKS", "GEEK"),
        ("ABABABABAB", "ABAB")
    ]

    for txt, pat in tests:
        print(f'Text: "{txt}", Pattern: "{pat}"')

        r1 = sol.Boyer_Moore_Bad_Char(txt, pat)
        print(f"Bad Char: {' '.join(map(str, r1))}")

        r2 = sol.Boyer_Moore_Simplified(txt, pat)
        print(f"Simplified: {' '.join(map(str, r2))}")

        print('-' * 50)


if __name__ == "__main__":
    Test_Boyer_Moore()
