"""
Problem: KMP Algorithm for Pattern Searching / Longest Prefix Suffix
URL: https://practice.geeksforgeeks.org/problems/longest-prefix-suffix2527/1
URL: https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/

Problem Statement:
1. Given a string, find the length of the longest proper prefix which is also a suffix.
2. Given a text and pattern, find all occurrences of pattern in text using KMP algorithm.

Sample Input/Output:
Input: s = "abab"
Output: LPS = 2 ("ab" is both prefix and suffix)

Input: txt = "ABABDABACDABABCABAB", pat = "ABABCABAB"
Output: Pattern found at index 9
"""


class Solution:
    def Compute_LPS_Array(self, pat):
        """
        Build LPS (Longest Proper Prefix which is also Suffix) array
        Time Complexity: O(m)
        Space Complexity: O(m)
        """
        m = len(pat)
        lps = [0] * m
        length = 0
        i = 1

        while i < m:
            if pat[i] == pat[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1

        return lps

    def KMP_Search(self, txt, pat):
        """
        KMP pattern searching using LPS array
        Time Complexity: O(n + m)
        Space Complexity: O(m)
        """
        result = []
        N, M = len(txt), len(pat)
        lps = self.Compute_LPS_Array(pat)

        i = j = 0
        while i < N:
            if pat[j] == txt[i]:
                i += 1
                j += 1

            if j == M:
                result.append(i - j)
                j = lps[j - 1]
            elif i < N and pat[j] != txt[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1

        return result

    def Longest_Prefix_Suffix(self, s):
        """
        Find length of longest proper prefix which is also suffix
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        lps = self.Compute_LPS_Array(s)
        return lps[len(s) - 1]


def Test_KMP_Algorithm():
    sol = Solution()

    lps_tests = ["abab", "aabaaab", "aaaa", "abcab", "abc"]
    print("=== Longest Prefix Suffix ===")
    for s in lps_tests:
        print(f"Input: {s} -> LPS: {sol.Longest_Prefix_Suffix(s)}")
    print('-' * 50)

    kmp_tests = [
        ("ABABDABACDABABCABAB", "ABABCABAB"),
        ("AABAACAADAABAABA", "AABA"),
        ("GEEKS FOR GEEKS", "GEEK"),
        ("AAAAAA", "AA")
    ]

    print("=== KMP Search ===")
    for txt, pat in kmp_tests:
        print(f'Text: "{txt}", Pattern: "{pat}"')
        result = sol.KMP_Search(txt, pat)
        print(f"Found at: {' '.join(map(str, result))}")
        print('-' * 50)


if __name__ == "__main__":
    Test_KMP_Algorithm()
