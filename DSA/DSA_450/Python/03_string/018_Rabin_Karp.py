"""
Problem: Rabin-Karp Algorithm for Pattern Searching
URL: https://www.geeksforgeeks.org/rabin-karp-algorithm-for-pattern-searching/

Problem Statement:
Given a text string and a pattern string, find all occurrences of the pattern
in the text using Rabin-Karp algorithm with rolling hash.

Sample Input/Output:
Input: text = "GEEKS FOR GEEKS", pattern = "GEEK"
Output: Pattern found at index 0, Pattern found at index 10
"""


class Solution:
    def Rabin_Karp_Search(self, txt, pat, q=101):
        """
        Rabin-Karp with rolling hash
        Time Complexity: O(n + m) average, O(n * m) worst case
        Space Complexity: O(1)
        """
        result = []
        d = 256
        M = len(pat)
        N = len(txt)
        p = t = 0
        h = 1

        for i in range(M - 1):
            h = (h * d) % q

        for i in range(M):
            p = (d * p + ord(pat[i])) % q
            t = (d * t + ord(txt[i])) % q

        for i in range(N - M + 1):
            if p == t:
                match = True
                for j in range(M):
                    if txt[i + j] != pat[j]:
                        match = False
                        break
                if match:
                    result.append(i)

            if i < N - M:
                t = (d * (t - ord(txt[i]) * h) + ord(txt[i + M])) % q
                if t < 0:
                    t += q

        return result

    def Naive_Search(self, txt, pat):
        """
        Naive pattern matching
        Time Complexity: O(n * m)
        Space Complexity: O(1)
        """
        result = []
        N, M = len(txt), len(pat)
        for i in range(N - M + 1):
            j = 0
            while j < M:
                if txt[i + j] != pat[j]:
                    break
                j += 1
            if j == M:
                result.append(i)
        return result

    def Rabin_Karp_Multiple_Hash(self, txt, pat):
        """
        Double hashing to reduce spurious hits
        Time Complexity: O(n + m) average
        Space Complexity: O(1)
        """
        result = []
        q1, q2, d = 101, 103, 256
        M, N = len(pat), len(txt)
        if M > N:
            return result

        p1 = t1 = h1 = 0
        p2 = t2 = h2 = 0
        h1_val = h2_val = 1

        for i in range(M - 1):
            h1_val = (h1_val * d) % q1
            h2_val = (h2_val * d) % q2

        for i in range(M):
            p1 = (d * p1 + ord(pat[i])) % q1
            t1 = (d * t1 + ord(txt[i])) % q1
            p2 = (d * p2 + ord(pat[i])) % q2
            t2 = (d * t2 + ord(txt[i])) % q2

        for i in range(N - M + 1):
            if p1 == t1 and p2 == t2:
                result.append(i)
            if i < N - M:
                t1 = (d * (t1 - ord(txt[i]) * h1_val) + ord(txt[i + M])) % q1
                if t1 < 0:
                    t1 += q1
                t2 = (d * (t2 - ord(txt[i]) * h2_val) + ord(txt[i + M])) % q2
                if t2 < 0:
                    t2 += q2

        return result


def Test_Rabin_Karp():
    sol = Solution()
    tests = [
        ("GEEKS FOR GEEKS", "GEEK"),
        ("AABAACAADAABAABA", "AABA"),
        ("ABABABAB", "ABA"),
        ("hello world", "world")
    ]

    for txt, pat in tests:
        print(f'Text: "{txt}", Pattern: "{pat}"')

        r1 = sol.Rabin_Karp_Search(txt, pat)
        print(f"Rabin-Karp: {' '.join(map(str, r1))}")

        r2 = sol.Naive_Search(txt, pat)
        print(f"Naive: {' '.join(map(str, r2))}")

        r3 = sol.Rabin_Karp_Multiple_Hash(txt, pat)
        print(f"Double Hash: {' '.join(map(str, r3))}")

        print('-' * 50)


if __name__ == "__main__":
    Test_Rabin_Karp()
