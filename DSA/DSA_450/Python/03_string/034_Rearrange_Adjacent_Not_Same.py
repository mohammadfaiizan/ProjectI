"""
Problem: Rearrange Characters Such That No Two Adjacent Are Same
URL: https://leetcode.com/problems/reorganize-string/

Problem Statement:
Given a string s, rearrange the characters so that no two adjacent characters
are the same. Return any valid rearrangement or empty string if not possible.

Sample Input/Output:
Input: "aab"
Output: "aba"

Input: "aaab"
Output: "" (not possible)
"""

import heapq


class Solution:
    def Rearrange_Max_Heap(self, s):
        """
        Using max heap - always place most frequent char first
        Time Complexity: O(n log k) where k = unique chars
        Space Complexity: O(k)
        """
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1

        pq = [(-count, char) for char, count in freq.items()]
        heapq.heapify(pq)

        result = ""
        prev = (0, '#')

        while pq:
            curr = heapq.heappop(pq)
            result += curr[1]
            curr = (curr[0] + 1, curr[1])

            if prev[0] < 0:
                heapq.heappush(pq, prev)
            prev = curr

        return result if len(result) == len(s) else ""

    def Rearrange_Fill_Even_Odd(self, s):
        """
        Count frequencies, fill even positions first then odd
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        n = len(s)
        freq = {}
        maxChar = s[0]
        maxFreq = 0

        for c in s:
            freq[c] = freq.get(c, 0) + 1
            if freq[c] > maxFreq:
                maxFreq = freq[c]
                maxChar = c

        if maxFreq > (n + 1) // 2:
            return ""

        result = [''] * n
        idx = 0

        while freq[maxChar] > 0:
            result[idx] = maxChar
            idx += 2
            freq[maxChar] -= 1

        for char, count in freq.items():
            while count > 0:
                if idx >= n:
                    idx = 1
                result[idx] = char
                idx += 2
                count -= 1

        return ''.join(result)

    def Rearrange_Sorting(self, s):
        """
        Sort by frequency then interleave
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        n = len(s)
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1

        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        if sorted_freq[0][1] > (n + 1) // 2:
            return ""

        result = [''] * n
        idx = 0
        for char, count in sorted_freq:
            for _ in range(count):
                if idx >= n:
                    idx = 1
                result[idx] = char
                idx += 2

        return ''.join(result)


def Test_Rearrange_Adjacent():
    sol = Solution()
    tests = ["aab", "aaab", "aabb", "aaabbc", "a", "abcdef"]

    for s in tests:
        print(f"Input: {s}")
        r1 = sol.Rearrange_Max_Heap(s)
        print(f"Max Heap: {'Not Possible' if not r1 else r1}")
        r2 = sol.Rearrange_Fill_Even_Odd(s)
        print(f"Fill Even/Odd: {'Not Possible' if not r2 else r2}")
        r3 = sol.Rearrange_Sorting(s)
        print(f"Sorting: {'Not Possible' if not r3 else r3}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Rearrange_Adjacent()
