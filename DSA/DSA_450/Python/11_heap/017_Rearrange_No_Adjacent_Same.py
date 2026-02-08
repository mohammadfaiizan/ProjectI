"""
Problem: Rearrange Characters So No Two Adjacent Are Same
URL: https://www.geeksforgeeks.org/rearrange-characters-string-no-two-adjacent/

Problem Statement:
Given a string, rearrange it so no two adjacent characters are the same. Return the rearranged string or empty if impossible.

Sample Input/Output:
Input: "aaabb"
Output: "ababa"
"""

import heapq
from collections import Counter


class Solution:
    def Rearrange_Max_Heap(self, s):
        """
        Max Heap Approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        freq = Counter(s)
        pq = []
        for char, count in freq.items():
            heapq.heappush(pq, (-count, char))

        result = ""
        prev = (-1, '#')

        while pq or prev[0] < 0:
            if not pq and prev[0] < 0:
                return ""

            count, char = heapq.heappop(pq)
            result += char
            count += 1

            if prev[0] < 0:
                heapq.heappush(pq, prev)

            prev = (count, char)

        return result

    def Rearrange_Fill_Even_Odd(self, s):
        """
        Fill Even-Odd Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        freq = Counter(s)
        max_freq = 0
        max_char = '#'

        for char, count in freq.items():
            if count > max_freq:
                max_freq = count
                max_char = char

        n = len(s)
        if max_freq > (n + 1) // 2:
            return ""

        result = [''] * n
        idx = 0

        for i in range(max_freq):
            result[idx] = max_char
            idx += 2

        freq[max_char] = 0

        for char, count in freq.items():
            while count > 0:
                if idx >= n:
                    idx = 1
                result[idx] = char
                idx += 2
                count -= 1

        return ''.join(result)


def IsValid(s):
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            return False
    return True


def Test_Rearrange():
    solution = Solution()

    s1 = "aaabb"
    result1 = solution.Rearrange_Max_Heap(s1)
    print("Input:", s1, "-> Output:", result1, "(Valid:", IsValid(result1) + ")")

    s1b = "aaabb"
    result1b = solution.Rearrange_Fill_Even_Odd(s1b)
    print("Input:", s1b, "-> Output:", result1b, "(Valid:", IsValid(result1b) + ")")

    s2 = "aaab"
    result2 = solution.Rearrange_Max_Heap(s2)
    print("Input:", s2, "-> Output:", "empty" if result2 == "" else result2)

    s2b = "aaab"
    result2b = solution.Rearrange_Fill_Even_Odd(s2b)
    print("Input:", s2b, "-> Output:", "empty" if result2b == "" else result2b)

    s3 = "aabb"
    result3 = solution.Rearrange_Max_Heap(s3)
    print("Input:", s3, "-> Output:", result3, "(Valid:", IsValid(result3) + ")")

    s3b = "aabb"
    result3b = solution.Rearrange_Fill_Even_Odd(s3b)
    print("Input:", s3b, "-> Output:", result3b, "(Valid:", IsValid(result3b) + ")")


if __name__ == "__main__":
    Test_Rearrange()
