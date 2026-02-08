"""
Problem: Reorganize String
URL: https://leetcode.com/problems/reorganize-string/

Problem Statement:
Given a string, rearrange it so no two adjacent characters are the same. Return empty string if impossible.

Sample Input/Output:
Input: "aab"
Output: "aba"
"""

import heapq
from collections import Counter


class Solution:
    def Reorganize_String_Max_Heap(self, s):
        """
        Greedy with Max Heap
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        freq = Counter(s)
        max_heap = []
        for char, count in freq.items():
            heapq.heappush(max_heap, (-count, char))

        result = ""
        prev = (-1, '#')

        while max_heap or prev[0] < 0:
            if not max_heap and prev[0] < 0:
                return ""

            count, char = heapq.heappop(max_heap)
            result += char
            count += 1

            if prev[0] < 0:
                heapq.heappush(max_heap, prev)

            prev = (count, char)

        return result

    def Reorganize_String_Counting(self, s):
        """
        Counting and Place at Even Indices
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        freq = Counter(s)
        max_freq = 0
        max_char = 'a'

        for char, count in freq.items():
            if count > max_freq:
                max_freq = count
                max_char = char

        n = len(s)
        if max_freq > (n + 1) // 2:
            return ""

        result = [''] * n
        idx = 0

        while freq[max_char] > 0:
            result[idx] = max_char
            idx += 2
            freq[max_char] -= 1

        for char, count in freq.items():
            while count > 0:
                if idx >= n:
                    idx = 1
                result[idx] = char
                idx += 2
                count -= 1

        return ''.join(result)


def Test_Reorganize_String():
    solution = Solution()

    s1 = "aab"
    print("Input:", '"' + s1 + '"')
    res1 = solution.Reorganize_String_Max_Heap(s1)
    print("Max Heap Result:", '"' + res1 + '"')
    res2 = solution.Reorganize_String_Counting(s1)
    print("Counting Result:", '"' + res2 + '"')

    s2 = "aaab"
    print("\nInput:", '"' + s2 + '"')
    res3 = solution.Reorganize_String_Max_Heap(s2)
    print("Max Heap Result:", '"' + res3 + '"')
    res4 = solution.Reorganize_String_Counting(s2)
    print("Counting Result:", '"' + res4 + '"')

    s3 = "aabbcc"
    print("\nInput:", '"' + s3 + '"')
    res5 = solution.Reorganize_String_Max_Heap(s3)
    print("Max Heap Result:", '"' + res5 + '"')
    res6 = solution.Reorganize_String_Counting(s3)
    print("Counting Result:", '"' + res6 + '"')

    s4 = "vvvlo"
    print("\nInput:", '"' + s4 + '"')
    res7 = solution.Reorganize_String_Max_Heap(s4)
    print("Max Heap Result:", '"' + res7 + '"')
    res8 = solution.Reorganize_String_Counting(s4)
    print("Counting Result:", '"' + res8 + '"')

    s5 = "aaabb"
    print("\nInput:", '"' + s5 + '"')
    res9 = solution.Reorganize_String_Max_Heap(s5)
    print("Max Heap Result:", '"' + res9 + '"')
    res10 = solution.Reorganize_String_Counting(s5)
    print("Counting Result:", '"' + res10 + '"')


if __name__ == "__main__":
    Test_Reorganize_String()
