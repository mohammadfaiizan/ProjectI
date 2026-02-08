"""
Problem: Print All Duplicates in a String
URL: https://www.geeksforgeeks.org/print-all-the-duplicates-in-the-input-string/

Problem Statement:
Given a string, find all characters that occur more than once and print them
along with their count.

Sample Input/Output:
Input: "geeksforgeeks"
Output: e, count = 4; g, count = 2; k, count = 2; s, count = 2
"""


class Solution:
    def Duplicate_Chars_Array(self, s):
        """
        Using frequency array of size 256
        Time Complexity: O(n)
        Space Complexity: O(1) - constant 256
        """
        freq = [0] * 256
        for c in s:
            freq[ord(c)] += 1
        for i in range(256):
            if freq[i] > 1:
                print(f"{chr(i)}, count = {freq[i]}")

    def Duplicate_Chars_Map(self, s):
        """
        Using dictionary
        Time Complexity: O(n)
        Space Complexity: O(k) where k = unique chars
        """
        mp = {}
        for c in s:
            mp[c] = mp.get(c, 0) + 1
        result = {}
        for char, count in mp.items():
            if count > 1:
                result[char] = count
        return result

    def Duplicate_Chars_Sorting(self, s):
        """
        Sort then scan adjacent
        Time Complexity: O(n log n)
        Space Complexity: O(1) if in-place sort
        """
        sorted_s = sorted(s)
        n = len(sorted_s)
        i = 0
        while i < n:
            count = 1
            while i + count < n and sorted_s[i] == sorted_s[i + count]:
                count += 1
            if count > 1:
                print(f"{sorted_s[i]}, count = {count}")
            i += count


def Test_Duplicate_Characters():
    sol = Solution()
    tests = ["geeksforgeeks", "hello", "aabbcc", "abcdef"]

    for s in tests:
        print(f"Input: {s}")

        print("Array Method:")
        sol.Duplicate_Chars_Array(s)

        print("Map Method:")
        res = sol.Duplicate_Chars_Map(s)
        for char, count in res.items():
            print(f"{char}, count = {count}")

        print("Sorting Method:")
        sol.Duplicate_Chars_Sorting(s)

        print('-' * 50)


if __name__ == "__main__":
    Test_Duplicate_Characters()
