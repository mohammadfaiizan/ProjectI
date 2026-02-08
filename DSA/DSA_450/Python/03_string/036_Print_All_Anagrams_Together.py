"""
Problem: Print All Anagrams Together
URL: https://practice.geeksforgeeks.org/problems/print-anagrams-together/1

Problem Statement:
Given an array of strings, group all anagrams together.

Sample Input/Output:
Input: ["eat", "tea", "tan", "ate", "nat", "bat"]
Output: [["eat","tea","ate"], ["tan","nat"], ["bat"]]
"""


class Solution:
    def Anagrams_Sort_Key(self, words):
        """
        Use sorted string as key in hashmap
        Time Complexity: O(n * k log k) where k = max word length
        Space Complexity: O(n * k)
        """
        mp = {}
        for word in words:
            key = ''.join(sorted(word))
            if key not in mp:
                mp[key] = []
            mp[key].append(word)

        result = list(mp.values())
        return result

    def Anagrams_Count_Key(self, words):
        """
        Use character count as key (frequency string)
        Time Complexity: O(n * k) where k = max word length
        Space Complexity: O(n * k)
        """
        mp = {}
        for word in words:
            count = [0] * 26
            for c in word:
                count[ord(c) - ord('a')] += 1
            key = '#'.join(map(str, count))
            if key not in mp:
                mp[key] = []
            mp[key].append(word)

        result = list(mp.values())
        return result

    def Anagrams_Prime_Hash(self, words):
        """
        Map each char to a prime number, product as key
        Time Complexity: O(n * k)
        Space Complexity: O(n)
        """
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                  43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
        mp = {}
        for word in words:
            hash_val = 1
            for c in word:
                hash_val *= primes[ord(c) - ord('a')]
            if hash_val not in mp:
                mp[hash_val] = []
            mp[hash_val].append(word)

        result = list(mp.values())
        return result


def Test_Print_All_Anagrams():
    sol = Solution()
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]

    print(f"Input: {' '.join(words)}")

    r1 = sol.Anagrams_Sort_Key(words)
    print("Sort Key:")
    for group in r1:
        print(f"  {group}")

    r2 = sol.Anagrams_Count_Key(words)
    print("Count Key:")
    for group in r2:
        print(f"  {group}")

    r3 = sol.Anagrams_Prime_Hash(words)
    print("Prime Hash:")
    for group in r3:
        print(f"  {group}")


if __name__ == "__main__":
    Test_Print_All_Anagrams()
