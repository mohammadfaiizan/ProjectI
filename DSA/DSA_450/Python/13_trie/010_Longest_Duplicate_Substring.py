"""
Problem: Longest Duplicate Substring
URL: https://leetcode.com/problems/longest-duplicate-substring/

Problem Statement:
Given a string, find the longest substring that occurs at least twice.

Sample Input/Output:
Input: "banana"
Output: "ana"
Input: "abcd"
Output: ""
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0


class Solution:
    def Longest_Dup_Binary_Search_Rolling_Hash(self, s):
        """
        Longest_Dup_Binary_Search_Rolling_Hash
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        n = len(s)
        left, right = 0, n - 1
        result = ""
        
        while left <= right:
            mid = left + (right - left) // 2
            candidate = self._CheckLength(s, mid)
            
            if candidate:
                result = candidate
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def _CheckLength(self, s, length):
        if length == 0:
            return ""
        
        seen = set()
        base = 26
        mod = 10**9 + 7
        power = 1
        hash_val = 0
        
        for i in range(length):
            hash_val = (hash_val * base + (ord(s[i]) - ord('a'))) % mod
            if i > 0:
                power = (power * base) % mod
        
        seen.add(hash_val)
        
        for i in range(length, len(s)):
            hash_val = (hash_val - (ord(s[i - length]) - ord('a')) * power % mod + mod) % mod
            hash_val = (hash_val * base + (ord(s[i]) - ord('a'))) % mod
            
            if hash_val in seen:
                return s[i - length + 1:i + 1]
            seen.add(hash_val)
        
        return ""
    
    def Longest_Dup_Suffix_Trie(self, s):
        """
        Longest_Dup_Suffix_Trie
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        n = len(s)
        result = ""
        max_len = 0
        
        for i in range(n):
            root = TrieNode()
            for j in range(i, n):
                node = root
                for k in range(j, n):
                    char = s[k]
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                    node.count += 1
                    
                    if node.count >= 2 and k - j + 1 > max_len:
                        max_len = k - j + 1
                        result = s[j:k + 1]
        
        return result


def Test_Longest_Duplicate_Substring():
    solution = Solution()
    
    s1 = "banana"
    print(f"Input: '{s1}'")
    result1 = solution.Longest_Dup_Binary_Search_Rolling_Hash(s1)
    print(f"Output (Binary Search + Rolling Hash): '{result1}'")
    result1_trie = solution.Longest_Dup_Suffix_Trie(s1)
    print(f"Output (Suffix Trie): '{result1_trie}'")
    
    s2 = "abcd"
    print(f"\nInput: '{s2}'")
    result2 = solution.Longest_Dup_Binary_Search_Rolling_Hash(s2)
    print(f"Output (Binary Search + Rolling Hash): '{result2}'")
    result2_trie = solution.Longest_Dup_Suffix_Trie(s2)
    print(f"Output (Suffix Trie): '{result2_trie}'")
    
    s3 = "aab"
    print(f"\nInput: '{s3}'")
    result3 = solution.Longest_Dup_Binary_Search_Rolling_Hash(s3)
    print(f"Output (Binary Search + Rolling Hash): '{result3}'")
    result3_trie = solution.Longest_Dup_Suffix_Trie(s3)
    print(f"Output (Suffix Trie): '{result3_trie}'")
    
    s4 = "aaaa"
    print(f"\nInput: '{s4}'")
    result4 = solution.Longest_Dup_Binary_Search_Rolling_Hash(s4)
    print(f"Output (Binary Search + Rolling Hash): '{result4}'")
    result4_trie = solution.Longest_Dup_Suffix_Trie(s4)
    print(f"Output (Suffix Trie): '{result4_trie}'")


if __name__ == "__main__":
    Test_Longest_Duplicate_Substring()
