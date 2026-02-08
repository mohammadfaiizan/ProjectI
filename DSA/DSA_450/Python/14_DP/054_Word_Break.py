"""
Problem: Word Break
URL: https://practice.geeksforgeeks.org/problems/word-break1352/1

Problem Statement:
Given a string s and a dictionary of words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

Sample Input/Output:
Input: s = "leetcode", dict = ["leet","code"]
Output: true
"""


class TrieNode:
    def __init__(self):
        self.is_end = False
        self.children = [None] * 26


class Solution:
    def Word_Break_DP(self, s: str, word_dict: list[str]) -> bool:
        """
        DP approach
        Time Complexity: O(n^2*L)
        Space Complexity: O(n)
        """
        n = len(s)
        dict_set = set(word_dict)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in dict_set:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def _insert(self, root: TrieNode, word: str) -> None:
        node = root
        for c in word:
            idx = ord(c) - ord('a')
            if not node.children[idx]:
                node.children[idx] = TrieNode()
            node = node.children[idx]
        node.is_end = True
    
    def Word_Break_Trie(self, s: str, word_dict: list[str]) -> bool:
        """
        Trie-based approach
        Time Complexity: O(n^2)
        Space Complexity: O(n + m*L)
        """
        root = TrieNode()
        for word in word_dict:
            self._insert(root, word)
        
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(n):
            if not dp[i]:
                continue
            
            node = root
            for j in range(i, n):
                idx = ord(s[j]) - ord('a')
                if not node.children[idx]:
                    break
                
                node = node.children[idx]
                if node.is_end:
                    dp[j + 1] = True
        
        return dp[n]


def Test_WordBreak():
    solution = Solution()
    
    s = "leetcode"
    word_dict = ["leet", "code"]
    
    assert solution.Word_Break_DP(s, word_dict) == True
    assert solution.Word_Break_Trie(s, word_dict) == True


if __name__ == "__main__":
    Test_WordBreak()
