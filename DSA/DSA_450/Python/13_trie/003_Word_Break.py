"""
Problem: Word Break Problem
URL: https://practice.geeksforgeeks.org/problems/word-break1352/1

Problem Statement:
Given a string and a dictionary of words, determine if the string can be segmented into space-separated dictionary words.

Sample Input/Output:
Input: s="leetcode", dict=["leet","code"]
Output: true
Input: s="catsandog", dict=["cats","dog","sand","and","cat"]
Output: false
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Solution:
    def Word_Break_Trie_DP(self, s, wordDict):
        """
        Word_Break_Trie_DP (build trie from dictionary, DP with trie lookup, O(n^2))
        Time Complexity: O(N*L + n^2) where N is dict size, L is avg word length, n is string length
        Space Complexity: O(N*L + n)
        """
        root = TrieNode()
        for word in wordDict:
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
            node.is_end_of_word = True
        
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(n):
            if not dp[i]:
                continue
            
            node = root
            for j in range(i, n):
                if s[j] not in node.children:
                    break
                node = node.children[s[j]]
                if node.is_end_of_word:
                    dp[j + 1] = True
        
        return dp[n]
    
    def Word_Break_DP_Set(self, s, wordDict):
        """
        Word_Break_DP_Set (DP with set lookup, O(n^2 * L))
        Time Complexity: O(n^2 * L) where n is string length, L is avg word length
        Space Complexity: O(N + n) where N is dict size
        """
        word_set = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        
        return dp[n]
    
    def Word_Break_Recursive_Memo(self, s, wordDict):
        """
        Word_Break_Recursive_Memo (recursive with memoization, O(n^2))
        Time Complexity: O(n^2) where n is string length
        Space Complexity: O(n + N) where N is dict size
        """
        word_set = set(wordDict)
        memo = {}
        return self._WordBreakHelper(s, word_set, memo)
    
    def _WordBreakHelper(self, s, word_set, memo):
        if not s:
            return True
        if s in memo:
            return memo[s]
        
        for i in range(1, len(s) + 1):
            prefix = s[:i]
            if prefix in word_set and self._WordBreakHelper(s[i:], word_set, memo):
                memo[s] = True
                return True
        
        memo[s] = False
        return False


def Test_Word_Break():
    solution = Solution()
    
    print("=== Test Case 1 ===")
    s1 = "leetcode"
    dict1 = ["leet", "code"]
    print(f"String: {s1}")
    print(f"Dictionary: {' '.join(dict1)}")
    print(f"Trie+DP: {solution.Word_Break_Trie_DP(s1, dict1)}")
    print(f"DP+Set: {solution.Word_Break_DP_Set(s1, dict1)}")
    print(f"Recursive+Memo: {solution.Word_Break_Recursive_Memo(s1, dict1)}")
    
    print("\n=== Test Case 2 ===")
    s2 = "catsandog"
    dict2 = ["cats", "dog", "sand", "and", "cat"]
    print(f"String: {s2}")
    print(f"Dictionary: {' '.join(dict2)}")
    print(f"Trie+DP: {solution.Word_Break_Trie_DP(s2, dict2)}")
    print(f"DP+Set: {solution.Word_Break_DP_Set(s2, dict2)}")
    print(f"Recursive+Memo: {solution.Word_Break_Recursive_Memo(s2, dict2)}")
    
    print("\n=== Test Case 3 ===")
    s3 = "applepenapple"
    dict3 = ["apple", "pen"]
    print(f"String: {s3}")
    print(f"Dictionary: {' '.join(dict3)}")
    print(f"Trie+DP: {solution.Word_Break_Trie_DP(s3, dict3)}")
    print(f"DP+Set: {solution.Word_Break_DP_Set(s3, dict3)}")
    print(f"Recursive+Memo: {solution.Word_Break_Recursive_Memo(s3, dict3)}")
    
    print("\n=== Test Case 4 ===")
    s4 = "aaaaaaa"
    dict4 = ["aaaa", "aaa"]
    print(f"String: {s4}")
    print(f"Dictionary: {' '.join(dict4)}")
    print(f"Trie+DP: {solution.Word_Break_Trie_DP(s4, dict4)}")
    print(f"DP+Set: {solution.Word_Break_DP_Set(s4, dict4)}")
    print(f"Recursive+Memo: {solution.Word_Break_Recursive_Memo(s4, dict4)}")
    
    print("\n=== Test Case 5 ===")
    s5 = "abcd"
    dict5 = ["a", "abc", "b", "cd"]
    print(f"String: {s5}")
    print(f"Dictionary: {' '.join(dict5)}")
    print(f"Trie+DP: {solution.Word_Break_Trie_DP(s5, dict5)}")
    print(f"DP+Set: {solution.Word_Break_DP_Set(s5, dict5)}")
    print(f"Recursive+Memo: {solution.Word_Break_Recursive_Memo(s5, dict5)}")


if __name__ == "__main__":
    Test_Word_Break()
