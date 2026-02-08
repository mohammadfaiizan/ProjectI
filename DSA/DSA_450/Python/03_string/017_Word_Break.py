"""
Problem: Word Break
URL: https://practice.geeksforgeeks.org/problems/word-break1352/1

Problem Statement:
Given a string A and a dictionary of n words B, find out if A can be segmented
into a space-separated sequence of one or more dictionary words.

Sample Input/Output:
Input: A = "ilike", B = ["i", "like", "sam", "sung"]
Output: 1

Input: A = "ilikesamsung", B = ["i", "like", "sam", "sung", "samsung"]
Output: 1
"""


class Solution:
    def Word_Break_DP(self, A, B):
        """
        Bottom-up DP
        Time Complexity: O(n^2 * m) where m = max word length
        Space Complexity: O(n)
        """
        dict_set = set(B)
        n = len(A)
        dp = [False] * (n + 1)
        dp[0] = True

        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and A[j:i] in dict_set:
                    dp[i] = True
                    break

        return 1 if dp[n] else 0

    def Word_Break_Recursive(self, A, dict_set, start, memo):
        """
        Top-down memoization
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        if start == len(A):
            return True
        if memo[start] != -1:
            return memo[start]

        for end in range(start + 1, len(A) + 1):
            if A[start:end] in dict_set and self.Word_Break_Recursive(A, dict_set, end, memo):
                memo[start] = 1
                return True

        memo[start] = 0
        return False

    def Word_Break_Trie(self, A, B):
        """
        Using Trie for dictionary lookup
        Time Complexity: O(n^2)
        Space Complexity: O(sum of word lengths + n)
        """
        class TrieNode:
            def __init__(self):
                self.children = [None] * 26
                self.isEnd = False

        root = TrieNode()
        for word in B:
            node = root
            for c in word:
                idx = ord(c) - ord('a')
                if not node.children[idx]:
                    node.children[idx] = TrieNode()
                node = node.children[idx]
            node.isEnd = True

        n = len(A)
        dp = [False] * (n + 1)
        dp[0] = True

        for i in range(n):
            if not dp[i]:
                continue
            node = root
            for j in range(i, n):
                idx = ord(A[j]) - ord('a')
                if not node.children[idx]:
                    break
                node = node.children[idx]
                if node.isEnd:
                    dp[j + 1] = True

        return 1 if dp[n] else 0


def Test_Word_Break():
    sol = Solution()
    tests = [
        ("ilike", ["i", "like", "sam", "sung"]),
        ("ilikesamsung", ["i", "like", "sam", "sung", "samsung"]),
        ("catsandog", ["cats", "dog", "sand", "and", "cat"]),
        ("leetcode", ["leet", "code"])
    ]

    for A, B in tests:
        print(f"String: {A}")
        print(f"DP: {sol.Word_Break_DP(A, B)}")

        dict_set = set(B)
        memo = [-1] * len(A)
        print(f"Recursive: {1 if sol.Word_Break_Recursive(A, dict_set, 0, memo) else 0}")
        print(f"Trie: {sol.Word_Break_Trie(A, B)}")
        print('-' * 50)


if __name__ == "__main__":
    Test_Word_Break()
