"""
Problem: Find Shortest Unique Prefix for Every Word in a Given List
URL: https://www.geeksforgeeks.org/find-all-shortest-unique-prefixes-to-represent-each-word-in-a-given-list/

Problem Statement:
Given a list of words, find the shortest prefix that uniquely identifies each word.
Build a trie, track frequency at each node. The prefix where frequency becomes 1 is the unique prefix.

Sample Input/Output:
Input: ["zebra","dog","duck","dove"]
Output: ["z","dog","du","dov"]
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.frequency = 0


class Solution:
    def Shortest_Prefix_Trie(self, words):
        """
        Shortest_Prefix_Trie (build trie with freq count, traverse for each word until freq=1, O(N*L))
        Time Complexity: O(N*L) where N is number of words, L is average length
        Space Complexity: O(N*L)
        """
        root = TrieNode()
        
        for word in words:
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
                node.frequency += 1
        
        result = []
        for word in words:
            node = root
            prefix = ""
            for c in word:
                node = node.children[c]
                prefix += c
                if node.frequency == 1:
                    break
            result.append(prefix)
        
        return result
    
    def Shortest_Prefix_Brute(self, words):
        """
        Brute force approach (compare each word with all others)
        Time Complexity: O(N^2 * L)
        Space Complexity: O(1)
        """
        result = []
        n = len(words)
        
        for i in range(n):
            min_len = len(words[i])
            for j in range(n):
                if i == j:
                    continue
                k = 0
                while k < len(words[i]) and k < len(words[j]) and words[i][k] == words[j][k]:
                    k += 1
                if k < len(words[i]):
                    min_len = min(min_len, k + 1)
            result.append(words[i][:min_len])
        
        return result


def Test_Shortest_Unique_Prefix():
    solution = Solution()
    
    print("=== Test Case 1 ===")
    words1 = ["zebra", "dog", "duck", "dove"]
    result1 = solution.Shortest_Prefix_Trie(words1)
    print(f"Input: {' '.join(words1)}")
    print(f"Output: {' '.join(result1)}")
    
    print("\n=== Test Case 2 ===")
    words2 = ["geeksgeeks", "geeksquiz", "geeksforgeeks"]
    result2 = solution.Shortest_Prefix_Trie(words2)
    print(f"Input: {' '.join(words2)}")
    print(f"Output: {' '.join(result2)}")
    
    print("\n=== Test Case 3 ===")
    words3 = ["apple", "app", "ape", "bat", "ball"]
    result3 = solution.Shortest_Prefix_Trie(words3)
    print(f"Input: {' '.join(words3)}")
    print(f"Output: {' '.join(result3)}")
    
    print("\n=== Test Case 4 (Single word) ===")
    words4 = ["hello"]
    result4 = solution.Shortest_Prefix_Trie(words4)
    print(f"Input: {' '.join(words4)}")
    print(f"Output: {' '.join(result4)}")
    
    print("\n=== Test Case 5 (All unique) ===")
    words5 = ["cat", "dog", "bird"]
    result5 = solution.Shortest_Prefix_Trie(words5)
    print(f"Input: {' '.join(words5)}")
    print(f"Output: {' '.join(result5)}")


if __name__ == "__main__":
    Test_Shortest_Unique_Prefix()
