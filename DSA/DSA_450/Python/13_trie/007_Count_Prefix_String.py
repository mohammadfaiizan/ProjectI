"""
Problem: Count Number of Strings with Given Prefix
URL: https://www.geeksforgeeks.org/counting-number-words-trie/

Problem Statement:
Given a set of strings and a prefix, count how many strings have that prefix.

Sample Input/Output:
Input: words=["apple","app","ape","bat","ball"], prefix="ap"
Output: 3
Input: words=["apple","app","ape","bat","ball"], prefix="ba"
Output: 2
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.prefix_count = 0
        self.is_end_of_word = False


class Solution:
    def Count_Prefix_Trie(self, words, prefix):
        """
        Count_Prefix_Trie (build trie with prefix count at each node, O(N*L) build + O(P) query)
        Time Complexity: O(N*L + P) where N is words, L is avg length, P is prefix length
        Space Complexity: O(N*L)
        """
        root = TrieNode()
        
        for word in words:
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
                node.prefix_count += 1
            node.is_end_of_word = True
        
        node = root
        for c in prefix:
            if c not in node.children:
                return 0
            node = node.children[c]
        
        return node.prefix_count
    
    def Count_Prefix_Brute(self, words, prefix):
        """
        Count_Prefix_Brute (iterate all strings, check prefix, O(N*P))
        Time Complexity: O(N*P) where N is words, P is prefix length
        Space Complexity: O(1)
        """
        count = 0
        for word in words:
            if len(word) >= len(prefix) and word[:len(prefix)] == prefix:
                count += 1
        return count
    
    def Count_Prefix_Binary_Search(self, words, prefix):
        """
        Count_Prefix_Binary_Search (sort words, use binary search, O(N log N + log N))
        Time Complexity: O(N log N + log N)
        Space Complexity: O(1)
        """
        words_sorted = sorted(words)
        n = len(words_sorted)
        
        if not prefix:
            return n
        
        next_prefix = prefix[:-1] + chr(ord(prefix[-1]) + 1) if prefix else ""
        
        left = 0
        right = n
        while left < right:
            mid = (left + right) // 2
            if words_sorted[mid] < prefix:
                left = mid + 1
            else:
                right = mid
        
        lower = left
        
        left = 0
        right = n
        while left < right:
            mid = (left + right) // 2
            if words_sorted[mid] < next_prefix:
                left = mid + 1
            else:
                right = mid
        
        upper = left
        
        return upper - lower
    
    def Count_Prefix_Trie_DFS(self, words, prefix):
        """
        Count_Prefix_Trie_DFS (build trie, traverse prefix, DFS to count words, O(N*L + P + W))
        Time Complexity: O(N*L + P + W) where W is words with prefix
        Space Complexity: O(N*L)
        """
        root = TrieNode()
        
        for word in words:
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
            node.is_end_of_word = True
        
        node = root
        for c in prefix:
            if c not in node.children:
                return 0
            node = node.children[c]
        
        return self._CountWordsFromNode(node)
    
    def _CountWordsFromNode(self, node):
        if not node:
            return 0
        count = 1 if node.is_end_of_word else 0
        for child in node.children.values():
            count += self._CountWordsFromNode(child)
        return count


def Test_Count_Prefix_String():
    solution = Solution()
    
    print("=== Test Case 1 ===")
    words1 = ["apple", "app", "ape", "bat", "ball"]
    prefix1 = "ap"
    print(f"Words: {' '.join(words1)}")
    print(f"Prefix: {prefix1}")
    print(f"Count (Trie): {solution.Count_Prefix_Trie(words1, prefix1)}")
    print(f"Count (Brute): {solution.Count_Prefix_Brute(words1, prefix1)}")
    print(f"Count (Binary Search): {solution.Count_Prefix_Binary_Search(words1, prefix1)}")
    print(f"Count (Trie DFS): {solution.Count_Prefix_Trie_DFS(words1, prefix1)}")
    
    print("\n=== Test Case 2 ===")
    prefix2 = "ba"
    print(f"Prefix: {prefix2}")
    print(f"Count (Trie): {solution.Count_Prefix_Trie(words1, prefix2)}")
    print(f"Count (Brute): {solution.Count_Prefix_Brute(words1, prefix2)}")
    print(f"Count (Binary Search): {solution.Count_Prefix_Binary_Search(words1, prefix2)}")
    print(f"Count (Trie DFS): {solution.Count_Prefix_Trie_DFS(words1, prefix2)}")
    
    print("\n=== Test Case 3 ===")
    words3 = ["geeks", "geeksforgeeks", "geeksquiz", "geek"]
    prefix3 = "geek"
    print(f"Words: {' '.join(words3)}")
    print(f"Prefix: {prefix3}")
    print(f"Count (Trie): {solution.Count_Prefix_Trie(words3, prefix3)}")
    print(f"Count (Brute): {solution.Count_Prefix_Brute(words3, prefix3)}")
    print(f"Count (Binary Search): {solution.Count_Prefix_Binary_Search(words3, prefix3)}")
    print(f"Count (Trie DFS): {solution.Count_Prefix_Trie_DFS(words3, prefix3)}")
    
    print("\n=== Test Case 4 (No match) ===")
    prefix4 = "xyz"
    print(f"Prefix: {prefix4}")
    print(f"Count (Trie): {solution.Count_Prefix_Trie(words1, prefix4)}")
    print(f"Count (Brute): {solution.Count_Prefix_Brute(words1, prefix4)}")
    print(f"Count (Binary Search): {solution.Count_Prefix_Binary_Search(words1, prefix4)}")
    print(f"Count (Trie DFS): {solution.Count_Prefix_Trie_DFS(words1, prefix4)}")
    
    print("\n=== Test Case 5 (Empty prefix) ===")
    prefix5 = ""
    print("Prefix: (empty)")
    print(f"Count (Trie): {solution.Count_Prefix_Trie(words1, prefix5)}")
    print(f"Count (Brute): {solution.Count_Prefix_Brute(words1, prefix5)}")
    print(f"Count (Binary Search): {solution.Count_Prefix_Binary_Search(words1, prefix5)}")
    print(f"Count (Trie DFS): {solution.Count_Prefix_Trie_DFS(words1, prefix5)}")
    
    print("\n=== Test Case 6 (Single word) ===")
    words6 = ["hello"]
    prefix6 = "he"
    print(f"Words: {' '.join(words6)}")
    print(f"Prefix: {prefix6}")
    print(f"Count (Trie): {solution.Count_Prefix_Trie(words6, prefix6)}")
    print(f"Count (Brute): {solution.Count_Prefix_Brute(words6, prefix6)}")
    print(f"Count (Binary Search): {solution.Count_Prefix_Binary_Search(words6, prefix6)}")
    print(f"Count (Trie DFS): {solution.Count_Prefix_Trie_DFS(words6, prefix6)}")
    
    print("\n=== Test Case 7 (Multiple prefixes) ===")
    words7 = ["a", "aa", "aaa", "aaaa"]
    prefixes7 = ["a", "aa", "aaa", "aaaa"]
    print(f"Words: {' '.join(words7)}")
    for p in prefixes7:
        print(f"Prefix '{p}': {solution.Count_Prefix_Trie(words7, p)}")


if __name__ == "__main__":
    Test_Count_Prefix_String()
