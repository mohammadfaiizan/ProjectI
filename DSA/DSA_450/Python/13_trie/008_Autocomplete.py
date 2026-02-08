"""
Problem: Autocomplete using Trie
URL: https://www.geeksforgeeks.org/auto-complete-feature-using-trie/

Problem Statement:
Given a set of words and a prefix, return all words that start with that prefix (autocomplete suggestions).
Build a trie, navigate to prefix node, then DFS to collect all words from there.

Sample Input/Output:
Input: words=["hello","dog","hell","help","helps","helping"], prefix="hel"
Output: [hell,hello,help,helps,helping]
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Solution:
    def Autocomplete_Trie(self, words, prefix):
        """
        Autocomplete_Trie
        Time Complexity: O(N*L + R) where N is number of words, L is average length, R is number of results
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
                return []
            node = node.children[c]
        
        result = []
        current = prefix
        self._DFS_Collect(node, current, result)
        return result
    
    def _DFS_Collect(self, node, current, result):
        if node.is_end_of_word:
            result.append(current)
        
        for char in sorted(node.children.keys()):
            self._DFS_Collect(node.children[char], current + char, result)


def Test_Autocomplete_Trie():
    solution = Solution()
    
    words1 = ["hello", "dog", "hell", "help", "helps", "helping"]
    prefix1 = "hel"
    result1 = solution.Autocomplete_Trie(words1, prefix1)
    print("Words: [hello, dog, hell, help, helps, helping], Prefix: 'hel'")
    print(f"Results: {' '.join(result1)}")
    
    words2 = ["apple", "app", "application", "apply", "apt", "ape"]
    prefix2 = "app"
    result2 = solution.Autocomplete_Trie(words2, prefix2)
    print("\nWords: [apple, app, application, apply, apt, ape], Prefix: 'app'")
    print(f"Results: {' '.join(result2)}")
    
    words3 = ["cat", "car", "card", "care", "careful"]
    prefix3 = "ca"
    result3 = solution.Autocomplete_Trie(words3, prefix3)
    print("\nWords: [cat, car, card, care, careful], Prefix: 'ca'")
    print(f"Results: {' '.join(result3)}")
    
    words4 = ["test", "testing", "tested"]
    prefix4 = "xyz"
    result4 = solution.Autocomplete_Trie(words4, prefix4)
    print("\nWords: [test, testing, tested], Prefix: 'xyz'")
    if not result4:
        print("Results: No matches")
    else:
        print(f"Results: {' '.join(result4)}")


if __name__ == "__main__":
    Test_Autocomplete_Trie()
