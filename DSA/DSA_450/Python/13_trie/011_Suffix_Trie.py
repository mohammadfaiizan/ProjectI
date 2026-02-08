"""
Problem: Suffix Trie Implementation
URL: https://www.geeksforgeeks.org/pattern-searching-using-suffix-tree/

Problem Statement:
Build a suffix trie for a given string and support pattern searching.
Insert all suffixes into a trie. Search checks if pattern exists as prefix of any suffix.

Sample Input/Output:
Input: text="banana", search "ana"->true, "ban"->true, "xyz"->false
Output: true, true, false
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class SuffixTrie:
    def __init__(self, text):
        self.root = TrieNode()
        self.Build(text)
    
    def Build(self, text):
        n = len(text)
        for i in range(n):
            self.InsertSuffix(text[i:])
    
    def InsertSuffix(self, suffix):
        node = self.root
        for c in suffix:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end_of_word = True
    
    def Search(self, pattern):
        node = self.root
        for c in pattern:
            if c not in node.children:
                return False
            node = node.children[c]
        return True


class Solution:
    def Suffix_Trie_Build_Search(self, text, pattern):
        """
        Suffix_Trie_Build_Search
        Time Complexity: O(n^2) build, O(m) search where n is text length, m is pattern length
        Space Complexity: O(n^2)
        """
        trie = SuffixTrie(text)
        return trie.Search(pattern)


def Test_Suffix_Trie():
    solution = Solution()
    
    text1 = "banana"
    print(f"Text: '{text1}'")
    print(f"Search 'ana': {solution.Suffix_Trie_Build_Search(text1, 'ana')}")
    print(f"Search 'ban': {solution.Suffix_Trie_Build_Search(text1, 'ban')}")
    print(f"Search 'xyz': {solution.Suffix_Trie_Build_Search(text1, 'xyz')}")
    print(f"Search 'nan': {solution.Suffix_Trie_Build_Search(text1, 'nan')}")
    print(f"Search 'a': {solution.Suffix_Trie_Build_Search(text1, 'a')}")
    
    text2 = "abababa"
    print(f"\nText: '{text2}'")
    print(f"Search 'aba': {solution.Suffix_Trie_Build_Search(text2, 'aba')}")
    print(f"Search 'bab': {solution.Suffix_Trie_Build_Search(text2, 'bab')}")
    print(f"Search 'abab': {solution.Suffix_Trie_Build_Search(text2, 'abab')}")
    
    text3 = "test"
    print(f"\nText: '{text3}'")
    print(f"Search 'test': {solution.Suffix_Trie_Build_Search(text3, 'test')}")
    print(f"Search 'est': {solution.Suffix_Trie_Build_Search(text3, 'est')}")
    print(f"Search 'xyz': {solution.Suffix_Trie_Build_Search(text3, 'xyz')}")


if __name__ == "__main__":
    Test_Suffix_Trie()
