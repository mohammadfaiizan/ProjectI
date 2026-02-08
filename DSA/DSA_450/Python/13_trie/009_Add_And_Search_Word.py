"""
Problem: Add and Search Word - Data Structure Design (with Wildcard)
URL: https://leetcode.com/problems/design-add-and-search-words-data-structure/

Problem Statement:
Design a data structure that supports adding words and searching with '.' wildcard (matches any single character).
Build a trie. For search, when encountering '.', try all 26 children recursively.

Sample Input/Output:
Input: add "bad","dad","mad"; search "pad"->false, "bad"->true, ".ad"->true, "b.."->true
Output: false, true, true, true
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class WordDictionary:
    def __init__(self):
        self.root = TrieNode()
    
    def AddWord(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end_of_word = True
    
    def Search(self, word):
        return self._SearchHelper(self.root, word, 0)
    
    def _SearchHelper(self, node, word, index):
        if index == len(word):
            return node.is_end_of_word
        
        if word[index] == '.':
            for child in node.children.values():
                if self._SearchHelper(child, word, index + 1):
                    return True
            return False
        else:
            if word[index] not in node.children:
                return False
            return self._SearchHelper(node.children[word[index]], word, index + 1)


class Solution:
    def Add_Search_Trie(self):
        """
        Add_Search_Trie
        Time Complexity: O(26^d * L) worst case for '.' heavy queries, O(L) for normal queries where d is number of wildcards
        Space Complexity: O(N*L) where N is number of words, L is average length
        """
        dict_obj = WordDictionary()
        
        dict_obj.AddWord("bad")
        dict_obj.AddWord("dad")
        dict_obj.AddWord("mad")
        
        print("Added words: bad, dad, mad")
        print(f"Search 'pad': {dict_obj.Search('pad')}")
        print(f"Search 'bad': {dict_obj.Search('bad')}")
        print(f"Search '.ad': {dict_obj.Search('.ad')}")
        print(f"Search 'b..': {dict_obj.Search('b..')}")
        print(f"Search 'b.d': {dict_obj.Search('b.d')}")
        print(f"Search '...': {dict_obj.Search('...')}")
        print(f"Search 'xyz': {dict_obj.Search('xyz')}")
        
        dict_obj.AddWord("test")
        dict_obj.AddWord("text")
        dict_obj.AddWord("tent")
        print("\nAdded words: test, text, tent")
        print(f"Search 'test': {dict_obj.Search('test')}")
        print(f"Search 'te.t': {dict_obj.Search('te.t')}")
        print(f"Search '.e.t': {dict_obj.Search('.e.t')}")
        print(f"Search 't..t': {dict_obj.Search('t..t')}")


def Test_Add_And_Search_Word():
    solution = Solution()
    solution.Add_Search_Trie()


if __name__ == "__main__":
    Test_Add_And_Search_Word()
