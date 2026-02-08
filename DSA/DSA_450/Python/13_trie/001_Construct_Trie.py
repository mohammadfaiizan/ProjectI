"""
Problem: Construct a Trie from Scratch
URL: https://www.geeksforgeeks.org/trie-insert-and-search/

Problem Statement:
Implement a Trie data structure with insert, search, startsWith (prefix search), delete, and countWordsWithPrefix operations.
Define a TrieNode struct with children array (26 lowercase letters), isEndOfWord flag, and a count for prefix tracking.

Sample Input/Output:
Input: insert("apple"), insert("app"), search("app"), startsWith("ap")
Output: search("app") -> true, startsWith("ap") -> true
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.prefix_count = 0


class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def Insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.prefix_count += 1
        node.is_end_of_word = True
    
    def Search(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return node.is_end_of_word
    
    def StartsWith(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children[c]
        return True
    
    def Delete(self, word):
        return self._DeleteHelper(self.root, word, 0)
    
    def _DeleteHelper(self, node, word, index):
        if index == len(word):
            if not node.is_end_of_word:
                return False
            node.is_end_of_word = False
            return node.prefix_count == 0
        
        char = word[index]
        if char not in node.children:
            return False
        
        should_delete = self._DeleteHelper(node.children[char], word, index + 1)
        
        if should_delete:
            del node.children[char]
            node.prefix_count -= 1
            return node.prefix_count == 0 and not node.is_end_of_word
        
        node.prefix_count -= 1
        return False
    
    def CountWordsWithPrefix(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node.children:
                return 0
            node = node.children[c]
        return self._CountWords(node)
    
    def _CountWords(self, node):
        if not node:
            return 0
        count = 1 if node.is_end_of_word else 0
        for child in node.children.values():
            count += self._CountWords(child)
        return count


class Solution:
    def Test_Array_Based_Trie(self):
        """
        Array-based TrieNode (children dict)
        Time Complexity: O(L) for insert/search/delete, O(N*L) for countWords
        Space Complexity: O(N*L) where N is number of words, L is average length
        """
        trie = Trie()
        
        print("=== Testing Array-Based Trie ===")
        
        trie.Insert("apple")
        trie.Insert("app")
        trie.Insert("ape")
        trie.Insert("bat")
        trie.Insert("ball")
        
        print(f"Search 'app': {'Found' if trie.Search('app') else 'Not Found'}")
        print(f"Search 'apple': {'Found' if trie.Search('apple') else 'Not Found'}")
        print(f"Search 'ap': {'Found' if trie.Search('ap') else 'Not Found'}")
        print(f"Search 'xyz': {'Found' if trie.Search('xyz') else 'Not Found'}")
        
        print(f"StartsWith 'ap': {'Yes' if trie.StartsWith('ap') else 'No'}")
        print(f"StartsWith 'ba': {'Yes' if trie.StartsWith('ba') else 'No'}")
        print(f"StartsWith 'xyz': {'Yes' if trie.StartsWith('xyz') else 'No'}")
        
        print(f"Count words with prefix 'ap': {trie.CountWordsWithPrefix('ap')}")
        print(f"Count words with prefix 'ba': {trie.CountWordsWithPrefix('ba')}")
        
        print(f"Delete 'app': {'Deleted' if trie.Delete('app') else 'Failed'}")
        print(f"Search 'app' after delete: {'Found' if trie.Search('app') else 'Not Found'}")
        print(f"Search 'apple' after delete: {'Found' if trie.Search('apple') else 'Not Found'}")
        print(f"Count words with prefix 'ap' after delete: {trie.CountWordsWithPrefix('ap')}")
    
    def Test_Map_Based_Trie(self):
        """
        Map-based TrieNode (dict children)
        Time Complexity: O(L) for insert/search/delete
        Space Complexity: O(N*L)
        """
        print("\n=== Testing Map-Based Trie ===")
        
        class MapTrieNode:
            def __init__(self):
                self.children = {}
                self.is_end_of_word = False
                self.prefix_count = 0
        
        class MapTrie:
            def __init__(self):
                self.root = MapTrieNode()
            
            def Insert(self, word):
                node = self.root
                for c in word:
                    if c not in node.children:
                        node.children[c] = MapTrieNode()
                    node = node.children[c]
                    node.prefix_count += 1
                node.is_end_of_word = True
            
            def Search(self, word):
                node = self.root
                for c in word:
                    if c not in node.children:
                        return False
                    node = node.children[c]
                return node.is_end_of_word
            
            def StartsWith(self, prefix):
                node = self.root
                for c in prefix:
                    if c not in node.children:
                        return False
                    node = node.children[c]
                return True
        
        map_trie = MapTrie()
        map_trie.Insert("test")
        map_trie.Insert("testing")
        map_trie.Insert("tested")
        
        print(f"Search 'test': {'Found' if map_trie.Search('test') else 'Not Found'}")
        print(f"Search 'testing': {'Found' if map_trie.Search('testing') else 'Not Found'}")
        print(f"StartsWith 'tes': {'Yes' if map_trie.StartsWith('tes') else 'No'}")


def Test_Construct_Trie():
    solution = Solution()
    solution.Test_Array_Based_Trie()
    solution.Test_Map_Based_Trie()


if __name__ == "__main__":
    Test_Construct_Trie()
