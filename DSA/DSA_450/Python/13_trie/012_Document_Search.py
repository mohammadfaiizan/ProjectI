"""
Problem: Search Words in a Document using Trie
URL: https://www.geeksforgeeks.org/trie-based-solution-for-searching-words-in-document/

Problem Statement:
Given a document (text string) and a list of words, find which words appear in the document.
Build trie from the word list, then scan the document to check for matches.

Sample Input/Output:
Input: document="the cat sat on the mat", words=["cat","mat","bat","sat"]
Output: found: cat, mat, sat
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Solution:
    def Document_Search_Trie(self, document, words):
        """
        Document_Search_Trie
        Time Complexity: O(N*L + D*L) where N is number of words, L is average length, D is document length
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
        
        found = set()
        n = len(document)
        
        for i in range(n):
            node = root
            current = ""
            
            for j in range(i, n):
                if document[j] == ' ':
                    break
                
                if document[j] not in node.children:
                    break
                
                current += document[j]
                node = node.children[document[j]]
                
                if node.is_end_of_word:
                    found.add(current)
        
        return list(found)
    
    def Document_Search_Set(self, document, words):
        """
        Document_Search_Set
        Time Complexity: O(N*L + D*L)
        Space Complexity: O(N*L)
        """
        word_set = set(words)
        found = set()
        
        for word in document.split():
            if word in word_set:
                found.add(word)
        
        return list(found)


def Test_Document_Search():
    solution = Solution()
    
    document1 = "the cat sat on the mat"
    words1 = ["cat", "mat", "bat", "sat"]
    print(f"Document: '{document1}'")
    print("Words to search: [cat, mat, bat, sat]")
    result1 = solution.Document_Search_Trie(document1, words1)
    print(f"Found (Trie): {' '.join(result1)}")
    result1_set = solution.Document_Search_Set(document1, words1)
    print(f"Found (Set): {' '.join(result1_set)}")
    
    document2 = "hello world hello everyone"
    words2 = ["hello", "world", "test", "everyone"]
    print(f"\nDocument: '{document2}'")
    print("Words to search: [hello, world, test, everyone]")
    result2 = solution.Document_Search_Trie(document2, words2)
    print(f"Found (Trie): {' '.join(result2)}")
    result2_set = solution.Document_Search_Set(document2, words2)
    print(f"Found (Set): {' '.join(result2_set)}")
    
    document3 = "the quick brown fox jumps over the lazy dog"
    words3 = ["fox", "dog", "cat", "the", "quick"]
    print(f"\nDocument: '{document3}'")
    print("Words to search: [fox, dog, cat, the, quick]")
    result3 = solution.Document_Search_Trie(document3, words3)
    print(f"Found (Trie): {' '.join(result3)}")
    result3_set = solution.Document_Search_Set(document3, words3)
    print(f"Found (Set): {' '.join(result3_set)}")


if __name__ == "__main__":
    Test_Document_Search()
