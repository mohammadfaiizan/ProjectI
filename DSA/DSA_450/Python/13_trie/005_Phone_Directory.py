"""
Problem: Implement Phone Directory using Trie
URL: https://practice.geeksforgeeks.org/problems/phone-directory4628/1

Problem Statement:
Given a list of contacts and a query string, for each prefix of the query, return all contacts that start with that prefix (autocomplete-style).

Sample Input/Output:
Input: contacts=["geeikistest","geeksforgeeks","geeksquiz"], query="geeq"
Output: suggestions for each prefix
"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Solution:
    def Phone_Directory_Trie(self, contacts, query):
        """
        Phone_Directory_Trie (build trie, for each prefix collect all words via DFS, O(N*L + Q*N*L))
        Time Complexity: O(N*L + Q*N*L) where N is contacts, L is avg length, Q is query length
        Space Complexity: O(N*L)
        """
        root = TrieNode()
        
        for contact in contacts:
            node = root
            for c in contact:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
            node.is_end_of_word = True
        
        result = []
        node = root
        prefix = ""
        
        for c in query:
            if c not in node.children:
                while len(result) < len(query):
                    result.append(["0"])
                break
            
            prefix += c
            node = node.children[c]
            suggestions = []
            self._CollectWords(node, prefix, suggestions)
            
            if not suggestions:
                suggestions.append("0")
            
            result.append(suggestions)
        
        return result
    
    def _CollectWords(self, node, prefix, result):
        if node.is_end_of_word:
            result.append(prefix)
        
        for char, child in sorted(node.children.items()):
            self._CollectWords(child, prefix + char, result)
    
    def Phone_Directory_Brute(self, contacts, query):
        """
        Brute force approach (filter contacts for each prefix)
        Time Complexity: O(Q * N * L) where Q is query length, N is contacts, L is avg length
        Space Complexity: O(N * L)
        """
        result = []
        
        for i in range(1, len(query) + 1):
            prefix = query[:i]
            suggestions = []
            
            for contact in contacts:
                if len(contact) >= len(prefix) and contact[:len(prefix)] == prefix:
                    suggestions.append(contact)
            
            suggestions.sort()
            
            if not suggestions:
                suggestions.append("0")
            
            result.append(suggestions)
        
        return result
    
    def Phone_Directory_Optimized(self, contacts, query):
        """
        Optimized with sorting and binary search
        Time Complexity: O(N log N + Q * log N * L)
        Space Complexity: O(N * L)
        """
        contacts_sorted = sorted(contacts)
        result = []
        
        for i in range(1, len(query) + 1):
            prefix = query[:i]
            suggestions = []
            
            for contact in contacts_sorted:
                if len(contact) >= len(prefix) and contact[:len(prefix)] == prefix:
                    suggestions.append(contact)
                elif suggestions and contact > prefix:
                    break
            
            if not suggestions:
                suggestions.append("0")
            
            result.append(suggestions)
        
        return result


def Test_Phone_Directory():
    solution = Solution()
    
    print("=== Test Case 1 ===")
    contacts1 = ["geeikistest", "geeksforgeeks", "geeksquiz"]
    query1 = "geeq"
    print(f"Contacts: {' '.join(contacts1)}")
    print(f"Query: {query1}")
    
    result1 = solution.Phone_Directory_Trie(contacts1, query1)
    print("Output:")
    for i in range(len(result1)):
        print(f"Prefix '{query1[:i+1]}': {' '.join(result1[i])}")
    
    print("\n=== Test Case 2 ===")
    contacts2 = ["g", "ge", "gee", "geek", "geeks", "geeksforgeeks"]
    query2 = "geeks"
    print(f"Contacts: {' '.join(contacts2)}")
    print(f"Query: {query2}")
    
    result2 = solution.Phone_Directory_Trie(contacts2, query2)
    print("Output:")
    for i in range(len(result2)):
        print(f"Prefix '{query2[:i+1]}': {' '.join(result2[i])}")
    
    print("\n=== Test Case 3 ===")
    contacts3 = ["apple", "app", "ape", "application", "apply"]
    query3 = "app"
    print(f"Contacts: {' '.join(contacts3)}")
    print(f"Query: {query3}")
    
    result3 = solution.Phone_Directory_Trie(contacts3, query3)
    print("Output:")
    for i in range(len(result3)):
        print(f"Prefix '{query3[:i+1]}': {' '.join(result3[i])}")
    
    print("\n=== Test Case 4 (No match) ===")
    contacts4 = ["cat", "dog", "bird"]
    query4 = "xyz"
    print(f"Contacts: {' '.join(contacts4)}")
    print(f"Query: {query4}")
    
    result4 = solution.Phone_Directory_Trie(contacts4, query4)
    print("Output:")
    for i in range(len(result4)):
        print(f"Prefix '{query4[:i+1]}': {' '.join(result4[i])}")
    
    print("\n=== Test Case 5 (Single contact) ===")
    contacts5 = ["hello"]
    query5 = "he"
    print(f"Contacts: {' '.join(contacts5)}")
    print(f"Query: {query5}")
    
    result5 = solution.Phone_Directory_Trie(contacts5, query5)
    print("Output:")
    for i in range(len(result5)):
        print(f"Prefix '{query5[:i+1]}': {' '.join(result5[i])}")


if __name__ == "__main__":
    Test_Phone_Directory()
