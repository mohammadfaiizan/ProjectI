/*
Problem: Construct a Trie from Scratch
URL: https://www.geeksforgeeks.org/trie-insert-and-search/

Problem Statement:
Implement a Trie data structure with insert, search, startsWith (prefix search), delete, and countWordsWithPrefix operations.
Define a TrieNode struct with children array (26 lowercase letters), isEndOfWord flag, and a count for prefix tracking.

Sample Input/Output:
Input: insert("apple"), insert("app"), search("app"), startsWith("ap")
Output: search("app") -> true, startsWith("ap") -> true
*/

#include <bits/stdc++.h>
using namespace std;

struct TrieNode {
    TrieNode* children[26];
    bool isEndOfWord;
    int prefixCount;
    
    TrieNode() {
        for (int i = 0; i < 26; i++) {
            children[i] = nullptr;
        }
        isEndOfWord = false;
        prefixCount = 0;
    }
};

class Trie {
private:
    TrieNode* root;
    
public:
    Trie() {
        root = new TrieNode();
    }
    
    void Insert(string word) {
        TrieNode* node = root;
        for (char c : word) {
            int index = c - 'a';
            if (!node->children[index]) {
                node->children[index] = new TrieNode();
            }
            node = node->children[index];
            node->prefixCount++;
        }
        node->isEndOfWord = true;
    }
    
    bool Search(string word) {
        TrieNode* node = root;
        for (char c : word) {
            int index = c - 'a';
            if (!node->children[index]) {
                return false;
            }
            node = node->children[index];
        }
        return node->isEndOfWord;
    }
    
    bool StartsWith(string prefix) {
        TrieNode* node = root;
        for (char c : prefix) {
            int index = c - 'a';
            if (!node->children[index]) {
                return false;
            }
            node = node->children[index];
        }
        return true;
    }
    
    bool Delete(string word) {
        return DeleteHelper(root, word, 0);
    }
    
    bool DeleteHelper(TrieNode* node, string word, int index) {
        if (index == word.length()) {
            if (!node->isEndOfWord) {
                return false;
            }
            node->isEndOfWord = false;
            return node->prefixCount == 0;
        }
        
        int charIndex = word[index] - 'a';
        if (!node->children[charIndex]) {
            return false;
        }
        
        bool shouldDelete = DeleteHelper(node->children[charIndex], word, index + 1);
        
        if (shouldDelete) {
            delete node->children[charIndex];
            node->children[charIndex] = nullptr;
            node->prefixCount--;
            return node->prefixCount == 0 && !node->isEndOfWord;
        }
        
        node->prefixCount--;
        return false;
    }
    
    int CountWordsWithPrefix(string prefix) {
        TrieNode* node = root;
        for (char c : prefix) {
            int index = c - 'a';
            if (!node->children[index]) {
                return 0;
            }
            node = node->children[index];
        }
        return CountWords(node);
    }
    
    int CountWords(TrieNode* node) {
        if (!node) return 0;
        int count = node->isEndOfWord ? 1 : 0;
        for (int i = 0; i < 26; i++) {
            if (node->children[i]) {
                count += CountWords(node->children[i]);
            }
        }
        return count;
    }
};

class Solution {
public:
    void Test_Array_Based_Trie() {
        /*
        Array-based TrieNode (children[26])
        Time Complexity: O(L) for insert/search/delete, O(N*L) for countWords
        Space Complexity: O(N*L) where N is number of words, L is average length
        */
        Trie trie;
        
        cout << "=== Testing Array-Based Trie ===" << endl;
        
        trie.Insert("apple");
        trie.Insert("app");
        trie.Insert("ape");
        trie.Insert("bat");
        trie.Insert("ball");
        
        cout << "Search 'app': " << (trie.Search("app") ? "Found" : "Not Found") << endl;
        cout << "Search 'apple': " << (trie.Search("apple") ? "Found" : "Not Found") << endl;
        cout << "Search 'ap': " << (trie.Search("ap") ? "Found" : "Not Found") << endl;
        cout << "Search 'xyz': " << (trie.Search("xyz") ? "Found" : "Not Found") << endl;
        
        cout << "StartsWith 'ap': " << (trie.StartsWith("ap") ? "Yes" : "No") << endl;
        cout << "StartsWith 'ba': " << (trie.StartsWith("ba") ? "Yes" : "No") << endl;
        cout << "StartsWith 'xyz': " << (trie.StartsWith("xyz") ? "Yes" : "No") << endl;
        
        cout << "Count words with prefix 'ap': " << trie.CountWordsWithPrefix("ap") << endl;
        cout << "Count words with prefix 'ba': " << trie.CountWordsWithPrefix("ba") << endl;
        
        cout << "Delete 'app': " << (trie.Delete("app") ? "Deleted" : "Failed") << endl;
        cout << "Search 'app' after delete: " << (trie.Search("app") ? "Found" : "Not Found") << endl;
        cout << "Search 'apple' after delete: " << (trie.Search("apple") ? "Found" : "Not Found") << endl;
        cout << "Count words with prefix 'ap' after delete: " << trie.CountWordsWithPrefix("ap") << endl;
    }
    
    void Test_Map_Based_Trie() {
        /*
        Map-based TrieNode (unordered_map children)
        Time Complexity: O(L) for insert/search/delete
        Space Complexity: O(N*L)
        */
        cout << "\n=== Testing Map-Based Trie ===" << endl;
        
        struct MapTrieNode {
            unordered_map<char, MapTrieNode*> children;
            bool isEndOfWord;
            int prefixCount;
            
            MapTrieNode() : isEndOfWord(false), prefixCount(0) {}
        };
        
        class MapTrie {
        private:
            MapTrieNode* root;
            
        public:
            MapTrie() {
                root = new MapTrieNode();
            }
            
            void Insert(string word) {
                MapTrieNode* node = root;
                for (char c : word) {
                    if (node->children.find(c) == node->children.end()) {
                        node->children[c] = new MapTrieNode();
                    }
                    node = node->children[c];
                    node->prefixCount++;
                }
                node->isEndOfWord = true;
            }
            
            bool Search(string word) {
                MapTrieNode* node = root;
                for (char c : word) {
                    if (node->children.find(c) == node->children.end()) {
                        return false;
                    }
                    node = node->children[c];
                }
                return node->isEndOfWord;
            }
            
            bool StartsWith(string prefix) {
                MapTrieNode* node = root;
                for (char c : prefix) {
                    if (node->children.find(c) == node->children.end()) {
                        return false;
                    }
                    node = node->children[c];
                }
                return true;
            }
        };
        
        MapTrie mapTrie;
        mapTrie.Insert("test");
        mapTrie.Insert("testing");
        mapTrie.Insert("tested");
        
        cout << "Search 'test': " << (mapTrie.Search("test") ? "Found" : "Not Found") << endl;
        cout << "Search 'testing': " << (mapTrie.Search("testing") ? "Found" : "Not Found") << endl;
        cout << "StartsWith 'tes': " << (mapTrie.StartsWith("tes") ? "Yes" : "No") << endl;
    }
};

void Test_Construct_Trie() {
    Solution solution;
    solution.Test_Array_Based_Trie();
    solution.Test_Map_Based_Trie();
}

int main() {
    Test_Construct_Trie();
    return 0;
}
