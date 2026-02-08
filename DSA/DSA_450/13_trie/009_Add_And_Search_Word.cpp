/*
Problem: Add and Search Word - Data Structure Design (with Wildcard)
URL: https://leetcode.com/problems/design-add-and-search-words-data-structure/

Problem Statement:
Design a data structure that supports adding words and searching with '.' wildcard (matches any single character).
Build a trie. For search, when encountering '.', try all 26 children recursively.

Sample Input/Output:
Input: add "bad","dad","mad"; search "pad"->false, "bad"->true, ".ad"->true, "b.."->true
Output: false, true, true, true
*/

#include <bits/stdc++.h>
using namespace std;

struct TrieNode {
    TrieNode* children[26];
    bool isEndOfWord;
    
    TrieNode() {
        for (int i = 0; i < 26; i++) {
            children[i] = nullptr;
        }
        isEndOfWord = false;
    }
};

class WordDictionary {
private:
    TrieNode* root;
    
public:
    WordDictionary() {
        root = new TrieNode();
    }
    
    void AddWord(string word) {
        TrieNode* node = root;
        for (char c : word) {
            int index = c - 'a';
            if (!node->children[index]) {
                node->children[index] = new TrieNode();
            }
            node = node->children[index];
        }
        node->isEndOfWord = true;
    }
    
    bool Search(string word) {
        return SearchHelper(root, word, 0);
    }
    
private:
    bool SearchHelper(TrieNode* node, string& word, int index) {
        if (index == word.length()) {
            return node->isEndOfWord;
        }
        
        if (word[index] == '.') {
            for (int i = 0; i < 26; i++) {
                if (node->children[i] && SearchHelper(node->children[i], word, index + 1)) {
                    return true;
                }
            }
            return false;
        } else {
            int charIndex = word[index] - 'a';
            if (!node->children[charIndex]) {
                return false;
            }
            return SearchHelper(node->children[charIndex], word, index + 1);
        }
    }
};

class Solution {
public:
    void Add_Search_Trie() {
        /*
        Add_Search_Trie
        Time Complexity: O(26^d * L) worst case for '.' heavy queries, O(L) for normal queries where d is number of wildcards
        Space Complexity: O(N*L) where N is number of words, L is average length
        */
        WordDictionary dict;
        
        dict.AddWord("bad");
        dict.AddWord("dad");
        dict.AddWord("mad");
        
        cout << "Added words: bad, dad, mad" << endl;
        cout << "Search 'pad': " << (dict.Search("pad") ? "true" : "false") << endl;
        cout << "Search 'bad': " << (dict.Search("bad") ? "true" : "false") << endl;
        cout << "Search '.ad': " << (dict.Search(".ad") ? "true" : "false") << endl;
        cout << "Search 'b..': " << (dict.Search("b..") ? "true" : "false") << endl;
        cout << "Search 'b.d': " << (dict.Search("b.d") ? "true" : "false") << endl;
        cout << "Search '...': " << (dict.Search("...") ? "true" : "false") << endl;
        cout << "Search 'xyz': " << (dict.Search("xyz") ? "true" : "false") << endl;
        
        dict.AddWord("test");
        dict.AddWord("text");
        dict.AddWord("tent");
        cout << "\nAdded words: test, text, tent" << endl;
        cout << "Search 'test': " << (dict.Search("test") ? "true" : "false") << endl;
        cout << "Search 'te.t': " << (dict.Search("te.t") ? "true" : "false") << endl;
        cout << "Search '.e.t': " << (dict.Search(".e.t") ? "true" : "false") << endl;
        cout << "Search 't..t': " << (dict.Search("t..t") ? "true" : "false") << endl;
    }
};

void Test_Add_And_Search_Word() {
    Solution solution;
    solution.Add_Search_Trie();
}

int main() {
    Test_Add_And_Search_Word();
    return 0;
}
