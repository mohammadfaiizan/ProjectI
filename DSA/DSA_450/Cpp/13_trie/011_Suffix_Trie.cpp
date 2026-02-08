/*
Problem: Suffix Trie Implementation
URL: https://www.geeksforgeeks.org/pattern-searching-using-suffix-tree/

Problem Statement:
Build a suffix trie for a given string and support pattern searching.
Insert all suffixes into a trie. Search checks if pattern exists as prefix of any suffix.

Sample Input/Output:
Input: text="banana", search "ana"->true, "ban"->true, "xyz"->false
Output: true, true, false
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

class SuffixTrie {
private:
    TrieNode* root;
    
public:
    SuffixTrie(string text) {
        root = new TrieNode();
        Build(text);
    }
    
    void Build(string& text) {
        int n = text.length();
        for (int i = 0; i < n; i++) {
            InsertSuffix(text.substr(i));
        }
    }
    
    void InsertSuffix(string suffix) {
        TrieNode* node = root;
        for (char c : suffix) {
            int index = c - 'a';
            if (!node->children[index]) {
                node->children[index] = new TrieNode();
            }
            node = node->children[index];
        }
        node->isEndOfWord = true;
    }
    
    bool Search(string pattern) {
        TrieNode* node = root;
        for (char c : pattern) {
            int index = c - 'a';
            if (!node->children[index]) {
                return false;
            }
            node = node->children[index];
        }
        return true;
    }
};

class Solution {
public:
    bool Suffix_Trie_Build_Search(string text, string pattern) {
        /*
        Suffix_Trie_Build_Search
        Time Complexity: O(n^2) build, O(m) search where n is text length, m is pattern length
        Space Complexity: O(n^2)
        */
        SuffixTrie trie(text);
        return trie.Search(pattern);
    }
};

void Test_Suffix_Trie() {
    Solution solution;
    
    string text1 = "banana";
    cout << "Text: '" << text1 << "'" << endl;
    cout << "Search 'ana': " << (solution.Suffix_Trie_Build_Search(text1, "ana") ? "true" : "false") << endl;
    cout << "Search 'ban': " << (solution.Suffix_Trie_Build_Search(text1, "ban") ? "true" : "false") << endl;
    cout << "Search 'xyz': " << (solution.Suffix_Trie_Build_Search(text1, "xyz") ? "true" : "false") << endl;
    cout << "Search 'nan': " << (solution.Suffix_Trie_Build_Search(text1, "nan") ? "true" : "false") << endl;
    cout << "Search 'a': " << (solution.Suffix_Trie_Build_Search(text1, "a") ? "true" : "false") << endl;
    
    string text2 = "abababa";
    cout << "\nText: '" << text2 << "'" << endl;
    cout << "Search 'aba': " << (solution.Suffix_Trie_Build_Search(text2, "aba") ? "true" : "false") << endl;
    cout << "Search 'bab': " << (solution.Suffix_Trie_Build_Search(text2, "bab") ? "true" : "false") << endl;
    cout << "Search 'abab': " << (solution.Suffix_Trie_Build_Search(text2, "abab") ? "true" : "false") << endl;
    
    string text3 = "test";
    cout << "\nText: '" << text3 << "'" << endl;
    cout << "Search 'test': " << (solution.Suffix_Trie_Build_Search(text3, "test") ? "true" : "false") << endl;
    cout << "Search 'est': " << (solution.Suffix_Trie_Build_Search(text3, "est") ? "true" : "false") << endl;
    cout << "Search 'xyz': " << (solution.Suffix_Trie_Build_Search(text3, "xyz") ? "true" : "false") << endl;
}

int main() {
    Test_Suffix_Trie();
    return 0;
}
