/*
Problem: Autocomplete using Trie
URL: https://www.geeksforgeeks.org/auto-complete-feature-using-trie/

Problem Statement:
Given a set of words and a prefix, return all words that start with that prefix (autocomplete suggestions).
Build a trie, navigate to prefix node, then DFS to collect all words from there.

Sample Input/Output:
Input: words=["hello","dog","hell","help","helps","helping"], prefix="hel"
Output: [hell,hello,help,helps,helping]
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

class Solution {
public:
    vector<string> Autocomplete_Trie(vector<string>& words, string prefix) {
        /*
        Autocomplete_Trie
        Time Complexity: O(N*L + R) where N is number of words, L is average length, R is number of results
        Space Complexity: O(N*L)
        */
        TrieNode* root = new TrieNode();
        
        for (string word : words) {
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
        
        TrieNode* node = root;
        for (char c : prefix) {
            int index = c - 'a';
            if (!node->children[index]) {
                return {};
            }
            node = node->children[index];
        }
        
        vector<string> result;
        string current = prefix;
        DFS_Collect(node, current, result);
        return result;
    }
    
private:
    void DFS_Collect(TrieNode* node, string& current, vector<string>& result) {
        if (node->isEndOfWord) {
            result.push_back(current);
        }
        
        for (int i = 0; i < 26; i++) {
            if (node->children[i]) {
                current.push_back('a' + i);
                DFS_Collect(node->children[i], current, result);
                current.pop_back();
            }
        }
    }
};

void Test_Autocomplete_Trie() {
    Solution solution;
    
    vector<string> words1 = {"hello", "dog", "hell", "help", "helps", "helping"};
    string prefix1 = "hel";
    vector<string> result1 = solution.Autocomplete_Trie(words1, prefix1);
    cout << "Words: [hello, dog, hell, help, helps, helping], Prefix: 'hel'" << endl;
    cout << "Results: ";
    for (string word : result1) {
        cout << word << " ";
    }
    cout << endl;
    
    vector<string> words2 = {"apple", "app", "application", "apply", "apt", "ape"};
    string prefix2 = "app";
    vector<string> result2 = solution.Autocomplete_Trie(words2, prefix2);
    cout << "\nWords: [apple, app, application, apply, apt, ape], Prefix: 'app'" << endl;
    cout << "Results: ";
    for (string word : result2) {
        cout << word << " ";
    }
    cout << endl;
    
    vector<string> words3 = {"cat", "car", "card", "care", "careful"};
    string prefix3 = "ca";
    vector<string> result3 = solution.Autocomplete_Trie(words3, prefix3);
    cout << "\nWords: [cat, car, card, care, careful], Prefix: 'ca'" << endl;
    cout << "Results: ";
    for (string word : result3) {
        cout << word << " ";
    }
    cout << endl;
    
    vector<string> words4 = {"test", "testing", "tested"};
    string prefix4 = "xyz";
    vector<string> result4 = solution.Autocomplete_Trie(words4, prefix4);
    cout << "\nWords: [test, testing, tested], Prefix: 'xyz'" << endl;
    cout << "Results: ";
    if (result4.empty()) {
        cout << "No matches";
    } else {
        for (string word : result4) {
            cout << word << " ";
        }
    }
    cout << endl;
}

int main() {
    Test_Autocomplete_Trie();
    return 0;
}
