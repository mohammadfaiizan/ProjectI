/*
Problem: Word Break
URL: https://practice.geeksforgeeks.org/problems/word-break1352/1

Problem Statement:
Given a string s and a dictionary of words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

Sample Input/Output:
Input: s = "leetcode", dict = ["leet","code"]
Output: true
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Word_Break_DP(string s, vector<string>& wordDict) {
        /*
        DP approach
        Time Complexity: O(n^2*L)
        Space Complexity: O(n)
        */
        int n = s.length();
        unordered_set<string> dict(wordDict.begin(), wordDict.end());
        vector<bool> dp(n + 1, false);
        dp[0] = true;
        
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && dict.find(s.substr(j, i - j)) != dict.end()) {
                    dp[i] = true;
                    break;
                }
            }
        }
        
        return dp[n];
    }
    
    struct TrieNode {
        bool isEnd;
        TrieNode* children[26];
        TrieNode() {
            isEnd = false;
            for (int i = 0; i < 26; i++) children[i] = nullptr;
        }
    };
    
    void Insert(TrieNode* root, string& word) {
        TrieNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx]) {
                node->children[idx] = new TrieNode();
            }
            node = node->children[idx];
        }
        node->isEnd = true;
    }
    
    bool Word_Break_Trie(string s, vector<string>& wordDict) {
        /*
        Trie-based approach
        Time Complexity: O(n^2)
        Space Complexity: O(n + m*L)
        */
        TrieNode* root = new TrieNode();
        for (string& word : wordDict) {
            Insert(root, word);
        }
        
        int n = s.length();
        vector<bool> dp(n + 1, false);
        dp[0] = true;
        
        for (int i = 0; i < n; i++) {
            if (!dp[i]) continue;
            
            TrieNode* node = root;
            for (int j = i; j < n; j++) {
                int idx = s[j] - 'a';
                if (!node->children[idx]) break;
                
                node = node->children[idx];
                if (node->isEnd) {
                    dp[j + 1] = true;
                }
            }
        }
        
        return dp[n];
    }
};

void Test_Word_Break() {
    Solution solution;
    
    string s = "leetcode";
    vector<string> dict = {"leet", "code"};
    
    cout << "DP: s=\"" << s << "\" -> " 
         << (solution.Word_Break_DP(s, dict) ? "true" : "false") << endl;
    cout << "Trie: s=\"" << s << "\" -> " 
         << (solution.Word_Break_Trie(s, dict) ? "true" : "false") << endl;
}

int main() {
    Test_Word_Break();
    return 0;
}
