/*
Problem: Word Break Problem
URL: https://practice.geeksforgeeks.org/problems/word-break1352/1

Problem Statement:
Given a string and a dictionary of words, determine if the string can be segmented into space-separated dictionary words.

Sample Input/Output:
Input: s="leetcode", dict=["leet","code"]
Output: true
Input: s="catsandog", dict=["cats","dog","sand","and","cat"]
Output: false
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
    bool Word_Break_Trie_DP(string s, vector<string>& wordDict) {
        /*
        Word_Break_Trie_DP (build trie from dictionary, DP with trie lookup, O(n^2))
        Time Complexity: O(N*L + n^2) where N is dict size, L is avg word length, n is string length
        Space Complexity: O(N*L + n)
        */
        TrieNode* root = new TrieNode();
        for (string word : wordDict) {
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
        
        int n = s.length();
        vector<bool> dp(n + 1, false);
        dp[0] = true;
        
        for (int i = 0; i < n; i++) {
            if (!dp[i]) continue;
            
            TrieNode* node = root;
            for (int j = i; j < n; j++) {
                int index = s[j] - 'a';
                if (!node->children[index]) {
                    break;
                }
                node = node->children[index];
                if (node->isEndOfWord) {
                    dp[j + 1] = true;
                }
            }
        }
        
        return dp[n];
    }
    
    bool Word_Break_DP_Set(string s, vector<string>& wordDict) {
        /*
        Word_Break_DP_Set (DP with unordered_set lookup, O(n^2 * L))
        Time Complexity: O(n^2 * L) where n is string length, L is avg word length
        Space Complexity: O(N + n) where N is dict size
        */
        unordered_set<string> dict(wordDict.begin(), wordDict.end());
        int n = s.length();
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
    
    bool Word_Break_Recursive_Memo(string s, vector<string>& wordDict) {
        /*
        Word_Break_Recursive_Memo (recursive with memoization, O(n^2))
        Time Complexity: O(n^2) where n is string length
        Space Complexity: O(n + N) where N is dict size
        */
        unordered_set<string> dict(wordDict.begin(), wordDict.end());
        unordered_map<string, bool> memo;
        return WordBreakHelper(s, dict, memo);
    }
    
    bool WordBreakHelper(string s, unordered_set<string>& dict, unordered_map<string, bool>& memo) {
        if (s.empty()) return true;
        if (memo.find(s) != memo.end()) return memo[s];
        
        for (int i = 1; i <= s.length(); i++) {
            string prefix = s.substr(0, i);
            if (dict.find(prefix) != dict.end() && WordBreakHelper(s.substr(i), dict, memo)) {
                memo[s] = true;
                return true;
            }
        }
        
        memo[s] = false;
        return false;
    }
};

void Test_Word_Break() {
    Solution solution;
    
    cout << "=== Test Case 1 ===" << endl;
    string s1 = "leetcode";
    vector<string> dict1 = {"leet", "code"};
    cout << "String: " << s1 << endl;
    cout << "Dictionary: ";
    for (string w : dict1) cout << w << " ";
    cout << endl;
    cout << "Trie+DP: " << (solution.Word_Break_Trie_DP(s1, dict1) ? "true" : "false") << endl;
    cout << "DP+Set: " << (solution.Word_Break_DP_Set(s1, dict1) ? "true" : "false") << endl;
    cout << "Recursive+Memo: " << (solution.Word_Break_Recursive_Memo(s1, dict1) ? "true" : "false") << endl;
    
    cout << "\n=== Test Case 2 ===" << endl;
    string s2 = "catsandog";
    vector<string> dict2 = {"cats", "dog", "sand", "and", "cat"};
    cout << "String: " << s2 << endl;
    cout << "Dictionary: ";
    for (string w : dict2) cout << w << " ";
    cout << endl;
    cout << "Trie+DP: " << (solution.Word_Break_Trie_DP(s2, dict2) ? "true" : "false") << endl;
    cout << "DP+Set: " << (solution.Word_Break_DP_Set(s2, dict2) ? "true" : "false") << endl;
    cout << "Recursive+Memo: " << (solution.Word_Break_Recursive_Memo(s2, dict2) ? "true" : "false") << endl;
    
    cout << "\n=== Test Case 3 ===" << endl;
    string s3 = "applepenapple";
    vector<string> dict3 = {"apple", "pen"};
    cout << "String: " << s3 << endl;
    cout << "Dictionary: ";
    for (string w : dict3) cout << w << " ";
    cout << endl;
    cout << "Trie+DP: " << (solution.Word_Break_Trie_DP(s3, dict3) ? "true" : "false") << endl;
    cout << "DP+Set: " << (solution.Word_Break_DP_Set(s3, dict3) ? "true" : "false") << endl;
    cout << "Recursive+Memo: " << (solution.Word_Break_Recursive_Memo(s3, dict3) ? "true" : "false") << endl;
    
    cout << "\n=== Test Case 4 ===" << endl;
    string s4 = "aaaaaaa";
    vector<string> dict4 = {"aaaa", "aaa"};
    cout << "String: " << s4 << endl;
    cout << "Dictionary: ";
    for (string w : dict4) cout << w << " ";
    cout << endl;
    cout << "Trie+DP: " << (solution.Word_Break_Trie_DP(s4, dict4) ? "true" : "false") << endl;
    cout << "DP+Set: " << (solution.Word_Break_DP_Set(s4, dict4) ? "true" : "false") << endl;
    cout << "Recursive+Memo: " << (solution.Word_Break_Recursive_Memo(s4, dict4) ? "true" : "false") << endl;
    
    cout << "\n=== Test Case 5 ===" << endl;
    string s5 = "abcd";
    vector<string> dict5 = {"a", "abc", "b", "cd"};
    cout << "String: " << s5 << endl;
    cout << "Dictionary: ";
    for (string w : dict5) cout << w << " ";
    cout << endl;
    cout << "Trie+DP: " << (solution.Word_Break_Trie_DP(s5, dict5) ? "true" : "false") << endl;
    cout << "DP+Set: " << (solution.Word_Break_DP_Set(s5, dict5) ? "true" : "false") << endl;
    cout << "Recursive+Memo: " << (solution.Word_Break_Recursive_Memo(s5, dict5) ? "true" : "false") << endl;
}

int main() {
    Test_Word_Break();
    return 0;
}
