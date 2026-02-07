/*
Problem: Word Break
URL: https://practice.geeksforgeeks.org/problems/word-break1352/1

Problem Statement:
Given a string A and a dictionary of n words B, find out if A can be segmented
into a space-separated sequence of one or more dictionary words.

Sample Input/Output:
Input: A = "ilike", B = ["i", "like", "sam", "sung"]
Output: 1

Input: A = "ilikesamsung", B = ["i", "like", "sam", "sung", "samsung"]
Output: 1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Word_Break_DP(string A, vector<string>& B) {
        /*
        Bottom-up DP
        Time Complexity: O(n^2 * m) where m = max word length
        Space Complexity: O(n)
        */
        unordered_set<string> dict(B.begin(), B.end());
        int n = A.size();
        vector<bool> dp(n + 1, false);
        dp[0] = true;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && dict.count(A.substr(j, i - j))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }

    bool Word_Break_Recursive(string& A, unordered_set<string>& dict, int start, vector<int>& memo) {
        /*
        Top-down memoization
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        if (start == (int)A.size()) return true;
        if (memo[start] != -1) return memo[start];

        for (int end = start + 1; end <= (int)A.size(); end++) {
            if (dict.count(A.substr(start, end - start)) && Word_Break_Recursive(A, dict, end, memo)) {
                return memo[start] = 1;
            }
        }
        return memo[start] = 0;
    }

    bool Word_Break_Trie(string A, vector<string>& B) {
        /*
        Using Trie for dictionary lookup
        Time Complexity: O(n^2)
        Space Complexity: O(sum of word lengths + n)
        */
        struct TrieNode {
            TrieNode* children[26] = {};
            bool isEnd = false;
        };

        TrieNode* root = new TrieNode();
        for (auto& word : B) {
            TrieNode* node = root;
            for (char c : word) {
                if (!node->children[c - 'a'])
                    node->children[c - 'a'] = new TrieNode();
                node = node->children[c - 'a'];
            }
            node->isEnd = true;
        }

        int n = A.size();
        vector<bool> dp(n + 1, false);
        dp[0] = true;

        for (int i = 0; i < n; i++) {
            if (!dp[i]) continue;
            TrieNode* node = root;
            for (int j = i; j < n; j++) {
                if (!node->children[A[j] - 'a']) break;
                node = node->children[A[j] - 'a'];
                if (node->isEnd) dp[j + 1] = true;
            }
        }
        return dp[n];
    }
};

void Test_Word_Break() {
    Solution sol;
    struct TestCase { string A; vector<string> B; };
    vector<TestCase> tests = {
        {"ilike", {"i", "like", "sam", "sung"}},
        {"ilikesamsung", {"i", "like", "sam", "sung", "samsung"}},
        {"catsandog", {"cats", "dog", "sand", "and", "cat"}},
        {"leetcode", {"leet", "code"}}
    };

    for (auto& t : tests) {
        cout << "String: " << t.A << endl;
        cout << "DP: " << sol.Word_Break_DP(t.A, t.B) << endl;

        unordered_set<string> dict(t.B.begin(), t.B.end());
        vector<int> memo(t.A.size(), -1);
        cout << "Recursive: " << sol.Word_Break_Recursive(t.A, dict, 0, memo) << endl;
        cout << "Trie: " << sol.Word_Break_Trie(t.A, t.B) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Word_Break();
    return 0;
}
