/*
Problem: Longest Duplicate Substring
URL: https://leetcode.com/problems/longest-duplicate-substring/

Problem Statement:
Given a string, find the longest substring that occurs at least twice.

Sample Input/Output:
Input: "banana"
Output: "ana"
Input: "abcd"
Output: ""
*/

#include <bits/stdc++.h>
using namespace std;

struct TrieNode {
    TrieNode* children[26];
    int count;
    
    TrieNode() {
        for (int i = 0; i < 26; i++) {
            children[i] = nullptr;
        }
        count = 0;
    }
};

class Solution {
public:
    string Longest_Dup_Binary_Search_Rolling_Hash(string s) {
        /*
        Longest_Dup_Binary_Search_Rolling_Hash
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        int n = s.length();
        int left = 0, right = n - 1;
        string result = "";
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            string candidate = CheckLength(s, mid);
            
            if (!candidate.empty()) {
                result = candidate;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return result;
    }
    
    string CheckLength(string& s, int len) {
        if (len == 0) return "";
        
        unordered_set<long long> seen;
        long long base = 26;
        long long mod = 1e9 + 7;
        long long power = 1;
        long long hash = 0;
        
        for (int i = 0; i < len; i++) {
            hash = (hash * base + (s[i] - 'a')) % mod;
            if (i > 0) power = (power * base) % mod;
        }
        seen.insert(hash);
        
        for (int i = len; i < s.length(); i++) {
            hash = (hash - (s[i - len] - 'a') * power % mod + mod) % mod;
            hash = (hash * base + (s[i] - 'a')) % mod;
            
            if (seen.find(hash) != seen.end()) {
                return s.substr(i - len + 1, len);
            }
            seen.insert(hash);
        }
        
        return "";
    }
    
    string Longest_Dup_Suffix_Trie(string s) {
        /*
        Longest_Dup_Suffix_Trie
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        int n = s.length();
        string result = "";
        int maxLen = 0;
        
        for (int i = 0; i < n; i++) {
            TrieNode* root = new TrieNode();
            for (int j = i; j < n; j++) {
                TrieNode* node = root;
                for (int k = j; k < n; k++) {
                    int index = s[k] - 'a';
                    if (!node->children[index]) {
                        node->children[index] = new TrieNode();
                    }
                    node = node->children[index];
                    node->count++;
                    
                    if (node->count >= 2 && k - j + 1 > maxLen) {
                        maxLen = k - j + 1;
                        result = s.substr(j, maxLen);
                    }
                }
            }
        }
        
        return result;
    }
};

void Test_Longest_Duplicate_Substring() {
    Solution solution;
    
    string s1 = "banana";
    cout << "Input: '" << s1 << "'" << endl;
    string result1 = solution.Longest_Dup_Binary_Search_Rolling_Hash(s1);
    cout << "Output (Binary Search + Rolling Hash): '" << result1 << "'" << endl;
    string result1_trie = solution.Longest_Dup_Suffix_Trie(s1);
    cout << "Output (Suffix Trie): '" << result1_trie << "'" << endl;
    
    string s2 = "abcd";
    cout << "\nInput: '" << s2 << "'" << endl;
    string result2 = solution.Longest_Dup_Binary_Search_Rolling_Hash(s2);
    cout << "Output (Binary Search + Rolling Hash): '" << result2 << "'" << endl;
    string result2_trie = solution.Longest_Dup_Suffix_Trie(s2);
    cout << "Output (Suffix Trie): '" << result2_trie << "'" << endl;
    
    string s3 = "aab";
    cout << "\nInput: '" << s3 << "'" << endl;
    string result3 = solution.Longest_Dup_Binary_Search_Rolling_Hash(s3);
    cout << "Output (Binary Search + Rolling Hash): '" << result3 << "'" << endl;
    string result3_trie = solution.Longest_Dup_Suffix_Trie(s3);
    cout << "Output (Suffix Trie): '" << result3_trie << "'" << endl;
    
    string s4 = "aaaa";
    cout << "\nInput: '" << s4 << "'" << endl;
    string result4 = solution.Longest_Dup_Binary_Search_Rolling_Hash(s4);
    cout << "Output (Binary Search + Rolling Hash): '" << result4 << "'" << endl;
    string result4_trie = solution.Longest_Dup_Suffix_Trie(s4);
    cout << "Output (Suffix Trie): '" << result4_trie << "'" << endl;
}

int main() {
    Test_Longest_Duplicate_Substring();
    return 0;
}
