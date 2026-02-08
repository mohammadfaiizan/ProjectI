/*
Problem: Find Shortest Unique Prefix for Every Word in a Given List
URL: https://www.geeksforgeeks.org/find-all-shortest-unique-prefixes-to-represent-each-word-in-a-given-list/

Problem Statement:
Given a list of words, find the shortest prefix that uniquely identifies each word.
Build a trie, track frequency at each node. The prefix where frequency becomes 1 is the unique prefix.

Sample Input/Output:
Input: ["zebra","dog","duck","dove"]
Output: ["z","dog","du","dov"]
*/

#include <bits/stdc++.h>
using namespace std;

struct TrieNode {
    TrieNode* children[26];
    int frequency;
    
    TrieNode() {
        for (int i = 0; i < 26; i++) {
            children[i] = nullptr;
        }
        frequency = 0;
    }
};

class Solution {
public:
    vector<string> Shortest_Prefix_Trie(vector<string>& words) {
        /*
        Shortest_Prefix_Trie (build trie with freq count, traverse for each word until freq=1, O(N*L))
        Time Complexity: O(N*L) where N is number of words, L is average length
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
                node->frequency++;
            }
        }
        
        vector<string> result;
        for (string word : words) {
            TrieNode* node = root;
            string prefix = "";
            for (char c : word) {
                int index = c - 'a';
                node = node->children[index];
                prefix += c;
                if (node->frequency == 1) {
                    break;
                }
            }
            result.push_back(prefix);
        }
        
        return result;
    }
    
    vector<string> Shortest_Prefix_Brute(vector<string>& words) {
        /*
        Brute force approach (compare each word with all others)
        Time Complexity: O(N^2 * L)
        Space Complexity: O(1)
        */
        vector<string> result;
        int n = words.size();
        
        for (int i = 0; i < n; i++) {
            int minLen = words[i].length();
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                int k = 0;
                while (k < words[i].length() && k < words[j].length() && 
                       words[i][k] == words[j][k]) {
                    k++;
                }
                if (k < words[i].length()) {
                    minLen = min(minLen, k + 1);
                }
            }
            result.push_back(words[i].substr(0, minLen));
        }
        
        return result;
    }
};

void Test_Shortest_Unique_Prefix() {
    Solution solution;
    
    cout << "=== Test Case 1 ===" << endl;
    vector<string> words1 = {"zebra", "dog", "duck", "dove"};
    vector<string> result1 = solution.Shortest_Prefix_Trie(words1);
    cout << "Input: ";
    for (string w : words1) cout << w << " ";
    cout << endl;
    cout << "Output: ";
    for (string r : result1) cout << r << " ";
    cout << endl;
    
    cout << "\n=== Test Case 2 ===" << endl;
    vector<string> words2 = {"geeksgeeks", "geeksquiz", "geeksforgeeks"};
    vector<string> result2 = solution.Shortest_Prefix_Trie(words2);
    cout << "Input: ";
    for (string w : words2) cout << w << " ";
    cout << endl;
    cout << "Output: ";
    for (string r : result2) cout << r << " ";
    cout << endl;
    
    cout << "\n=== Test Case 3 ===" << endl;
    vector<string> words3 = {"apple", "app", "ape", "bat", "ball"};
    vector<string> result3 = solution.Shortest_Prefix_Trie(words3);
    cout << "Input: ";
    for (string w : words3) cout << w << " ";
    cout << endl;
    cout << "Output: ";
    for (string r : result3) cout << r << " ";
    cout << endl;
    
    cout << "\n=== Test Case 4 (Single word) ===" << endl;
    vector<string> words4 = {"hello"};
    vector<string> result4 = solution.Shortest_Prefix_Trie(words4);
    cout << "Input: ";
    for (string w : words4) cout << w << " ";
    cout << endl;
    cout << "Output: ";
    for (string r : result4) cout << r << " ";
    cout << endl;
    
    cout << "\n=== Test Case 5 (All unique) ===" << endl;
    vector<string> words5 = {"cat", "dog", "bird"};
    vector<string> result5 = solution.Shortest_Prefix_Trie(words5);
    cout << "Input: ";
    for (string w : words5) cout << w << " ";
    cout << endl;
    cout << "Output: ";
    for (string r : result5) cout << r << " ";
    cout << endl;
}

int main() {
    Test_Shortest_Unique_Prefix();
    return 0;
}
