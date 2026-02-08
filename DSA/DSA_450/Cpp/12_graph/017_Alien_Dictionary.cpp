/*
Problem: Alien Dictionary (Find Order of Characters)
URL: https://practice.geeksforgeeks.org/problems/alien-dictionary/1

Problem Statement:
Given a sorted dictionary of words in an alien language, find the order of characters in that language. The words are sorted lexicographically according to the alien language rules.

Sample Input/Output:
Input: words=["baa","abcd","abca","cab","cad"], k=4
Output: b d a c
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Alien_Dict_Topological(vector<string>& words, int k) {
        /*
        Build DAG from adjacent word comparisons + topological sort
        Time Complexity: O(N*|S|+K) where N=number of words, |S|=avg length, K=alphabet size
        Space Complexity: O(K)
        */
        vector<vector<int>> adj(k);
        vector<int> inDegree(k, -1);
        
        for (string& word : words) {
            for (char c : word) {
                if (inDegree[c - 'a'] == -1) {
                    inDegree[c - 'a'] = 0;
                }
            }
        }
        
        for (int i = 0; i < words.size() - 1; i++) {
            string word1 = words[i];
            string word2 = words[i + 1];
            
            int len = min(word1.length(), word2.length());
            bool found = false;
            
            for (int j = 0; j < len; j++) {
                if (word1[j] != word2[j]) {
                    int u = word1[j] - 'a';
                    int v = word2[j] - 'a';
                    adj[u].push_back(v);
                    inDegree[v]++;
                    found = true;
                    break;
                }
            }
            
            if (!found && word1.length() > word2.length()) {
                return "";
            }
        }
        
        queue<int> q;
        for (int i = 0; i < k; i++) {
            if (inDegree[i] == 0) {
                q.push(i);
            }
        }
        
        string result;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            result += (char)('a' + u);
            
            for (int v : adj[u]) {
                inDegree[v]--;
                if (inDegree[v] == 0) {
                    q.push(v);
                }
            }
        }
        
        for (int i = 0; i < k; i++) {
            if (inDegree[i] != -1 && inDegree[i] > 0) {
                return "";
            }
        }
        
        return result;
    }
};

void Test_Alien_Dict() {
    Solution solution;
    
    cout << "Test Case 1: Standard alien dictionary" << endl;
    vector<string> words1 = {"baa", "abcd", "abca", "cab", "cad"};
    int k1 = 4;
    string result1 = solution.Alien_Dict_Topological(words1, k1);
    cout << "Order: ";
    for (char c : result1) cout << c << " ";
    cout << endl;
    
    cout << "\nTest Case 2: Simple order" << endl;
    vector<string> words2 = {"caa", "aaa", "aab"};
    int k2 = 3;
    string result2 = solution.Alien_Dict_Topological(words2, k2);
    cout << "Order: ";
    for (char c : result2) cout << c << " ";
    cout << endl;
    
    cout << "\nTest Case 3: Single character" << endl;
    vector<string> words3 = {"a"};
    int k3 = 1;
    string result3 = solution.Alien_Dict_Topological(words3, k3);
    cout << "Order: ";
    for (char c : result3) cout << c << " ";
    cout << endl;
}

int main() {
    Test_Alien_Dict();
    return 0;
}
