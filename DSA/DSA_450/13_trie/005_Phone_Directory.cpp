/*
Problem: Implement Phone Directory using Trie
URL: https://practice.geeksforgeeks.org/problems/phone-directory4628/1

Problem Statement:
Given a list of contacts and a query string, for each prefix of the query, return all contacts that start with that prefix (autocomplete-style).

Sample Input/Output:
Input: contacts=["geeikistest","geeksforgeeks","geeksquiz"], query="geeq"
Output: suggestions for each prefix
*/

#include <bits/stdc++.h>
using namespace std;

struct TrieNode {
    TrieNode* children[26];
    bool isEndOfWord;
    vector<string> words;
    
    TrieNode() {
        for (int i = 0; i < 26; i++) {
            children[i] = nullptr;
        }
        isEndOfWord = false;
    }
};

class Solution {
public:
    vector<vector<string>> Phone_Directory_Trie(vector<string>& contacts, string query) {
        /*
        Phone_Directory_Trie (build trie, for each prefix collect all words via DFS, O(N*L + Q*N*L))
        Time Complexity: O(N*L + Q*N*L) where N is contacts, L is avg length, Q is query length
        Space Complexity: O(N*L)
        */
        TrieNode* root = new TrieNode();
        
        for (string contact : contacts) {
            TrieNode* node = root;
            for (char c : contact) {
                int index = c - 'a';
                if (!node->children[index]) {
                    node->children[index] = new TrieNode();
                }
                node = node->children[index];
            }
            node->isEndOfWord = true;
        }
        
        vector<vector<string>> result;
        TrieNode* node = root;
        string prefix = "";
        
        for (char c : query) {
            int index = c - 'a';
            prefix += c;
            
            if (!node->children[index]) {
                while (result.size() < query.length()) {
                    result.push_back({"0"});
                }
                break;
            }
            
            node = node->children[index];
            vector<string> suggestions;
            CollectWords(node, prefix, suggestions);
            
            if (suggestions.empty()) {
                suggestions.push_back("0");
            }
            
            result.push_back(suggestions);
        }
        
        return result;
    }
    
    void CollectWords(TrieNode* node, string prefix, vector<string>& result) {
        if (node->isEndOfWord) {
            result.push_back(prefix);
        }
        
        for (int i = 0; i < 26; i++) {
            if (node->children[i]) {
                CollectWords(node->children[i], prefix + char('a' + i), result);
            }
        }
    }
    
    vector<vector<string>> Phone_Directory_Brute(vector<string>& contacts, string query) {
        /*
        Brute force approach (filter contacts for each prefix)
        Time Complexity: O(Q * N * L) where Q is query length, N is contacts, L is avg length
        Space Complexity: O(N * L)
        */
        vector<vector<string>> result;
        
        for (int i = 1; i <= query.length(); i++) {
            string prefix = query.substr(0, i);
            vector<string> suggestions;
            
            for (string contact : contacts) {
                if (contact.length() >= prefix.length() && 
                    contact.substr(0, prefix.length()) == prefix) {
                    suggestions.push_back(contact);
                }
            }
            
            sort(suggestions.begin(), suggestions.end());
            
            if (suggestions.empty()) {
                suggestions.push_back("0");
            }
            
            result.push_back(suggestions);
        }
        
        return result;
    }
    
    vector<vector<string>> Phone_Directory_Optimized(vector<string>& contacts, string query) {
        /*
        Optimized with sorting and binary search
        Time Complexity: O(N log N + Q * log N * L)
        Space Complexity: O(N * L)
        */
        sort(contacts.begin(), contacts.end());
        vector<vector<string>> result;
        
        for (int i = 1; i <= query.length(); i++) {
            string prefix = query.substr(0, i);
            vector<string> suggestions;
            
            for (string contact : contacts) {
                if (contact.length() >= prefix.length() && 
                    contact.substr(0, prefix.length()) == prefix) {
                    suggestions.push_back(contact);
                } else if (!suggestions.empty() && contact > prefix) {
                    break;
                }
            }
            
            if (suggestions.empty()) {
                suggestions.push_back("0");
            }
            
            result.push_back(suggestions);
        }
        
        return result;
    }
};

void Test_Phone_Directory() {
    Solution solution;
    
    cout << "=== Test Case 1 ===" << endl;
    vector<string> contacts1 = {"geeikistest", "geeksforgeeks", "geeksquiz"};
    string query1 = "geeq";
    cout << "Contacts: ";
    for (string c : contacts1) cout << c << " ";
    cout << endl;
    cout << "Query: " << query1 << endl;
    
    vector<vector<string>> result1 = solution.Phone_Directory_Trie(contacts1, query1);
    cout << "Output:" << endl;
    for (int i = 0; i < result1.size(); i++) {
        cout << "Prefix '" << query1.substr(0, i + 1) << "': ";
        for (string s : result1[i]) {
            cout << s << " ";
        }
        cout << endl;
    }
    
    cout << "\n=== Test Case 2 ===" << endl;
    vector<string> contacts2 = {"g", "ge", "gee", "geek", "geeks", "geeksforgeeks"};
    string query2 = "geeks";
    cout << "Contacts: ";
    for (string c : contacts2) cout << c << " ";
    cout << endl;
    cout << "Query: " << query2 << endl;
    
    vector<vector<string>> result2 = solution.Phone_Directory_Trie(contacts2, query2);
    cout << "Output:" << endl;
    for (int i = 0; i < result2.size(); i++) {
        cout << "Prefix '" << query2.substr(0, i + 1) << "': ";
        for (string s : result2[i]) {
            cout << s << " ";
        }
        cout << endl;
    }
    
    cout << "\n=== Test Case 3 ===" << endl;
    vector<string> contacts3 = {"apple", "app", "ape", "application", "apply"};
    string query3 = "app";
    cout << "Contacts: ";
    for (string c : contacts3) cout << c << " ";
    cout << endl;
    cout << "Query: " << query3 << endl;
    
    vector<vector<string>> result3 = solution.Phone_Directory_Trie(contacts3, query3);
    cout << "Output:" << endl;
    for (int i = 0; i < result3.size(); i++) {
        cout << "Prefix '" << query3.substr(0, i + 1) << "': ";
        for (string s : result3[i]) {
            cout << s << " ";
        }
        cout << endl;
    }
    
    cout << "\n=== Test Case 4 (No match) ===" << endl;
    vector<string> contacts4 = {"cat", "dog", "bird"};
    string query4 = "xyz";
    cout << "Contacts: ";
    for (string c : contacts4) cout << c << " ";
    cout << endl;
    cout << "Query: " << query4 << endl;
    
    vector<vector<string>> result4 = solution.Phone_Directory_Trie(contacts4, query4);
    cout << "Output:" << endl;
    for (int i = 0; i < result4.size(); i++) {
        cout << "Prefix '" << query4.substr(0, i + 1) << "': ";
        for (string s : result4[i]) {
            cout << s << " ";
        }
        cout << endl;
    }
    
    cout << "\n=== Test Case 5 (Single contact) ===" << endl;
    vector<string> contacts5 = {"hello"};
    string query5 = "he";
    cout << "Contacts: ";
    for (string c : contacts5) cout << c << " ";
    cout << endl;
    cout << "Query: " << query5 << endl;
    
    vector<vector<string>> result5 = solution.Phone_Directory_Trie(contacts5, query5);
    cout << "Output:" << endl;
    for (int i = 0; i < result5.size(); i++) {
        cout << "Prefix '" << query5.substr(0, i + 1) << "': ";
        for (string s : result5[i]) {
            cout << s << " ";
        }
        cout << endl;
    }
}

int main() {
    Test_Phone_Directory();
    return 0;
}
