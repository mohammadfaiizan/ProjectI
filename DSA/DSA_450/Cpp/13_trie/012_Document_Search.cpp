/*
Problem: Search Words in a Document using Trie
URL: https://www.geeksforgeeks.org/trie-based-solution-for-searching-words-in-document/

Problem Statement:
Given a document (text string) and a list of words, find which words appear in the document.
Build trie from the word list, then scan the document to check for matches.

Sample Input/Output:
Input: document="the cat sat on the mat", words=["cat","mat","bat","sat"]
Output: found: cat, mat, sat
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
    vector<string> Document_Search_Trie(string document, vector<string>& words) {
        /*
        Document_Search_Trie
        Time Complexity: O(N*L + D*L) where N is number of words, L is average length, D is document length
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
        
        unordered_set<string> found;
        int n = document.length();
        
        for (int i = 0; i < n; i++) {
            TrieNode* node = root;
            string current = "";
            
            for (int j = i; j < n; j++) {
                if (document[j] == ' ') break;
                
                int index = document[j] - 'a';
                if (index < 0 || index >= 26) break;
                
                if (!node->children[index]) {
                    break;
                }
                
                current += document[j];
                node = node->children[index];
                
                if (node->isEndOfWord) {
                    found.insert(current);
                }
            }
        }
        
        return vector<string>(found.begin(), found.end());
    }
    
    vector<string> Document_Search_Set(string document, vector<string>& words) {
        /*
        Document_Search_Set
        Time Complexity: O(N*L + D*L)
        Space Complexity: O(N*L)
        */
        unordered_set<string> wordSet(words.begin(), words.end());
        unordered_set<string> found;
        
        stringstream ss(document);
        string word;
        
        while (ss >> word) {
            if (wordSet.find(word) != wordSet.end()) {
                found.insert(word);
            }
        }
        
        return vector<string>(found.begin(), found.end());
    }
};

void Test_Document_Search() {
    Solution solution;
    
    string document1 = "the cat sat on the mat";
    vector<string> words1 = {"cat", "mat", "bat", "sat"};
    cout << "Document: '" << document1 << "'" << endl;
    cout << "Words to search: [cat, mat, bat, sat]" << endl;
    vector<string> result1 = solution.Document_Search_Trie(document1, words1);
    cout << "Found (Trie): ";
    for (string word : result1) {
        cout << word << " ";
    }
    cout << endl;
    vector<string> result1_set = solution.Document_Search_Set(document1, words1);
    cout << "Found (Set): ";
    for (string word : result1_set) {
        cout << word << " ";
    }
    cout << endl;
    
    string document2 = "hello world hello everyone";
    vector<string> words2 = {"hello", "world", "test", "everyone"};
    cout << "\nDocument: '" << document2 << "'" << endl;
    cout << "Words to search: [hello, world, test, everyone]" << endl;
    vector<string> result2 = solution.Document_Search_Trie(document2, words2);
    cout << "Found (Trie): ";
    for (string word : result2) {
        cout << word << " ";
    }
    cout << endl;
    vector<string> result2_set = solution.Document_Search_Set(document2, words2);
    cout << "Found (Set): ";
    for (string word : result2_set) {
        cout << word << " ";
    }
    cout << endl;
    
    string document3 = "the quick brown fox jumps over the lazy dog";
    vector<string> words3 = {"fox", "dog", "cat", "the", "quick"};
    cout << "\nDocument: '" << document3 << "'" << endl;
    cout << "Words to search: [fox, dog, cat, the, quick]" << endl;
    vector<string> result3 = solution.Document_Search_Trie(document3, words3);
    cout << "Found (Trie): ";
    for (string word : result3) {
        cout << word << " ";
    }
    cout << endl;
    vector<string> result3_set = solution.Document_Search_Set(document3, words3);
    cout << "Found (Set): ";
    for (string word : result3_set) {
        cout << word << " ";
    }
    cout << endl;
}

int main() {
    Test_Document_Search();
    return 0;
}
