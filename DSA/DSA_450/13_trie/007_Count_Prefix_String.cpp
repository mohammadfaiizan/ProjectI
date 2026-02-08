/*
Problem: Count Number of Strings with Given Prefix
URL: https://www.geeksforgeeks.org/counting-number-words-trie/

Problem Statement:
Given a set of strings and a prefix, count how many strings have that prefix.

Sample Input/Output:
Input: words=["apple","app","ape","bat","ball"], prefix="ap"
Output: 3
Input: words=["apple","app","ape","bat","ball"], prefix="ba"
Output: 2
*/

#include <bits/stdc++.h>
using namespace std;

struct TrieNode {
    TrieNode* children[26];
    int prefixCount;
    bool isEndOfWord;
    
    TrieNode() {
        for (int i = 0; i < 26; i++) {
            children[i] = nullptr;
        }
        prefixCount = 0;
        isEndOfWord = false;
    }
};

class Solution {
public:
    int Count_Prefix_Trie(vector<string>& words, string prefix) {
        /*
        Count_Prefix_Trie (build trie with prefix count at each node, O(N*L) build + O(P) query)
        Time Complexity: O(N*L + P) where N is words, L is avg length, P is prefix length
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
                node->prefixCount++;
            }
            node->isEndOfWord = true;
        }
        
        TrieNode* node = root;
        for (char c : prefix) {
            int index = c - 'a';
            if (!node->children[index]) {
                return 0;
            }
            node = node->children[index];
        }
        
        return node->prefixCount;
    }
    
    int Count_Prefix_Brute(vector<string>& words, string prefix) {
        /*
        Count_Prefix_Brute (iterate all strings, check prefix, O(N*P))
        Time Complexity: O(N*P) where N is words, P is prefix length
        Space Complexity: O(1)
        */
        int count = 0;
        for (string word : words) {
            if (word.length() >= prefix.length() && 
                word.substr(0, prefix.length()) == prefix) {
                count++;
            }
        }
        return count;
    }
    
    int Count_Prefix_Binary_Search(vector<string>& words, string prefix) {
        /*
        Count_Prefix_Binary_Search (sort words, use binary search, O(N log N + log N))
        Time Complexity: O(N log N + log N)
        Space Complexity: O(1)
        */
        sort(words.begin(), words.end());
        int count = 0;
        int n = words.size();
        
        string nextPrefix = prefix;
        if (!nextPrefix.empty()) {
            nextPrefix[nextPrefix.length() - 1]++;
        }
        
        auto lower = lower_bound(words.begin(), words.end(), prefix);
        auto upper = lower_bound(words.begin(), words.end(), nextPrefix);
        
        return distance(lower, upper);
    }
    
    int Count_Prefix_Trie_DFS(vector<string>& words, string prefix) {
        /*
        Count_Prefix_Trie_DFS (build trie, traverse prefix, DFS to count words, O(N*L + P + W))
        Time Complexity: O(N*L + P + W) where W is words with prefix
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
                return 0;
            }
            node = node->children[index];
        }
        
        return CountWordsFromNode(node);
    }
    
    int CountWordsFromNode(TrieNode* node) {
        if (!node) return 0;
        int count = node->isEndOfWord ? 1 : 0;
        for (int i = 0; i < 26; i++) {
            if (node->children[i]) {
                count += CountWordsFromNode(node->children[i]);
            }
        }
        return count;
    }
};

void Test_Count_Prefix_String() {
    Solution solution;
    
    cout << "=== Test Case 1 ===" << endl;
    vector<string> words1 = {"apple", "app", "ape", "bat", "ball"};
    string prefix1 = "ap";
    cout << "Words: ";
    for (string w : words1) cout << w << " ";
    cout << endl;
    cout << "Prefix: " << prefix1 << endl;
    cout << "Count (Trie): " << solution.Count_Prefix_Trie(words1, prefix1) << endl;
    cout << "Count (Brute): " << solution.Count_Prefix_Brute(words1, prefix1) << endl;
    cout << "Count (Binary Search): " << solution.Count_Prefix_Binary_Search(words1, prefix1) << endl;
    cout << "Count (Trie DFS): " << solution.Count_Prefix_Trie_DFS(words1, prefix1) << endl;
    
    cout << "\n=== Test Case 2 ===" << endl;
    string prefix2 = "ba";
    cout << "Prefix: " << prefix2 << endl;
    cout << "Count (Trie): " << solution.Count_Prefix_Trie(words1, prefix2) << endl;
    cout << "Count (Brute): " << solution.Count_Prefix_Brute(words1, prefix2) << endl;
    cout << "Count (Binary Search): " << solution.Count_Prefix_Binary_Search(words1, prefix2) << endl;
    cout << "Count (Trie DFS): " << solution.Count_Prefix_Trie_DFS(words1, prefix2) << endl;
    
    cout << "\n=== Test Case 3 ===" << endl;
    vector<string> words3 = {"geeks", "geeksforgeeks", "geeksquiz", "geek"};
    string prefix3 = "geek";
    cout << "Words: ";
    for (string w : words3) cout << w << " ";
    cout << endl;
    cout << "Prefix: " << prefix3 << endl;
    cout << "Count (Trie): " << solution.Count_Prefix_Trie(words3, prefix3) << endl;
    cout << "Count (Brute): " << solution.Count_Prefix_Brute(words3, prefix3) << endl;
    cout << "Count (Binary Search): " << solution.Count_Prefix_Binary_Search(words3, prefix3) << endl;
    cout << "Count (Trie DFS): " << solution.Count_Prefix_Trie_DFS(words3, prefix3) << endl;
    
    cout << "\n=== Test Case 4 (No match) ===" << endl;
    string prefix4 = "xyz";
    cout << "Prefix: " << prefix4 << endl;
    cout << "Count (Trie): " << solution.Count_Prefix_Trie(words1, prefix4) << endl;
    cout << "Count (Brute): " << solution.Count_Prefix_Brute(words1, prefix4) << endl;
    cout << "Count (Binary Search): " << solution.Count_Prefix_Binary_Search(words1, prefix4) << endl;
    cout << "Count (Trie DFS): " << solution.Count_Prefix_Trie_DFS(words1, prefix4) << endl;
    
    cout << "\n=== Test Case 5 (Empty prefix) ===" << endl;
    string prefix5 = "";
    cout << "Prefix: (empty)" << endl;
    cout << "Count (Trie): " << solution.Count_Prefix_Trie(words1, prefix5) << endl;
    cout << "Count (Brute): " << solution.Count_Prefix_Brute(words1, prefix5) << endl;
    cout << "Count (Binary Search): " << solution.Count_Prefix_Binary_Search(words1, prefix5) << endl;
    cout << "Count (Trie DFS): " << solution.Count_Prefix_Trie_DFS(words1, prefix5) << endl;
    
    cout << "\n=== Test Case 6 (Single word) ===" << endl;
    vector<string> words6 = {"hello"};
    string prefix6 = "he";
    cout << "Words: ";
    for (string w : words6) cout << w << " ";
    cout << endl;
    cout << "Prefix: " << prefix6 << endl;
    cout << "Count (Trie): " << solution.Count_Prefix_Trie(words6, prefix6) << endl;
    cout << "Count (Brute): " << solution.Count_Prefix_Brute(words6, prefix6) << endl;
    cout << "Count (Binary Search): " << solution.Count_Prefix_Binary_Search(words6, prefix6) << endl;
    cout << "Count (Trie DFS): " << solution.Count_Prefix_Trie_DFS(words6, prefix6) << endl;
    
    cout << "\n=== Test Case 7 (Multiple prefixes) ===" << endl;
    vector<string> words7 = {"a", "aa", "aaa", "aaaa"};
    vector<string> prefixes7 = {"a", "aa", "aaa", "aaaa"};
    cout << "Words: ";
    for (string w : words7) cout << w << " ";
    cout << endl;
    for (string p : prefixes7) {
        cout << "Prefix '" << p << "': " << solution.Count_Prefix_Trie(words7, p) << endl;
    }
}

int main() {
    Test_Count_Prefix_String();
    return 0;
}
