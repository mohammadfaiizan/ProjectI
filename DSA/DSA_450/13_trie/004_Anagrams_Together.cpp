/*
Problem: Print All Anagrams Together
URL: https://practice.geeksforgeeks.org/problems/print-anagrams-together/1

Problem Statement:
Given a sequence of words, group all anagrams together.

Sample Input/Output:
Input: ["cat","dog","tac","god","act","ogd"]
Output: groups [cat,tac,act],[dog,god,ogd]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<string>> Anagrams_Sorted_Key(vector<string>& words) {
        /*
        Anagrams_Sorted_Key (sort each word as key, group using map, O(N * K log K))
        Time Complexity: O(N * K log K) where N is number of words, K is average length
        Space Complexity: O(N * K)
        */
        unordered_map<string, vector<string>> groups;
        
        for (string word : words) {
            string key = word;
            sort(key.begin(), key.end());
            groups[key].push_back(word);
        }
        
        vector<vector<string>> result;
        for (auto& pair : groups) {
            result.push_back(pair.second);
        }
        
        return result;
    }
    
    vector<vector<string>> Anagrams_Count_Key(vector<string>& words) {
        /*
        Anagrams_Count_Key (use character frequency as key, O(N * K))
        Time Complexity: O(N * K) where N is number of words, K is average length
        Space Complexity: O(N * K)
        */
        unordered_map<string, vector<string>> groups;
        
        for (string word : words) {
            vector<int> count(26, 0);
            for (char c : word) {
                count[c - 'a']++;
            }
            
            string key = "";
            for (int i = 0; i < 26; i++) {
                if (count[i] > 0) {
                    key += string(1, 'a' + i) + to_string(count[i]);
                }
            }
            
            groups[key].push_back(word);
        }
        
        vector<vector<string>> result;
        for (auto& pair : groups) {
            result.push_back(pair.second);
        }
        
        return result;
    }
    
    vector<vector<string>> Anagrams_Trie(vector<string>& words) {
        /*
        Anagrams_Trie (group by sorted characters using trie-like structure)
        Time Complexity: O(N * K log K)
        Space Complexity: O(N * K)
        */
        map<string, vector<string>> groups;
        
        for (string word : words) {
            string key = word;
            sort(key.begin(), key.end());
            groups[key].push_back(word);
        }
        
        vector<vector<string>> result;
        for (auto& pair : groups) {
            result.push_back(pair.second);
        }
        
        return result;
    }
};

void Test_Anagrams_Together() {
    Solution solution;
    
    cout << "=== Test Case 1 ===" << endl;
    vector<string> words1 = {"cat", "dog", "tac", "god", "act", "ogd"};
    cout << "Input: ";
    for (string w : words1) cout << w << " ";
    cout << endl;
    
    vector<vector<string>> result1 = solution.Anagrams_Sorted_Key(words1);
    cout << "Output (Sorted Key):" << endl;
    for (auto& group : result1) {
        cout << "[";
        for (int i = 0; i < group.size(); i++) {
            cout << group[i];
            if (i < group.size() - 1) cout << ",";
        }
        cout << "] ";
    }
    cout << endl;
    
    vector<vector<string>> result1b = solution.Anagrams_Count_Key(words1);
    cout << "Output (Count Key):" << endl;
    for (auto& group : result1b) {
        cout << "[";
        for (int i = 0; i < group.size(); i++) {
            cout << group[i];
            if (i < group.size() - 1) cout << ",";
        }
        cout << "] ";
    }
    cout << endl;
    
    cout << "\n=== Test Case 2 ===" << endl;
    vector<string> words2 = {"eat", "tea", "tan", "ate", "nat", "bat"};
    cout << "Input: ";
    for (string w : words2) cout << w << " ";
    cout << endl;
    
    vector<vector<string>> result2 = solution.Anagrams_Sorted_Key(words2);
    cout << "Output:" << endl;
    for (auto& group : result2) {
        cout << "[";
        for (int i = 0; i < group.size(); i++) {
            cout << group[i];
            if (i < group.size() - 1) cout << ",";
        }
        cout << "] ";
    }
    cout << endl;
    
    cout << "\n=== Test Case 3 ===" << endl;
    vector<string> words3 = {"listen", "silent", "enlist", "hello", "world"};
    cout << "Input: ";
    for (string w : words3) cout << w << " ";
    cout << endl;
    
    vector<vector<string>> result3 = solution.Anagrams_Sorted_Key(words3);
    cout << "Output:" << endl;
    for (auto& group : result3) {
        cout << "[";
        for (int i = 0; i < group.size(); i++) {
            cout << group[i];
            if (i < group.size() - 1) cout << ",";
        }
        cout << "] ";
    }
    cout << endl;
    
    cout << "\n=== Test Case 4 (Single group) ===" << endl;
    vector<string> words4 = {"abc", "bca", "cab"};
    cout << "Input: ";
    for (string w : words4) cout << w << " ";
    cout << endl;
    
    vector<vector<string>> result4 = solution.Anagrams_Sorted_Key(words4);
    cout << "Output:" << endl;
    for (auto& group : result4) {
        cout << "[";
        for (int i = 0; i < group.size(); i++) {
            cout << group[i];
            if (i < group.size() - 1) cout << ",";
        }
        cout << "] ";
    }
    cout << endl;
    
    cout << "\n=== Test Case 5 (No anagrams) ===" << endl;
    vector<string> words5 = {"apple", "banana", "cherry"};
    cout << "Input: ";
    for (string w : words5) cout << w << " ";
    cout << endl;
    
    vector<vector<string>> result5 = solution.Anagrams_Sorted_Key(words5);
    cout << "Output:" << endl;
    for (auto& group : result5) {
        cout << "[";
        for (int i = 0; i < group.size(); i++) {
            cout << group[i];
            if (i < group.size() - 1) cout << ",";
        }
        cout << "] ";
    }
    cout << endl;
}

int main() {
    Test_Anagrams_Together();
    return 0;
}
