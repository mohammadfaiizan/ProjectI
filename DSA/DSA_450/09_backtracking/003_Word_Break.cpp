/*
Problem: Word Break
URL: https://practice.geeksforgeeks.org/problems/word-break-part-23249/1

Problem Statement:
Given string s and dictionary, find all possible sentences by breaking s into dictionary words.

Sample Input/Output:
Input: s="catsanddog", dict=["cat","cats","and","sand","dog"]
Output: ["cats and dog","cat sand dog"]
Explanation: Two ways to break the string into valid words
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<string> Word_Break_Backtracking(string s, vector<string> &dict) {
        /*
        Backtracking with substring check
        Time Complexity: O(2^n * n)
        Space Complexity: O(n)
        */
        vector<string> result;
        unordered_set<string> word_set(dict.begin(), dict.end());
        string current_sentence = "";
        
        function<void(int)> backtrack = [&](int start) {
            if (start == s.length()) {
                result.push_back(current_sentence.substr(1));
                return;
            }
            
            for (int end = start + 1; end <= s.length(); end++) {
                string word = s.substr(start, end - start);
                if (word_set.find(word) != word_set.end()) {
                    string old_sentence = current_sentence;
                    current_sentence += " " + word;
                    backtrack(end);
                    current_sentence = old_sentence;
                }
            }
        };
        
        backtrack(0);
        return result;
    }
};

void Test_Word_Break() {
    Solution solution;
    
    string s = "catsanddog";
    vector<string> dict = {"cat", "cats", "and", "sand", "dog"};
    vector<string> sentences = solution.Word_Break_Backtracking(s, dict);
    
    cout << "Possible sentences:" << endl;
    for (const string &sentence : sentences) {
        cout << sentence << endl;
    }
}

int main() {
    Test_Word_Break();
    return 0;
}
