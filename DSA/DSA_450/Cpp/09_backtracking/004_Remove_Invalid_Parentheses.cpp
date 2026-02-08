/*
Problem: Remove Invalid Parentheses
URL: https://leetcode.com/problems/remove-invalid-parentheses/

Problem Statement:
Remove minimum number of invalid parentheses to make string valid. Return all unique results.

Sample Input/Output:
Input: s="()())()"
Output: ["(())()","()()()"]
Explanation: Remove one ')' to make valid
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<string> Remove_Invalid_Parentheses_BFS(string s) {
        /*
        BFS level-by-level
        Time Complexity: O(2^n)
        Space Complexity: O(2^n)
        */
        vector<string> result;
        unordered_set<string> visited;
        queue<string> q;
        q.push(s);
        visited.insert(s);
        bool found = false;
        
        while (!q.empty()) {
            string current = q.front();
            q.pop();
            
            if (Is_Valid(current)) {
                result.push_back(current);
                found = true;
            }
            
            if (found) continue;
            
            for (int i = 0; i < current.length(); i++) {
                if (current[i] != '(' && current[i] != ')') continue;
                
                string next = current.substr(0, i) + current.substr(i + 1);
                if (visited.find(next) == visited.end()) {
                    visited.insert(next);
                    q.push(next);
                }
            }
        }
        
        return result;
    }
    
    vector<string> Remove_Invalid_Parentheses_Backtracking(string s) {
        /*
        Backtracking with min removal count
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        */
        vector<string> result;
        int min_removal = Get_Min_Removal(s);
        unordered_set<string> result_set;
        
        function<void(string, int, int, int, int)> backtrack = [&](string current, int index, int left_count, int right_count, int removed) {
            if (index == s.length()) {
                if (left_count == right_count && removed == min_removal) {
                    result_set.insert(current);
                }
                return;
            }
            
            if (s[index] != '(' && s[index] != ')') {
                backtrack(current + s[index], index + 1, left_count, right_count, removed);
                return;
            }
            
            backtrack(current, index + 1, left_count, right_count, removed + 1);
            
            if (s[index] == '(') {
                backtrack(current + '(', index + 1, left_count + 1, right_count, removed);
            } else if (right_count < left_count) {
                backtrack(current + ')', index + 1, left_count, right_count + 1, removed);
            }
        };
        
        backtrack("", 0, 0, 0, 0);
        result.assign(result_set.begin(), result_set.end());
        return result;
    }
    
private:
    bool Is_Valid(string s) {
        int count = 0;
        for (char c : s) {
            if (c == '(') count++;
            else if (c == ')') {
                count--;
                if (count < 0) return false;
            }
        }
        return count == 0;
    }
    
    int Get_Min_Removal(string s) {
        int left = 0, right = 0;
        for (char c : s) {
            if (c == '(') left++;
            else if (c == ')') {
                if (left > 0) left--;
                else right++;
            }
        }
        return left + right;
    }
};

void Test_Remove_Invalid_Parentheses() {
    Solution solution;
    
    string s = "()())()";
    vector<string> result1 = solution.Remove_Invalid_Parentheses_BFS(s);
    cout << "BFS Results:" << endl;
    for (const string &str : result1) {
        cout << str << endl;
    }
    
    vector<string> result2 = solution.Remove_Invalid_Parentheses_Backtracking(s);
    cout << "Backtracking Results:" << endl;
    for (const string &str : result2) {
        cout << str << endl;
    }
}

int main() {
    Test_Remove_Invalid_Parentheses();
    return 0;
}
