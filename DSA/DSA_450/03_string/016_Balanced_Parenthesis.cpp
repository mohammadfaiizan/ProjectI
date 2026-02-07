/*
Problem: Balanced Parenthesis (Parenthesis Checker)
URL: https://practice.geeksforgeeks.org/problems/parenthesis-checker2744/1

Problem Statement:
Given an expression string x, examine whether the pairs and the orders of
{, }, (, ), [, ] are correct.

Sample Input/Output:
Input: "{([])}"
Output: true

Input: "[(])"
Output: false
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Balanced_Parenthesis_Stack(string x) {
        /*
        Using stack to match opening and closing brackets
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        stack<char> s;
        for (char c : x) {
            if (c == '(' || c == '{' || c == '[') {
                s.push(c);
            } else {
                if (s.empty()) return false;
                if (c == ')' && s.top() == '(') s.pop();
                else if (c == '}' && s.top() == '{') s.pop();
                else if (c == ']' && s.top() == '[') s.pop();
                else return false;
            }
        }
        return s.empty();
    }

    bool Balanced_Parenthesis_Map(string x) {
        /*
        Using map for bracket matching
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        stack<char> s;
        unordered_map<char, char> mp = {{')', '('}, {'}', '{'}, {']', '['}};

        for (char c : x) {
            if (mp.find(c) == mp.end()) {
                s.push(c);
            } else {
                if (s.empty() || s.top() != mp[c]) return false;
                s.pop();
            }
        }
        return s.empty();
    }

    bool Balanced_Parenthesis_Counter(string x) {
        /*
        Counter approach - works only for single type of brackets
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int count = 0;
        for (char c : x) {
            if (c == '(') count++;
            else if (c == ')') count--;
            if (count < 0) return false;
        }
        return count == 0;
    }
};

void Test_Balanced_Parenthesis() {
    Solution sol;
    vector<string> tests = {"{([])}", "[(])", "()", "((()))", "{[()]}", "{{[[(())]]}}", "(]", ""};

    for (auto& x : tests) {
        cout << "Input: \"" << x << "\"" << endl;
        cout << "Stack: " << (sol.Balanced_Parenthesis_Stack(x) ? "true" : "false") << endl;
        cout << "Map: " << (sol.Balanced_Parenthesis_Map(x) ? "true" : "false") << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Balanced_Parenthesis();
    return 0;
}
