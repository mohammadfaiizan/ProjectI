/*
Problem: Check for Balanced Parentheses
URL: https://practice.geeksforgeeks.org/problems/parenthesis-checker2744/1

Problem Statement:
Given a string of brackets, check if it is balanced. Handle '(', ')', '{', '}', '[', ']'.

Sample Input/Output:
Input: "()"
Output: true
Input: "([{}])"
Output: true
Input: "(]"
Output: false
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Is_Balanced_Stack(string s) {
        stack<char> st;
        for (char c : s) {
            if (c == '(' || c == '{' || c == '[') {
                st.push(c);
            } else {
                if (st.empty()) return false;
                char top = st.top();
                if ((c == ')' && top != '(') ||
                    (c == '}' && top != '{') ||
                    (c == ']' && top != '[')) {
                    return false;
                }
                st.pop();
            }
        }
        return st.empty();
    }
};

void Test_Balanced_Parentheses() {
    Solution solution;
    cout << "Balanced Parentheses Tests:" << endl;
    
    cout << "\"()\": " << solution.Is_Balanced_Stack("()") << endl;
    cout << "\"([{}])\": " << solution.Is_Balanced_Stack("([{}])") << endl;
    cout << "\"(]\": " << solution.Is_Balanced_Stack("(]") << endl;
    cout << "\"\": " << solution.Is_Balanced_Stack("") << endl;
    cout << "\"(((\": " << solution.Is_Balanced_Stack("(((") << endl;
    cout << "\"()[]{}\": " << solution.Is_Balanced_Stack("()[]{}") << endl;
    cout << "\"([)]\": " << solution.Is_Balanced_Stack("([)]") << endl;
    cout << "\"({[]})\": " << solution.Is_Balanced_Stack("({[]})") << endl;
}

int main() {
    Test_Balanced_Parentheses();
    return 0;
}
