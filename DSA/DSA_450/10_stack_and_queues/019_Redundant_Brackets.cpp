/*
Problem: Expression Contains Redundant Brackets or Not
URL: https://www.geeksforgeeks.org/expression-contains-redundant-bracket-not/

Problem Statement:
Check if expression contains redundant brackets (brackets without operator).

Sample Input/Output:
Input: "((a+b))"
Output: true (redundant)
Input: "(a+b*(c-d))"
Output: false
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Has_Redundant_Brackets_Stack(string s) {
        stack<char> st;
        for (char c : s) {
            if (c == '(' || c == '+' || c == '-' || c == '*' || c == '/') {
                st.push(c);
            } else if (c == ')') {
                bool hasOperator = false;
                while (!st.empty() && st.top() != '(') {
                    if (st.top() == '+' || st.top() == '-' || st.top() == '*' || st.top() == '/') {
                        hasOperator = true;
                    }
                    st.pop();
                }
                if (!st.empty()) st.pop();
                if (!hasOperator) return true;
            }
        }
        return false;
    }
};

void Test_Redundant_Brackets() {
    Solution solution;
    
    cout << "=== Redundant Brackets Check ===" << endl;
    cout << "Input: \"((a+b))\"" << endl;
    cout << "Output: " << (solution.Has_Redundant_Brackets_Stack("((a+b))") ? "true (redundant)" : "false") << endl;
    
    cout << "\nInput: \"(a+b*(c-d))\"" << endl;
    cout << "Output: " << (solution.Has_Redundant_Brackets_Stack("(a+b*(c-d))") ? "true (redundant)" : "false") << endl;
    
    cout << "\nInput: \"(a+b)\"" << endl;
    cout << "Output: " << (solution.Has_Redundant_Brackets_Stack("(a+b)") ? "true (redundant)" : "false") << endl;
    
    cout << "\nInput: \"((a+b)+c)\"" << endl;
    cout << "Output: " << (solution.Has_Redundant_Brackets_Stack("((a+b)+c)") ? "true (redundant)" : "false") << endl;
    
    cout << "\nInput: \"(a+(b)/c)\"" << endl;
    cout << "Output: " << (solution.Has_Redundant_Brackets_Stack("(a+(b)/c)") ? "true (redundant)" : "false") << endl;
    
    cout << "\nInput: \"(a+b*(c-d))\"" << endl;
    cout << "Output: " << (solution.Has_Redundant_Brackets_Stack("(a+b*(c-d))") ? "true (redundant)" : "false") << endl;
    
    cout << "\nInput: \"((a))\"" << endl;
    cout << "Output: " << (solution.Has_Redundant_Brackets_Stack("((a))") ? "true (redundant)" : "false") << endl;
}

int main() {
    Test_Redundant_Brackets();
    return 0;
}
