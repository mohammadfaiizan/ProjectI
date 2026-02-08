/*
Problem: Arithmetic Expression Conversion (Infix/Prefix/Postfix)
URL: https://www.geeksforgeeks.org/arithmetic-expression-evalution/

Problem Statement:
Implement all 6 conversions: infix-to-postfix, infix-to-prefix, prefix-to-infix, postfix-to-infix, prefix-to-postfix, postfix-to-prefix.

Sample Input/Output:
Input: "A+B*C"
Output (Infix to Postfix): "ABC*+"
Output (Infix to Prefix): "+A*BC"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Precedence(char op) {
        if (op == '^') return 3;
        if (op == '*' || op == '/') return 2;
        if (op == '+' || op == '-') return 1;
        return 0;
    }

    bool Is_Operator(char c) {
        return c == '+' || c == '-' || c == '*' || c == '/' || c == '^';
    }

    string Infix_To_Postfix_Stack(string infix) {
        stack<char> st;
        string postfix = "";
        for (char c : infix) {
            if (isalnum(c)) {
                postfix += c;
            } else if (c == '(') {
                st.push(c);
            } else if (c == ')') {
                while (!st.empty() && st.top() != '(') {
                    postfix += st.top();
                    st.pop();
                }
                st.pop();
            } else if (Is_Operator(c)) {
                while (!st.empty() && st.top() != '(' && Precedence(st.top()) >= Precedence(c)) {
                    postfix += st.top();
                    st.pop();
                }
                st.push(c);
            }
        }
        while (!st.empty()) {
            postfix += st.top();
            st.pop();
        }
        return postfix;
    }

    string Infix_To_Prefix_Stack(string infix) {
        reverse(infix.begin(), infix.end());
        for (int i = 0; i < infix.length(); i++) {
            if (infix[i] == '(') infix[i] = ')';
            else if (infix[i] == ')') infix[i] = '(';
        }
        stack<char> st;
        string prefix = "";
        for (char c : infix) {
            if (isalnum(c)) {
                prefix += c;
            } else if (c == '(') {
                st.push(c);
            } else if (c == ')') {
                while (!st.empty() && st.top() != '(') {
                    prefix += st.top();
                    st.pop();
                }
                st.pop();
            } else if (Is_Operator(c)) {
                while (!st.empty() && st.top() != '(' && Precedence(st.top()) > Precedence(c)) {
                    prefix += st.top();
                    st.pop();
                }
                st.push(c);
            }
        }
        while (!st.empty()) {
            prefix += st.top();
            st.pop();
        }
        reverse(prefix.begin(), prefix.end());
        return prefix;
    }

    string Prefix_To_Infix_Stack(string prefix) {
        stack<string> st;
        reverse(prefix.begin(), prefix.end());
        for (char c : prefix) {
            if (isalnum(c)) {
                st.push(string(1, c));
            } else if (Is_Operator(c)) {
                string op1 = st.top(); st.pop();
                string op2 = st.top(); st.pop();
                string temp = "(" + op1 + c + op2 + ")";
                st.push(temp);
            }
        }
        return st.top();
    }

    string Postfix_To_Infix_Stack(string postfix) {
        stack<string> st;
        for (char c : postfix) {
            if (isalnum(c)) {
                st.push(string(1, c));
            } else if (Is_Operator(c)) {
                string op2 = st.top(); st.pop();
                string op1 = st.top(); st.pop();
                string temp = "(" + op1 + c + op2 + ")";
                st.push(temp);
            }
        }
        return st.top();
    }

    string Prefix_To_Postfix_Stack(string prefix) {
        stack<string> st;
        reverse(prefix.begin(), prefix.end());
        for (char c : prefix) {
            if (isalnum(c)) {
                st.push(string(1, c));
            } else if (Is_Operator(c)) {
                string op1 = st.top(); st.pop();
                string op2 = st.top(); st.pop();
                string temp = op1 + op2 + c;
                st.push(temp);
            }
        }
        return st.top();
    }

    string Postfix_To_Prefix_Stack(string postfix) {
        stack<string> st;
        for (char c : postfix) {
            if (isalnum(c)) {
                st.push(string(1, c));
            } else if (Is_Operator(c)) {
                string op2 = st.top(); st.pop();
                string op1 = st.top(); st.pop();
                string temp = c + op1 + op2;
                st.push(temp);
            }
        }
        return st.top();
    }
};

void Test_Expression_Conversion() {
    Solution solution;
    
    cout << "=== Infix to Postfix ===" << endl;
    cout << "A+B*C -> " << solution.Infix_To_Postfix_Stack("A+B*C") << endl;
    cout << "(A+B)*C -> " << solution.Infix_To_Postfix_Stack("(A+B)*C") << endl;
    cout << "A+B*(C-D) -> " << solution.Infix_To_Postfix_Stack("A+B*(C-D)") << endl;
    
    cout << "\n=== Infix to Prefix ===" << endl;
    cout << "A+B*C -> " << solution.Infix_To_Prefix_Stack("A+B*C") << endl;
    cout << "(A+B)*C -> " << solution.Infix_To_Prefix_Stack("(A+B)*C") << endl;
    
    cout << "\n=== Prefix to Infix ===" << endl;
    cout << "+A*BC -> " << solution.Prefix_To_Infix_Stack("+A*BC") << endl;
    cout << "*+ABC -> " << solution.Prefix_To_Infix_Stack("*+ABC") << endl;
    
    cout << "\n=== Postfix to Infix ===" << endl;
    cout << "ABC*+ -> " << solution.Postfix_To_Infix_Stack("ABC*+") << endl;
    cout << "AB+C* -> " << solution.Postfix_To_Infix_Stack("AB+C*") << endl;
    
    cout << "\n=== Prefix to Postfix ===" << endl;
    cout << "+A*BC -> " << solution.Prefix_To_Postfix_Stack("+A*BC") << endl;
    cout << "*+ABC -> " << solution.Prefix_To_Postfix_Stack("*+ABC") << endl;
    
    cout << "\n=== Postfix to Prefix ===" << endl;
    cout << "ABC*+ -> " << solution.Postfix_To_Prefix_Stack("ABC*+") << endl;
    cout << "AB+C* -> " << solution.Postfix_To_Prefix_Stack("AB+C*") << endl;
}

int main() {
    Test_Expression_Conversion();
    return 0;
}
