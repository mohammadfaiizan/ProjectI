/*
Problem: Evaluation of Postfix and Prefix Expressions
URL: https://practice.geeksforgeeks.org/problems/evaluation-of-postfix-expression1735/1

Problem Statement:
Evaluate postfix and prefix expressions given as strings with single-digit operands and +,-,*,/,^ operators.

Sample Input/Output:
Input: "231*+9-"
Output: -4
Input: "+9*26"
Output: 21
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Evaluate_Postfix_Stack(string postfix) {
        stack<int> st;
        for (char c : postfix) {
            if (isdigit(c)) {
                st.push(c - '0');
            } else {
                int op2 = st.top(); st.pop();
                int op1 = st.top(); st.pop();
                int result = 0;
                if (c == '+') result = op1 + op2;
                else if (c == '-') result = op1 - op2;
                else if (c == '*') result = op1 * op2;
                else if (c == '/') result = op1 / op2;
                else if (c == '^') result = pow(op1, op2);
                st.push(result);
            }
        }
        return st.top();
    }

    int Evaluate_Prefix_Stack(string prefix) {
        stack<int> st;
        reverse(prefix.begin(), prefix.end());
        for (char c : prefix) {
            if (isdigit(c)) {
                st.push(c - '0');
            } else {
                int op1 = st.top(); st.pop();
                int op2 = st.top(); st.pop();
                int result = 0;
                if (c == '+') result = op1 + op2;
                else if (c == '-') result = op1 - op2;
                else if (c == '*') result = op1 * op2;
                else if (c == '/') result = op1 / op2;
                else if (c == '^') result = pow(op1, op2);
                st.push(result);
            }
        }
        return st.top();
    }
};

void Test_Expression_Evaluation() {
    Solution solution;
    
    cout << "=== Postfix Evaluation ===" << endl;
    cout << "231*+9- -> " << solution.Evaluate_Postfix_Stack("231*+9-") << endl;
    cout << "123+* -> " << solution.Evaluate_Postfix_Stack("123+*") << endl;
    cout << "23*4+ -> " << solution.Evaluate_Postfix_Stack("23*4+") << endl;
    cout << "52^3+ -> " << solution.Evaluate_Postfix_Stack("52^3+") << endl;
    
    cout << "\n=== Prefix Evaluation ===" << endl;
    cout << "+9*26 -> " << solution.Evaluate_Prefix_Stack("+9*26") << endl;
    cout << "*+123 -> " << solution.Evaluate_Prefix_Stack("*+123") << endl;
    cout << "+*234 -> " << solution.Evaluate_Prefix_Stack("+*234") << endl;
    cout << "+^523 -> " << solution.Evaluate_Prefix_Stack("+^523") << endl;
}

int main() {
    Test_Expression_Evaluation();
    return 0;
}
