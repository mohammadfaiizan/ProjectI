/*
Problem: Length of the Longest Valid Parentheses Substring
URL: https://practice.geeksforgeeks.org/problems/valid-substring0624/1

Problem Statement:
Find length of the longest valid (well-formed) parentheses substring.

Sample Input/Output:
Input: "(()"
Output: 2
Input: ")()())"
Output: 4
Input: "((()()()()(((())"
Output: 8
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Longest_Valid_Parentheses_Stack(string s) {
        stack<int> st;
        st.push(-1);
        int maxLen = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s[i] == '(') {
                st.push(i);
            } else {
                st.pop();
                if (st.empty()) {
                    st.push(i);
                } else {
                    maxLen = max(maxLen, i - st.top());
                }
            }
        }
        return maxLen;
    }

    int Longest_Valid_Parentheses_Two_Pass(string s) {
        int left = 0, right = 0, maxLen = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s[i] == '(') left++;
            else right++;
            if (left == right) maxLen = max(maxLen, 2 * right);
            else if (right > left) left = right = 0;
        }
        left = right = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            if (s[i] == '(') left++;
            else right++;
            if (left == right) maxLen = max(maxLen, 2 * left);
            else if (left > right) left = right = 0;
        }
        return maxLen;
    }
};

void Test_Longest_Valid_Parentheses() {
    Solution solution;
    
    cout << "=== Stack Approach ===" << endl;
    cout << "Input: \"(()\"" << endl;
    cout << "Output: " << solution.Longest_Valid_Parentheses_Stack("(()") << endl;
    
    cout << "\nInput: \")()())\"" << endl;
    cout << "Output: " << solution.Longest_Valid_Parentheses_Stack(")()())") << endl;
    
    cout << "\nInput: \"((()()()()(((())" << endl;
    cout << "Output: " << solution.Longest_Valid_Parentheses_Stack("((()()()()(((())") << endl;
    
    cout << "\nInput: \"\"" << endl;
    cout << "Output: " << solution.Longest_Valid_Parentheses_Stack("") << endl;
    
    cout << "\nInput: \"()(()\"" << endl;
    cout << "Output: " << solution.Longest_Valid_Parentheses_Stack("()(()") << endl;
    
    cout << "\n=== Two-Pass Approach ===" << endl;
    cout << "Input: \"(()\"" << endl;
    cout << "Output: " << solution.Longest_Valid_Parentheses_Two_Pass("(()") << endl;
    
    cout << "\nInput: \")()())\"" << endl;
    cout << "Output: " << solution.Longest_Valid_Parentheses_Two_Pass(")()())") << endl;
    
    cout << "\nInput: \"((()()()()(((())" << endl;
    cout << "Output: " << solution.Longest_Valid_Parentheses_Two_Pass("((()()()()(((())") << endl;
}

int main() {
    Test_Longest_Valid_Parentheses();
    return 0;
}
