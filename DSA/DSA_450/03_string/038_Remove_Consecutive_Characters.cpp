/*
Problem: Recursively Remove All Adjacent Duplicates / Consecutive Characters
URL: https://practice.geeksforgeeks.org/problems/consecutive-elements2306/1

Problem Statement:
Given a string, remove all consecutive duplicate characters and return the result.

Sample Input/Output:
Input: "aabb"
Output: "ab"

Input: "aabaa"
Output: "aba"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Remove_Consecutive_Iterative(string s) {
        /*
        Skip consecutive duplicates using iteration
        Time Complexity: O(n)
        Space Complexity: O(n) for result
        */
        string ans = "";
        int n = s.size();
        int i = 0;
        while (i < n) {
            ans += s[i];
            char temp = s[i];
            while (i < n && s[i] == temp) i++;
        }
        return ans;
    }

    string Remove_Consecutive_Stack(string s) {
        /*
        Using stack to track unique consecutive chars
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        stack<char> st;
        int n = s.size();
        for (int i = 0; i < n; i++) {
            if (st.empty() || st.top() != s[i])
                st.push(s[i]);
        }
        string res = "";
        while (!st.empty()) {
            res = st.top() + res;
            st.pop();
        }
        return res;
    }

    string Remove_Consecutive_Recursive(string s, int i) {
        /*
        Recursive approach
        Time Complexity: O(n)
        Space Complexity: O(n) recursion stack
        */
        if (i >= (int)s.size()) return "";
        string rest = Remove_Consecutive_Recursive(s, i + 1);
        if (!rest.empty() && rest[0] == s[i]) return rest;
        return s[i] + rest;
    }

    string Remove_Consecutive_Two_Pointer(string s) {
        /*
        In-place two pointer approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        if (s.empty()) return s;
        int j = 0;
        for (int i = 1; i < (int)s.size(); i++) {
            if (s[i] != s[j]) {
                j++;
                s[j] = s[i];
            }
        }
        return s.substr(0, j + 1);
    }
};

void Test_Remove_Consecutive() {
    Solution sol;
    vector<string> tests = {"aabb", "aabaa", "geeksforgeeks", "aabccba", "a", "aaaa"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        cout << "Iterative: " << sol.Remove_Consecutive_Iterative(s) << endl;
        cout << "Stack: " << sol.Remove_Consecutive_Stack(s) << endl;
        cout << "Recursive: " << sol.Remove_Consecutive_Recursive(s, 0) << endl;
        cout << "Two Pointer: " << sol.Remove_Consecutive_Two_Pointer(s) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Remove_Consecutive();
    return 0;
}
