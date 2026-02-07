/*
Problem: Minimum Number of Bracket Reversals
URL: https://practice.geeksforgeeks.org/problems/count-the-reversals0401/1

Problem Statement:
Given a string consisting of only '{' and '}', find the minimum number of
reversals required to make the expression balanced. Return -1 if not possible.

Sample Input/Output:
Input: "}}{{"
Output: 2

Input: "{{{"
Output: -1

Input: "{{}{{{}}{{"
Output: 3
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Reversals_Stack(string str) {
        /*
        Remove balanced pairs using stack, then compute from remaining
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        int len = str.length();
        if (len % 2) return -1;

        stack<char> stck;
        for (int i = 0; i < len; i++) {
            if (str[i] == '}' && !stck.empty() && stck.top() == '{')
                stck.pop();
            else
                stck.push(str[i]);
        }

        int stack_len = stck.size();
        int left = 0;
        while (!stck.empty() && stck.top() == '{') {
            stck.pop();
            left++;
        }
        int right = stack_len - left;
        return (int)ceil((double)right / 2) + (int)ceil((double)left / 2);
    }

    int Min_Reversals_Counter(string s) {
        /*
        Counter approach - no extra space for stack
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int len = s.length();
        if (len % 2) return -1;

        int left = 0, right = 0;
        for (int i = 0; i < len; i++) {
            if (s[i] == '{') {
                left++;
            } else {
                if (left == 0) right++;
                else left--;
            }
        }
        return (int)ceil((double)right / 2) + (int)ceil((double)left / 2);
    }
};

void Test_Min_Bracket_Reversals() {
    Solution sol;
    vector<string> tests = {"}{", "{{{{", "}{{}}{{{", "}}{{", "{{{", "{{}{{{}}{{"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        cout << "Stack: " << sol.Min_Reversals_Stack(s) << endl;
        cout << "Counter: " << sol.Min_Reversals_Counter(s) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Min_Bracket_Reversals();
    return 0;
}
