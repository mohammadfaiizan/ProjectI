/*
Problem: Minimum Number of Flips to Make Binary String Alternating
URL: https://practice.geeksforgeeks.org/problems/min-number-of-flips3210/1

Problem Statement:
Given a binary string, find the minimum number of flips required to make it alternating.

Sample Input/Output:
Input: "001"
Output: 1

Input: "0001010111"
Output: 2
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Flips_Two_Patterns(string s) {
        /*
        Compare with both possible alternating patterns (010... and 101...)
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int ans1 = 0, ans2 = 0;
        for (int i = 0; i < (int)s.size(); i++) {
            if (i % 2 == 0 && s[i] != '1' || i % 2 && s[i] != '0') ans1++;
            if (i % 2 == 0 && s[i] != '0' || i % 2 && s[i] != '1') ans2++;
        }
        return min(ans1, ans2);
    }

    int Min_Flips_Expected_Char(string s) {
        /*
        Build expected char and count mismatches
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = s.size();
        int flips1 = 0, flips2 = 0;
        for (int i = 0; i < n; i++) {
            char expected1 = (i % 2 == 0) ? '0' : '1';
            char expected2 = (i % 2 == 0) ? '1' : '0';
            if (s[i] != expected1) flips1++;
            if (s[i] != expected2) flips2++;
        }
        return min(flips1, flips2);
    }
};

void Test_Min_Flips_To_Alternate() {
    Solution sol;
    vector<string> tests = {"001", "0001010111", "01", "10", "1111", "0000", "0101"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        cout << "Two Patterns: " << sol.Min_Flips_Two_Patterns(s) << endl;
        cout << "Expected Char: " << sol.Min_Flips_Expected_Char(s) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Min_Flips_To_Alternate();
    return 0;
}
