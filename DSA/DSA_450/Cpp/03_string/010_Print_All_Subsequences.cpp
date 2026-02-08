/*
Problem: Print All Subsequences of a String
URL: https://www.geeksforgeeks.org/print-subsequences-string/

Problem Statement:
Given a string, print all possible subsequences of the string.
A subsequence is a sequence derived from another sequence by deleting some or no
elements without changing the order of the remaining elements.

Sample Input/Output:
Input: "abc"
Output: "", "a", "b", "c", "ab", "ac", "bc", "abc"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Subsequences_Include_Exclude(string s, string curr, int i, vector<string>& result) {
        /*
        Include/Exclude recursion
        Time Complexity: O(2^n)
        Space Complexity: O(n) recursion depth
        */
        if (i == (int)s.size()) {
            result.push_back(curr);
            return;
        }
        Subsequences_Include_Exclude(s, curr, i + 1, result);
        Subsequences_Include_Exclude(s, curr + s[i], i + 1, result);
    }

    void Subsequences_Backtracking(string& s, int n, int idx, string curr, vector<string>& result) {
        /*
        Backtracking - pick characters from index onwards
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        */
        if (!curr.empty()) result.push_back(curr);
        for (int i = idx; i < n; i++) {
            curr += s[i];
            Subsequences_Backtracking(s, n, i + 1, curr, result);
            curr.pop_back();
        }
    }

    vector<string> Subsequences_Bitmask(string s) {
        /*
        Bitmask - iterate over all 2^n subsets
        Time Complexity: O(n * 2^n)
        Space Complexity: O(2^n)
        */
        int n = s.size();
        int total = 1 << n;
        vector<string> result;
        for (int mask = 0; mask < total; mask++) {
            string sub = "";
            for (int j = 0; j < n; j++) {
                if (mask & (1 << j)) sub += s[j];
            }
            result.push_back(sub);
        }
        return result;
    }
};

void Test_Print_All_Subsequences() {
    Solution sol;
    vector<string> tests = {"abc", "ab", "a"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;

        vector<string> r1;
        sol.Subsequences_Include_Exclude(s, "", 0, r1);
        cout << "Include/Exclude (" << r1.size() << "): ";
        for (auto& x : r1) cout << "\"" << x << "\" ";
        cout << endl;

        vector<string> r2;
        sol.Subsequences_Backtracking(s, s.size(), 0, "", r2);
        cout << "Backtracking (" << r2.size() << "): ";
        for (auto& x : r2) cout << "\"" << x << "\" ";
        cout << endl;

        auto r3 = sol.Subsequences_Bitmask(s);
        cout << "Bitmask (" << r3.size() << "): ";
        for (auto& x : r3) cout << "\"" << x << "\" ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Print_All_Subsequences();
    return 0;
}
