/*
Problem: Check if a String is a Valid Shuffle of Two Other Strings
URL: https://www.programiz.com/java-programming/examples/check-valid-shuffle-of-strings

Problem Statement:
Given three strings s1, s2, and result, check if result is a valid shuffle of s1 and s2.
A valid shuffle maintains the relative order of characters from both strings.

Sample Input/Output:
Input: s1 = "XY", s2 = "12", result = "1XY2"
Output: YES (order of XY and 12 both maintained)

Input: s1 = "XY", s2 = "12", result = "Y12X"
Output: NO (order of XY not maintained)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Valid_Shuffle_Greedy(string s1, string s2, string result) {
        /*
        Greedy two-pointer approach
        Time Complexity: O(n) where n = result.size()
        Space Complexity: O(1)
        */
        if (result.size() != s1.size() + s2.size()) return false;
        int i = 0, j = 0;
        for (char c : result) {
            if (i < (int)s1.size() && c == s1[i]) i++;
            else if (j < (int)s2.size() && c == s2[j]) j++;
            else return false;
        }
        return i == (int)s1.size() && j == (int)s2.size();
    }

    bool Valid_Shuffle_Recursive(string s1, string s2, string result, int i, int j, int k) {
        /*
        Recursive approach
        Time Complexity: O(2^n) worst case
        Space Complexity: O(n) recursion stack
        */
        if (k == (int)result.size()) return i == (int)s1.size() && j == (int)s2.size();
        bool take_s1 = false, take_s2 = false;
        if (i < (int)s1.size() && s1[i] == result[k])
            take_s1 = Valid_Shuffle_Recursive(s1, s2, result, i + 1, j, k + 1);
        if (j < (int)s2.size() && s2[j] == result[k])
            take_s2 = Valid_Shuffle_Recursive(s1, s2, result, i, j + 1, k + 1);
        return take_s1 || take_s2;
    }

    bool Valid_Shuffle_DP(string s1, string s2, string result) {
        /*
        DP approach
        Time Complexity: O(m * n)
        Space Complexity: O(m * n)
        */
        int m = s1.size(), n = s2.size();
        if ((int)result.size() != m + n) return false;
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i > 0 && s1[i - 1] == result[i + j - 1])
                    dp[i][j] = dp[i][j] || dp[i - 1][j];
                if (j > 0 && s2[j - 1] == result[i + j - 1])
                    dp[i][j] = dp[i][j] || dp[i][j - 1];
            }
        }
        return dp[m][n];
    }
};

void Test_Valid_Shuffle() {
    Solution sol;
    struct TestCase { string s1, s2, result; };
    vector<TestCase> tests = {
        {"XY", "12", "1XY2"},
        {"XY", "12", "Y12X"},
        {"XY", "12", "X1Y2"},
        {"abc", "def", "adbecf"},
        {"abc", "def", "abcdef"}
    };

    for (auto& t : tests) {
        cout << "s1: " << t.s1 << ", s2: " << t.s2 << ", result: " << t.result << endl;
        cout << "Greedy: " << (sol.Valid_Shuffle_Greedy(t.s1, t.s2, t.result) ? "YES" : "NO") << endl;
        cout << "Recursive: " << (sol.Valid_Shuffle_Recursive(t.s1, t.s2, t.result, 0, 0, 0) ? "YES" : "NO") << endl;
        cout << "DP: " << (sol.Valid_Shuffle_DP(t.s1, t.s2, t.result) ? "YES" : "NO") << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Valid_Shuffle();
    return 0;
}
