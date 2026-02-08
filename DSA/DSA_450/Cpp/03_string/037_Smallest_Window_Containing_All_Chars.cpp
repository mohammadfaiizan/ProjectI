/*
Problem: Smallest Window in a String Containing All Characters of Another String
URL: https://www.geeksforgeeks.org/find-the-smallest-window-in-a-string-containing-all-characters-of-another-string/

Problem Statement:
Given two strings s and t, find the smallest window in s which contains all
characters of t (including duplicates).

Sample Input/Output:
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"

Input: s = "this is a test string", t = "tist"
Output: "t stri"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Min_Window_Sliding(string s, string t) {
        /*
        Sliding window with frequency counting
        Time Complexity: O(n)
        Space Complexity: O(1) - fixed 256
        */
        if (s.size() < t.size()) return "";

        int hash_pat[256] = {0}, hash_str[256] = {0};
        for (char c : t) hash_pat[(int)c]++;

        int start = 0, start_index = -1, min_len = INT_MAX;
        int count = 0;
        int len2 = t.size();

        for (int j = 0; j < (int)s.size(); j++) {
            hash_str[(int)s[j]]++;
            if (hash_str[(int)s[j]] <= hash_pat[(int)s[j]])
                count++;

            if (count == len2) {
                while (hash_str[(int)s[start]] > hash_pat[(int)s[start]] ||
                       hash_pat[(int)s[start]] == 0) {
                    if (hash_str[(int)s[start]] > hash_pat[(int)s[start]])
                        hash_str[(int)s[start]]--;
                    start++;
                }
                int len_window = j - start + 1;
                if (min_len > len_window) {
                    min_len = len_window;
                    start_index = start;
                }
            }
        }

        if (start_index == -1) return "";
        return s.substr(start_index, min_len);
    }

    string Min_Window_Optimized(string s, string t) {
        /*
        Optimized sliding window with distinct char count
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int m[256] = {0};
        int count = 0;
        for (char c : t) {
            if (m[(int)c] == 0) count++;
            m[(int)c]++;
        }

        int ans = INT_MAX, start = 0;
        int i = 0, j = 0;

        while (j < (int)s.size()) {
            m[(int)s[j]]--;
            if (m[(int)s[j]] == 0) count--;

            if (count == 0) {
                while (count == 0) {
                    if (ans > j - i + 1) {
                        ans = j - i + 1;
                        start = i;
                    }
                    m[(int)s[i]]++;
                    if (m[(int)s[i]] > 0) count++;
                    i++;
                }
            }
            j++;
        }

        return ans == INT_MAX ? "" : s.substr(start, ans);
    }
};

void Test_Smallest_Window_Containing_All() {
    Solution sol;
    struct TestCase { string s, t; };
    vector<TestCase> tests = {
        {"ADOBECODEBANC", "ABC"},
        {"this is a test string", "tist"},
        {"aa", "aa"},
        {"a", "aa"},
        {"ab", "b"}
    };

    for (auto& tc : tests) {
        cout << "s: \"" << tc.s << "\", t: \"" << tc.t << "\"" << endl;
        cout << "Sliding: \"" << sol.Min_Window_Sliding(tc.s, tc.t) << "\"" << endl;
        cout << "Optimized: \"" << sol.Min_Window_Optimized(tc.s, tc.t) << "\"" << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Smallest_Window_Containing_All();
    return 0;
}
