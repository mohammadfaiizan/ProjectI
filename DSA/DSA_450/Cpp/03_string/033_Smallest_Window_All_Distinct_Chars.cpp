/*
Problem: Smallest Window Containing All Distinct Characters of String
URL: https://practice.geeksforgeeks.org/problems/smallest-distant-window3132/1

Problem Statement:
Given a string s, find the smallest window (substring) that contains all
distinct characters of the string.

Sample Input/Output:
Input: "aabcbcdbca"
Output: "dbca" (length 4)

Input: "aaab"
Output: "ab" (length 2)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Smallest_Window_Sliding(string str) {
        /*
        Sliding window approach
        Time Complexity: O(n)
        Space Complexity: O(1) - fixed 256 chars
        */
        int n = str.length();
        int dist_count = unordered_set<char>(str.begin(), str.end()).size();

        int start = 0, start_index = -1, min_len = INT_MAX;
        int count = 0;
        int curr_count[256] = {0};

        for (int j = 0; j < n; j++) {
            curr_count[(int)str[j]]++;
            if (curr_count[(int)str[j]] == 1) count++;

            if (count == dist_count) {
                while (curr_count[(int)str[start]] > 1) {
                    curr_count[(int)str[start]]--;
                    start++;
                }
                int len_window = j - start + 1;
                if (min_len > len_window) {
                    min_len = len_window;
                    start_index = start;
                }
            }
        }
        return str.substr(start_index, min_len);
    }

    string Smallest_Window_Brute(string str) {
        /*
        Brute force - check all substrings
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = str.length();
        int dist_count = unordered_set<char>(str.begin(), str.end()).size();
        int min_len = INT_MAX;
        string res;

        for (int i = 0; i < n; i++) {
            int count = 0;
            int visited[256] = {0};
            string sub = "";
            for (int j = i; j < n; j++) {
                if (visited[(int)str[j]] == 0) {
                    count++;
                    visited[(int)str[j]] = 1;
                }
                sub += str[j];
                if (count == dist_count) break;
            }
            if ((int)sub.length() < min_len && count == dist_count) {
                res = sub;
                min_len = res.length();
            }
        }
        return res;
    }
};

void Test_Smallest_Window_All_Distinct() {
    Solution sol;
    vector<string> tests = {"aabcbcdbca", "aaab", "abcdef", "aabcbcdbcaabc"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        cout << "Sliding: " << sol.Smallest_Window_Sliding(s) << endl;
        cout << "Brute: " << sol.Smallest_Window_Brute(s) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Smallest_Window_All_Distinct();
    return 0;
}
