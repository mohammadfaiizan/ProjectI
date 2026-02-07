/*
Problem: Longest Common Prefix
URL: https://leetcode.com/problems/longest-common-prefix/

Problem Statement:
Write a function to find the longest common prefix string amongst an array of strings.
If there is no common prefix, return an empty string.

Sample Input/Output:
Input: strs = ["flower","flow","flight"]
Output: "fl"

Input: strs = ["dog","racecar","car"]
Output: ""
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string LCP_Horizontal_Scan(vector<string>& strs) {
        /*
        Compare prefix with each string and shrink
        Time Complexity: O(S) where S = sum of all chars
        Space Complexity: O(1)
        */
        if (strs.empty()) return "";
        string ans = strs[0];
        for (int i = 1; i < (int)strs.size(); i++) {
            int j = 0;
            while (j < (int)min(ans.size(), strs[i].size()) && ans[j] == strs[i][j]) j++;
            ans = ans.substr(0, j);
            if (ans.empty()) return "";
        }
        return ans;
    }

    string LCP_Vertical_Scan(vector<string>& strs) {
        /*
        Compare characters column by column
        Time Complexity: O(S)
        Space Complexity: O(1)
        */
        if (strs.empty()) return "";
        for (int i = 0; i < (int)strs[0].size(); i++) {
            char c = strs[0][i];
            for (int j = 1; j < (int)strs.size(); j++) {
                if (i >= (int)strs[j].size() || strs[j][i] != c)
                    return strs[0].substr(0, i);
            }
        }
        return strs[0];
    }

    string LCP_Sorting(vector<string>& strs) {
        /*
        Sort and compare only first and last strings
        Time Complexity: O(n * m * log n) for sorting
        Space Complexity: O(1)
        */
        if (strs.empty()) return "";
        sort(strs.begin(), strs.end());
        string first = strs[0], last = strs.back();
        int i = 0;
        while (i < (int)min(first.size(), last.size()) && first[i] == last[i]) i++;
        return first.substr(0, i);
    }
};

void Test_Longest_Common_Prefix() {
    Solution sol;
    vector<vector<string>> tests = {
        {"flower", "flow", "flight"},
        {"dog", "racecar", "car"},
        {"interspecies", "interstellar", "interstate"},
        {"a"},
        {"", "abc"}
    };

    for (auto strs : tests) {
        cout << "Input: ";
        for (auto& s : strs) cout << "\"" << s << "\" ";
        cout << endl;

        vector<string> copy1 = strs, copy2 = strs, copy3 = strs;
        cout << "Horizontal: \"" << sol.LCP_Horizontal_Scan(copy1) << "\"" << endl;
        cout << "Vertical: \"" << sol.LCP_Vertical_Scan(copy2) << "\"" << endl;
        cout << "Sorting: \"" << sol.LCP_Sorting(copy3) << "\"" << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Longest_Common_Prefix();
    return 0;
}
