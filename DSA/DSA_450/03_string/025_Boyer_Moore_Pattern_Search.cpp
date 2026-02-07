/*
Problem: Boyer Moore Algorithm for Pattern Searching
URL: https://www.geeksforgeeks.org/boyer-moore-algorithm-for-pattern-searching/

Problem Statement:
Given a text and a pattern, find all occurrences of the pattern in the text
using Boyer Moore's Bad Character Heuristic.

Sample Input/Output:
Input: txt = "ABAAABCD", pat = "ABC"
Output: Pattern found at shift 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Boyer_Moore_Bad_Char(string txt, string pat) {
        /*
        Boyer Moore Bad Character Heuristic
        Time Complexity: O(n/m) best, O(n*m) worst
        Space Complexity: O(256) = O(1)
        */
        int m = pat.size(), n = txt.size();
        vector<int> result;

        int badchar[256];
        fill(badchar, badchar + 256, -1);
        for (int i = 0; i < m; i++)
            badchar[(int)pat[i]] = i;

        int s = 0;
        while (s <= n - m) {
            int j = m - 1;
            while (j >= 0 && pat[j] == txt[s + j]) j--;

            if (j < 0) {
                result.push_back(s);
                s += (s + m < n) ? m - badchar[(int)txt[s + m]] : 1;
            } else {
                s += max(1, j - badchar[(int)txt[s + j]]);
            }
        }
        return result;
    }

    vector<int> Boyer_Moore_Simplified(string txt, string pat) {
        /*
        Simplified Boyer Moore with only bad character rule
        Time Complexity: O(n*m) worst case
        Space Complexity: O(256)
        */
        int n = txt.size(), m = pat.size();
        vector<int> result;
        unordered_map<char, int> lastOccurrence;
        for (int i = 0; i < m; i++)
            lastOccurrence[pat[i]] = i;

        int i = 0;
        while (i <= n - m) {
            int j = m - 1;
            while (j >= 0 && pat[j] == txt[i + j]) j--;

            if (j < 0) {
                result.push_back(i);
                i++;
            } else {
                int lo = lastOccurrence.count(txt[i + j]) ? lastOccurrence[txt[i + j]] : -1;
                i += max(1, j - lo);
            }
        }
        return result;
    }
};

void Test_Boyer_Moore() {
    Solution sol;
    struct TestCase { string txt, pat; };
    vector<TestCase> tests = {
        {"ABAAABCD", "ABC"},
        {"AABAACAADAABAABA", "AABA"},
        {"GEEKS FOR GEEKS", "GEEK"},
        {"ABABABABAB", "ABAB"}
    };

    for (auto& t : tests) {
        cout << "Text: \"" << t.txt << "\", Pattern: \"" << t.pat << "\"" << endl;

        auto r1 = sol.Boyer_Moore_Bad_Char(t.txt, t.pat);
        cout << "Bad Char: ";
        for (int idx : r1) cout << idx << " ";
        cout << endl;

        auto r2 = sol.Boyer_Moore_Simplified(t.txt, t.pat);
        cout << "Simplified: ";
        for (int idx : r2) cout << idx << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Boyer_Moore();
    return 0;
}
