/*
Problem: KMP Algorithm for Pattern Searching / Longest Prefix Suffix
URL: https://practice.geeksforgeeks.org/problems/longest-prefix-suffix2527/1
URL: https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/

Problem Statement:
1. Given a string, find the length of the longest proper prefix which is also a suffix.
2. Given a text and pattern, find all occurrences of pattern in text using KMP algorithm.

Sample Input/Output:
Input: s = "abab"
Output: LPS = 2 ("ab" is both prefix and suffix)

Input: txt = "ABABDABACDABABCABAB", pat = "ABABCABAB"
Output: Pattern found at index 9
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Compute_LPS_Array(string pat) {
        /*
        Build LPS (Longest Proper Prefix which is also Suffix) array
        Time Complexity: O(m)
        Space Complexity: O(m)
        */
        int m = pat.size();
        vector<int> lps(m, 0);
        int len = 0, i = 1;

        while (i < m) {
            if (pat[i] == pat[len]) {
                len++;
                lps[i] = len;
                i++;
            } else {
                if (len != 0)
                    len = lps[len - 1];
                else {
                    lps[i] = 0;
                    i++;
                }
            }
        }
        return lps;
    }

    vector<int> KMP_Search(string txt, string pat) {
        /*
        KMP pattern searching using LPS array
        Time Complexity: O(n + m)
        Space Complexity: O(m)
        */
        vector<int> result;
        int N = txt.size(), M = pat.size();
        vector<int> lps = Compute_LPS_Array(pat);

        int i = 0, j = 0;
        while (i < N) {
            if (pat[j] == txt[i]) {
                i++;
                j++;
            }
            if (j == M) {
                result.push_back(i - j);
                j = lps[j - 1];
            } else if (i < N && pat[j] != txt[i]) {
                if (j != 0) j = lps[j - 1];
                else i++;
            }
        }
        return result;
    }

    int Longest_Prefix_Suffix(string s) {
        /*
        Find length of longest proper prefix which is also suffix
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> lps = Compute_LPS_Array(s);
        return lps[s.size() - 1];
    }
};

void Test_KMP_Algorithm() {
    Solution sol;

    vector<string> lps_tests = {"abab", "aabaaab", "aaaa", "abcab", "abc"};
    cout << "=== Longest Prefix Suffix ===" << endl;
    for (auto& s : lps_tests) {
        cout << "Input: " << s << " -> LPS: " << sol.Longest_Prefix_Suffix(s) << endl;
    }
    cout << string(50, '-') << endl;

    struct TestCase { string txt, pat; };
    vector<TestCase> kmp_tests = {
        {"ABABDABACDABABCABAB", "ABABCABAB"},
        {"AABAACAADAABAABA", "AABA"},
        {"GEEKS FOR GEEKS", "GEEK"},
        {"AAAAAA", "AA"}
    };

    cout << "=== KMP Search ===" << endl;
    for (auto& t : kmp_tests) {
        cout << "Text: \"" << t.txt << "\", Pattern: \"" << t.pat << "\"" << endl;
        auto result = sol.KMP_Search(t.txt, t.pat);
        cout << "Found at: ";
        for (int idx : result) cout << idx << " ";
        cout << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_KMP_Algorithm();
    return 0;
}
