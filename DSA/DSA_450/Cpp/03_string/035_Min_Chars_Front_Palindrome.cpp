/*
Problem: Minimum Characters to Add at Front to Make String Palindrome
URL: https://www.geeksforgeeks.org/minimum-characters-added-front-make-string-palindrome/

Problem Statement:
Given a string str, find the minimum number of characters to be added at the
front to make the string a palindrome.

Sample Input/Output:
Input: "AACECAAAA"
Output: 2

Input: "ABC"
Output: 2
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Chars_LPS(string str) {
        /*
        Using KMP LPS array on str + "$" + reverse(str)
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        string rev = str;
        reverse(rev.begin(), rev.end());
        string concat = str + "$" + rev;
        int n = concat.size();

        vector<int> lps(n, 0);
        int len = 0, i = 1;
        while (i < n) {
            if (concat[i] == concat[len]) {
                len++;
                lps[i] = len;
                i++;
            } else {
                if (len != 0) len = lps[len - 1];
                else { lps[i] = 0; i++; }
            }
        }
        return str.size() - lps[n - 1];
    }

    int Min_Chars_Brute(string str) {
        /*
        Keep removing last char until palindrome
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        int cnt = 0;
        string s = str;
        while (!s.empty()) {
            string rev = s;
            reverse(rev.begin(), rev.end());
            if (s == rev) break;
            s.pop_back();
            cnt++;
        }
        return cnt;
    }

    int Min_Chars_Two_Pointer(string str) {
        /*
        Two pointer - find longest palindromic prefix
        Time Complexity: O(n^2) worst case
        Space Complexity: O(n)
        */
        int n = str.size();
        int i = 0, j = n - 1;
        int suffixEnd = n - 1;

        while (i < j) {
            if (str[i] == str[j]) {
                i++;
                j--;
            } else {
                i = 0;
                suffixEnd--;
                j = suffixEnd;
            }
        }
        return n - suffixEnd - 1;
    }
};

void Test_Min_Chars_Front_Palindrome() {
    Solution sol;
    vector<string> tests = {"AACECAAAA", "ABC", "BABABAA", "a", "ab", "aaa"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        cout << "LPS: " << sol.Min_Chars_LPS(s) << endl;
        cout << "Brute: " << sol.Min_Chars_Brute(s) << endl;
        cout << "Two Pointer: " << sol.Min_Chars_Two_Pointer(s) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Min_Chars_Front_Palindrome();
    return 0;
}
