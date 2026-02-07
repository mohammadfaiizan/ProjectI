/*
Problem: Print All Duplicates in a String
URL: https://www.geeksforgeeks.org/print-all-the-duplicates-in-the-input-string/

Problem Statement:
Given a string, find all characters that occur more than once and print them
along with their count.

Sample Input/Output:
Input: "geeksforgeeks"
Output: e, count = 4; g, count = 2; k, count = 2; s, count = 2
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Duplicate_Chars_Array(string s) {
        /*
        Using frequency array of size 256
        Time Complexity: O(n)
        Space Complexity: O(1) - constant 256
        */
        int freq[256] = {0};
        for (char c : s) freq[(int)c]++;
        for (int i = 0; i < 256; i++) {
            if (freq[i] > 1) {
                cout << (char)i << ", count = " << freq[i] << endl;
            }
        }
    }

    unordered_map<char, int> Duplicate_Chars_Map(string s) {
        /*
        Using unordered_map
        Time Complexity: O(n)
        Space Complexity: O(k) where k = unique chars
        */
        unordered_map<char, int> mp;
        for (char c : s) mp[c]++;
        unordered_map<char, int> result;
        for (auto& p : mp) {
            if (p.second > 1) result[p.first] = p.second;
        }
        return result;
    }

    void Duplicate_Chars_Sorting(string s) {
        /*
        Sort then scan adjacent
        Time Complexity: O(n log n)
        Space Complexity: O(1) if in-place sort
        */
        sort(s.begin(), s.end());
        int n = s.size();
        int i = 0;
        while (i < n) {
            int count = 1;
            while (i + count < n && s[i] == s[i + count]) count++;
            if (count > 1) cout << s[i] << ", count = " << count << endl;
            i += count;
        }
    }
};

void Test_Duplicate_Characters() {
    Solution sol;
    vector<string> tests = {"geeksforgeeks", "hello", "aabbcc", "abcdef"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;

        cout << "Array Method:" << endl;
        sol.Duplicate_Chars_Array(s);

        cout << "Map Method:" << endl;
        auto res = sol.Duplicate_Chars_Map(s);
        for (auto& p : res) cout << p.first << ", count = " << p.second << endl;

        cout << "Sorting Method:" << endl;
        sol.Duplicate_Chars_Sorting(s);

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Duplicate_Characters();
    return 0;
}
