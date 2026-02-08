/*
Problem: Rearrange Characters Such That No Two Adjacent Are Same
URL: https://leetcode.com/problems/reorganize-string/

Problem Statement:
Given a string s, rearrange the characters so that no two adjacent characters
are the same. Return any valid rearrangement or empty string if not possible.

Sample Input/Output:
Input: "aab"
Output: "aba"

Input: "aaab"
Output: "" (not possible)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Rearrange_Max_Heap(string s) {
        /*
        Using max heap - always place most frequent char first
        Time Complexity: O(n log k) where k = unique chars
        Space Complexity: O(k)
        */
        unordered_map<char, int> freq;
        for (char c : s) freq[c]++;

        priority_queue<pair<int, char>> pq;
        for (auto& p : freq) pq.push({p.second, p.first});

        string result = "";
        pair<int, char> prev = {0, '#'};

        while (!pq.empty()) {
            auto curr = pq.top();
            pq.pop();
            result += curr.second;
            curr.first--;

            if (prev.first > 0) pq.push(prev);
            prev = curr;
        }

        return (int)result.size() == (int)s.size() ? result : "";
    }

    string Rearrange_Fill_Even_Odd(string s) {
        /*
        Count frequencies, fill even positions first then odd
        Time Complexity: O(n)
        Space Complexity: O(k)
        */
        int n = s.size();
        unordered_map<char, int> freq;
        char maxChar = s[0];
        int maxFreq = 0;

        for (char c : s) {
            freq[c]++;
            if (freq[c] > maxFreq) {
                maxFreq = freq[c];
                maxChar = c;
            }
        }

        if (maxFreq > (n + 1) / 2) return "";

        string result(n, ' ');
        int idx = 0;

        while (freq[maxChar] > 0) {
            result[idx] = maxChar;
            idx += 2;
            freq[maxChar]--;
        }

        for (auto& p : freq) {
            while (p.second > 0) {
                if (idx >= n) idx = 1;
                result[idx] = p.first;
                idx += 2;
                p.second--;
            }
        }
        return result;
    }

    string Rearrange_Sorting(string s) {
        /*
        Sort by frequency then interleave
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        int n = s.size();
        unordered_map<char, int> freq;
        for (char c : s) freq[c]++;

        vector<pair<int, char>> sorted_freq;
        for (auto& p : freq) sorted_freq.push_back({p.second, p.first});
        sort(sorted_freq.rbegin(), sorted_freq.rend());

        if (sorted_freq[0].first > (n + 1) / 2) return "";

        string result(n, ' ');
        int idx = 0;
        for (auto& p : sorted_freq) {
            for (int i = 0; i < p.first; i++) {
                if (idx >= n) idx = 1;
                result[idx] = p.second;
                idx += 2;
            }
        }
        return result;
    }
};

void Test_Rearrange_Adjacent() {
    Solution sol;
    vector<string> tests = {"aab", "aaab", "aabb", "aaabbc", "a", "abcdef"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        string r1 = sol.Rearrange_Max_Heap(s);
        cout << "Max Heap: " << (r1.empty() ? "Not Possible" : r1) << endl;
        string r2 = sol.Rearrange_Fill_Even_Odd(s);
        cout << "Fill Even/Odd: " << (r2.empty() ? "Not Possible" : r2) << endl;
        string r3 = sol.Rearrange_Sorting(s);
        cout << "Sorting: " << (r3.empty() ? "Not Possible" : r3) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Rearrange_Adjacent();
    return 0;
}
