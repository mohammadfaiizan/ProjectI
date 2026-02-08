/*
Problem: Second Most Repeated String in a Sequence
URL: https://practice.geeksforgeeks.org/problems/second-most-repeated-string-in-a-sequence0534/1

Problem Statement:
Given a sequence of strings, find the second most repeated string in the sequence.

Sample Input/Output:
Input: arr = ["aaa", "bbb", "ccc", "bbb", "aaa", "aaa"]
Output: "bbb"

Input: arr = ["abc", "abc", "xyz", "xyz", "xyz"]
Output: "abc"
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Second_Most_Repeated_Map(vector<string>& arr) {
        /*
        Using map to count frequencies, then find second max
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_map<string, int> mp;
        int mx = 0;
        for (auto& s : arr) {
            mp[s]++;
            mx = max(mx, mp[s]);
        }

        int secondMax = 0;
        string ans;
        for (auto& p : mp) {
            if (p.second != mx && p.second > secondMax) {
                secondMax = p.second;
                ans = p.first;
            }
        }
        return ans;
    }

    string Second_Most_Repeated_Sorting(vector<string>& arr) {
        /*
        Sort, count frequencies, sort by frequency
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        unordered_map<string, int> mp;
        for (auto& s : arr) mp[s]++;

        vector<pair<int, string>> freqs;
        for (auto& p : mp) freqs.push_back({p.second, p.first});
        sort(freqs.rbegin(), freqs.rend());

        if (freqs.size() >= 2) return freqs[1].second;
        return "";
    }

    string Second_Most_Repeated_Heap(vector<string>& arr) {
        /*
        Using max heap to find top 2
        Time Complexity: O(n + k log k) where k = unique strings
        Space Complexity: O(n)
        */
        unordered_map<string, int> mp;
        for (auto& s : arr) mp[s]++;

        priority_queue<pair<int, string>> pq;
        for (auto& p : mp) pq.push({p.second, p.first});

        if (!pq.empty()) pq.pop();
        if (!pq.empty()) return pq.top().second;
        return "";
    }
};

void Test_Second_Most_Repeated() {
    Solution sol;
    vector<vector<string>> tests = {
        {"aaa", "bbb", "ccc", "bbb", "aaa", "aaa"},
        {"abc", "abc", "xyz", "xyz", "xyz"},
        {"one", "two", "three", "one", "two", "one", "two"}
    };

    for (auto& arr : tests) {
        cout << "Input: ";
        for (auto& s : arr) cout << s << " ";
        cout << endl;
        cout << "Map: " << sol.Second_Most_Repeated_Map(arr) << endl;
        cout << "Sorting: " << sol.Second_Most_Repeated_Sorting(arr) << endl;
        cout << "Heap: " << sol.Second_Most_Repeated_Heap(arr) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Second_Most_Repeated();
    return 0;
}
