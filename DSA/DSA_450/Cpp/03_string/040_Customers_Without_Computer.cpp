/*
Problem: Number of Customers Who Could Not Get a Computer
URL: https://www.geeksforgeeks.org/function-to-find-number-of-customers-who-could-not-get-a-computer/

Problem Statement:
Given N computers in a cafe and a string of uppercase characters representing
customers entering/leaving. First occurrence means entering, second means leaving.
Find the number of customers who could not get a computer.

Sample Input/Output:
Input: N = 2, seq = "ABCBCA"
Output: 1 (C couldn't get a computer)

Input: N = 2, seq = "ABCBCADEED"
Output: 2
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Disappointed_Array(int capacity, string s) {
        /*
        Using array to track customer state
        Time Complexity: O(n)
        Space Complexity: O(1) - fixed 26 chars
        */
        int cnt[26] = {0};
        int occupied = 0, ans = 0;

        for (int i = 0; i < (int)s.size(); i++) {
            int idx = s[i] - 'A';
            if (cnt[idx] == 0) {
                if (occupied < capacity) {
                    cnt[idx] = 1;
                    occupied++;
                } else {
                    ans++;
                    cnt[idx] = -1;
                }
            } else if (cnt[idx] == 1) {
                cnt[idx] = 0;
                occupied--;
            }
        }
        return ans;
    }

    int Disappointed_Map(int capacity, string s) {
        /*
        Using unordered_map for tracking
        Time Complexity: O(n)
        Space Complexity: O(k) where k = unique customers
        */
        unordered_map<char, int> state;
        int occupied = 0, ans = 0;

        for (char c : s) {
            if (state.find(c) == state.end() || state[c] == 0) {
                if (occupied < capacity) {
                    state[c] = 1;
                    occupied++;
                } else {
                    ans++;
                    state[c] = -1;
                }
            } else if (state[c] == 1) {
                state[c] = 0;
                occupied--;
            }
        }
        return ans;
    }
};

void Test_Customers_Without_Computer() {
    Solution sol;
    struct TestCase { int n; string seq; };
    vector<TestCase> tests = {
        {2, "ABCBCA"},
        {2, "ABCBCADEED"},
        {3, "ABCABC"},
        {1, "ABAB"},
        {3, "ABCDABCD"}
    };

    for (auto& t : tests) {
        cout << "N=" << t.n << ", Seq: " << t.seq << endl;
        cout << "Array: " << sol.Disappointed_Array(t.n, t.seq) << endl;
        cout << "Map: " << sol.Disappointed_Map(t.n, t.seq) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Customers_Without_Computer();
    return 0;
}
