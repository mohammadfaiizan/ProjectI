/*
Problem: Permutations of a Given String
URL: https://practice.geeksforgeeks.org/problems/permutations-of-a-given-string2041/1

Problem Statement:
Given a string S, find all permutations of the string and return them sorted.

Sample Input/Output:
Input: S = "ABC"
Output: ABC ACB BAC BCA CAB CBA

Input: S = "AB"
Output: AB BA
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Permutations_Swap(string& s, int l, int r, vector<string>& result) {
        /*
        Swap-based recursion
        Time Complexity: O(n! * n)
        Space Complexity: O(n) recursion stack
        */
        if (l == r) {
            result.push_back(s);
            return;
        }
        for (int i = l; i <= r; i++) {
            swap(s[l], s[i]);
            Permutations_Swap(s, l + 1, r, result);
            swap(s[l], s[i]);
        }
    }

    vector<string> Permutations_STL(string s) {
        /*
        Using next_permutation from STL
        Time Complexity: O(n! * n)
        Space Complexity: O(n!)
        */
        vector<string> result;
        sort(s.begin(), s.end());
        do {
            result.push_back(s);
        } while (next_permutation(s.begin(), s.end()));
        return result;
    }

    void Permutations_Backtrack(string& s, vector<bool>& used, string& curr, vector<string>& result) {
        /*
        Backtracking with visited array
        Time Complexity: O(n! * n)
        Space Complexity: O(n)
        */
        if ((int)curr.size() == (int)s.size()) {
            result.push_back(curr);
            return;
        }
        for (int i = 0; i < (int)s.size(); i++) {
            if (used[i]) continue;
            used[i] = true;
            curr += s[i];
            Permutations_Backtrack(s, used, curr, result);
            curr.pop_back();
            used[i] = false;
        }
    }
};

void Test_Print_All_Permutations() {
    Solution sol;
    vector<string> tests = {"ABC", "AB", "A"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;

        string temp = s;
        vector<string> r1;
        sol.Permutations_Swap(temp, 0, s.size() - 1, r1);
        sort(r1.begin(), r1.end());
        cout << "Swap: ";
        for (auto& x : r1) cout << x << " ";
        cout << endl;

        auto r2 = sol.Permutations_STL(s);
        cout << "STL: ";
        for (auto& x : r2) cout << x << " ";
        cout << endl;

        vector<bool> used(s.size(), false);
        string curr = "";
        vector<string> r3;
        string sorted_s = s;
        sort(sorted_s.begin(), sorted_s.end());
        sol.Permutations_Backtrack(sorted_s, used, curr, r3);
        cout << "Backtrack: ";
        for (auto& x : r3) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Print_All_Permutations();
    return 0;
}
