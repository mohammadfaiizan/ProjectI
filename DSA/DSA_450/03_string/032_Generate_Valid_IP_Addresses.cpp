/*
Problem: Generate All Valid IP Addresses
URL: https://www.geeksforgeeks.org/program-generate-possible-valid-ip-addresses-given-string/

Problem Statement:
Given a string containing only digits, restore it by returning all possible
valid IP address combinations.

Sample Input/Output:
Input: "25525511135"
Output: ["255.255.11.135", "255.255.111.35"]

Input: "0000"
Output: ["0.0.0.0"]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<string> Generate_IP_Backtrack(string s) {
        /*
        Backtracking - try placing dots at all valid positions
        Time Complexity: O(1) - max 27 combinations (3^3)
        Space Complexity: O(1)
        */
        vector<string> result;
        int n = s.size();
        if (n < 4 || n > 12) return result;
        Backtrack(s, 0, 0, "", result);
        return result;
    }

    vector<string> Generate_IP_Three_Loops(string s) {
        /*
        Three nested loops for three dot positions
        Time Complexity: O(n^3) but n <= 12
        Space Complexity: O(1) excluding result
        */
        vector<string> result;
        int n = s.size();
        if (n < 4 || n > 12) return result;

        for (int i = 1; i <= 3 && i < n; i++) {
            for (int j = i + 1; j <= i + 3 && j < n; j++) {
                for (int k = j + 1; k <= j + 3 && k < n; k++) {
                    string p1 = s.substr(0, i);
                    string p2 = s.substr(i, j - i);
                    string p3 = s.substr(j, k - j);
                    string p4 = s.substr(k);

                    if (Is_Valid_Part(p1) && Is_Valid_Part(p2) &&
                        Is_Valid_Part(p3) && Is_Valid_Part(p4)) {
                        result.push_back(p1 + "." + p2 + "." + p3 + "." + p4);
                    }
                }
            }
        }
        return result;
    }

private:
    bool Is_Valid_Part(string& s) {
        if (s.empty() || s.size() > 3) return false;
        if (s.size() > 1 && s[0] == '0') return false;
        int val = stoi(s);
        return val >= 0 && val <= 255;
    }

    void Backtrack(string& s, int start, int parts, string current, vector<string>& result) {
        if (parts == 4) {
            if (start == (int)s.size()) result.push_back(current);
            return;
        }

        for (int len = 1; len <= 3 && start + len <= (int)s.size(); len++) {
            string part = s.substr(start, len);
            if (!Is_Valid_Part(part)) continue;
            string next = current.empty() ? part : current + "." + part;
            Backtrack(s, start + len, parts + 1, next, result);
        }
    }
};

void Test_Generate_Valid_IP() {
    Solution sol;
    vector<string> tests = {"25525511135", "0000", "1111", "101023", "255255255255"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;

        auto r1 = sol.Generate_IP_Backtrack(s);
        cout << "Backtrack (" << r1.size() << "): ";
        for (auto& ip : r1) cout << ip << " | ";
        cout << endl;

        auto r2 = sol.Generate_IP_Three_Loops(s);
        cout << "Three Loops (" << r2.size() << "): ";
        for (auto& ip : r2) cout << ip << " | ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Generate_Valid_IP();
    return 0;
}
