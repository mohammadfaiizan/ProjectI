/*
Problem: Roman Numeral to Integer
URL: https://practice.geeksforgeeks.org/problems/roman-number-to-integer3201/1

Problem Statement:
Given a string in Roman numeral format, convert it to an integer.

Sample Input/Output:
Input: "III"
Output: 3

Input: "MCMXCIV"
Output: 1994

Input: "IX"
Output: 9
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Roman_To_Int_Right_To_Left(string s) {
        /*
        Iterate right to left, subtract if current < next
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        auto val = [](char c) -> int {
            if (c == 'M') return 1000;
            if (c == 'D') return 500;
            if (c == 'C') return 100;
            if (c == 'L') return 50;
            if (c == 'X') return 10;
            if (c == 'V') return 5;
            if (c == 'I') return 1;
            return 0;
        };

        int m = s.size();
        int ans = 0;
        for (int i = m - 1; i >= 0; i--) {
            if (i > 0 && val(s[i - 1]) < val(s[i])) {
                ans += val(s[i]) - val(s[i - 1]);
                i--;
            } else {
                ans += val(s[i]);
            }
        }
        return ans;
    }

    int Roman_To_Int_Left_To_Right(string s) {
        /*
        Iterate left to right, add or subtract based on next value
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        unordered_map<char, int> mp = {
            {'I',1}, {'V',5}, {'X',10}, {'L',50},
            {'C',100}, {'D',500}, {'M',1000}
        };

        int result = 0;
        for (int i = 0; i < (int)s.size(); i++) {
            if (i + 1 < (int)s.size() && mp[s[i]] < mp[s[i + 1]]) {
                result += mp[s[i + 1]] - mp[s[i]];
                i++;
            } else {
                result += mp[s[i]];
            }
        }
        return result;
    }

    int Roman_To_Int_Prev_Track(string s) {
        /*
        Track previous value, subtract if prev < current
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        unordered_map<char, int> mp = {
            {'I',1}, {'V',5}, {'X',10}, {'L',50},
            {'C',100}, {'D',500}, {'M',1000}
        };

        int m = s.size();
        int ans = mp[s[m - 1]];
        int prev = mp[s[m - 1]];
        for (int i = m - 2; i >= 0; i--) {
            int curr = mp[s[i]];
            if (curr >= prev) ans += curr;
            else ans -= curr;
            prev = curr;
        }
        return ans;
    }
};

void Test_Roman_To_Integer() {
    Solution sol;
    vector<string> tests = {"III", "IV", "IX", "LVIII", "MCMXCIV", "MMMDCCXLIX", "CDXLIV"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        cout << "Right to Left: " << sol.Roman_To_Int_Right_To_Left(s) << endl;
        cout << "Left to Right: " << sol.Roman_To_Int_Left_To_Right(s) << endl;
        cout << "Prev Track: " << sol.Roman_To_Int_Prev_Track(s) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Roman_To_Integer();
    return 0;
}
