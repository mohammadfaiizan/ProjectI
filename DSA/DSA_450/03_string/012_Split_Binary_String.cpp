/*
Problem: Split Binary String into Substrings with Equal 0s and 1s
URL: https://www.geeksforgeeks.org/split-the-binary-string-into-substrings-with-equal-number-of-0s-and-1s/

Problem Statement:
Given a binary string, split it into maximum number of substrings such that
each substring contains equal number of 0s and 1s. Return -1 if not possible.

Sample Input/Output:
Input: "0100110101"
Output: 4

Input: "0111100010"
Output: 3

Input: "0"
Output: -1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Split_Binary_Counter(string s) {
        /*
        Count 0s and 1s, increment result when counts are equal
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int count0 = 0, count1 = 0, cnt = 0;
        int n = s.size();
        for (int i = 0; i < n; i++) {
            if (s[i] == '0') count0++;
            else count1++;
            if (count0 == count1) cnt++;
        }
        if (count0 != count1) return -1;
        return cnt;
    }

    int Split_Binary_Prefix_Sum(string s) {
        /*
        Using prefix sum: treat 0 as -1 and 1 as +1
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = s.size();
        int sum = 0, count = 0;
        for (int i = 0; i < n; i++) {
            sum += (s[i] == '1') ? 1 : -1;
            if (sum == 0) count++;
        }
        if (sum != 0) return -1;
        return count;
    }
};

void Test_Split_Binary_String() {
    Solution sol;
    vector<string> tests = {"0100110101", "0111100010", "0", "01", "0011", "000111"};

    for (auto& s : tests) {
        cout << "Input: " << s << endl;
        cout << "Counter: " << sol.Split_Binary_Counter(s) << endl;
        cout << "Prefix Sum: " << sol.Split_Binary_Prefix_Sum(s) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Split_Binary_String();
    return 0;
}
