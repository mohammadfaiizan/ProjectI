/*
Problem: Maximum Difference of Zeros and Ones in Binary String
URL: https://practice.geeksforgeeks.org/problems/maximum-difference-of-zeros-and-ones-in-binary-string4111/1

Problem Statement:
Given a binary string S of 0s and 1s. The task is to find the maximum difference of the number of 0s and the number of 1s (number of 0s - number of 1s) in the substrings of a string.

Sample Input/Output:
Input: "11000010001"
Output: 6
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Diff_Kadane(string& s) {
        /*
        Kadane's algorithm approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = s.length();
        int max_diff = -1;
        int curr_diff = 0;
        for (int i = 0; i < n; i++) {
            int val = (s[i] == '0') ? 1 : -1;
            curr_diff += val;
            if (curr_diff < 0) curr_diff = 0;
            max_diff = max(max_diff, curr_diff);
        }
        return max_diff;
    }
};

void Test_Max_Diff() {
    Solution solution;
    string s = "11000010001";
    cout << "Max Difference: " << solution.Max_Diff_Kadane(s) << endl;
}

int main() {
    Test_Max_Diff();
    return 0;
}
