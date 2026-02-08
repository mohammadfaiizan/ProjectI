/*
Problem: Maximum Number in K Swaps
URL: https://practice.geeksforgeeks.org/problems/largest-number-in-k-swaps-1587115620/1

Problem Statement:
Given a number as a string and K swaps allowed, find the maximum possible number.

Sample Input/Output:
Input: str = "1234567", K = 4
Output: "7654321"
Explanation: After 4 swaps, we get maximum number
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    string Max_Number_K_Swaps_Backtracking(string str, int k) {
        /*
        Backtracking all pairs
        Time Complexity: O((n^2)^k)
        Space Complexity: O(n)
        */
        string max_str = str;
        
        function<void(string&, int, int)> backtrack = [&](string& s, int swaps_left, int start) {
            if (swaps_left == 0 || start >= s.length()) {
                if (s > max_str) {
                    max_str = s;
                }
                return;
            }
            
            for (int i = start; i < s.length(); i++) {
                for (int j = i + 1; j < s.length(); j++) {
                    if (s[j] > s[i]) {
                        swap(s[i], s[j]);
                        backtrack(s, swaps_left - 1, start + 1);
                        swap(s[i], s[j]);
                    }
                }
            }
            
            if (s > max_str) {
                max_str = s;
            }
        };
        
        string temp = str;
        backtrack(temp, k, 0);
        return max_str;
    }
    
    string Max_Number_K_Swaps_Optimized(string str, int k) {
        /*
        Optimized find max digit first
        Time Complexity: O(n^k)
        Space Complexity: O(n)
        */
        string max_str = str;
        
        function<void(string&, int, int)> backtrack = [&](string& s, int swaps_left, int idx) {
            if (swaps_left == 0 || idx >= s.length()) {
                if (s > max_str) {
                    max_str = s;
                }
                return;
            }
            
            char max_char = s[idx];
            for (int i = idx + 1; i < s.length(); i++) {
                if (s[i] > max_char) {
                    max_char = s[i];
                }
            }
            
            if (max_char == s[idx]) {
                backtrack(s, swaps_left, idx + 1);
            } else {
                for (int i = idx + 1; i < s.length(); i++) {
                    if (s[i] == max_char) {
                        swap(s[idx], s[i]);
                        backtrack(s, swaps_left - 1, idx + 1);
                        swap(s[idx], s[i]);
                    }
                }
            }
        };
        
        string temp = str;
        backtrack(temp, k, 0);
        return max_str;
    }
};

void Test_Max_Number_K_Swaps() {
    Solution solution;
    string str = "1234567";
    int k = 4;
    cout << "Original: " << str << endl;
    cout << "Backtracking Approach: " << solution.Max_Number_K_Swaps_Backtracking(str, k) << endl;
    cout << "Optimized Approach: " << solution.Max_Number_K_Swaps_Optimized(str, k) << endl;
}

int main() {
    Test_Max_Number_K_Swaps();
    return 0;
}
