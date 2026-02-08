/*
Problem: Chocolate Distribution
URL: https://practice.geeksforgeeks.org/problems/chocolate-distribution-problem3825/1

Problem Statement:
Given N packets with chocolates and M children, distribute one packet each. Minimize difference between max and min chocolates given.

Sample Input/Output:
Input: packets[] = {7, 3, 2, 4, 9, 12, 56}, M = 3
Output: 2
Explanation: Distribute packets {2, 3, 4}. Difference = 4 - 2 = 2 (minimum).
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Chocolate_Distribution_Sort_Sliding_Window(vector<int>& packets, int M) {
        /*
        Sort + sliding window of size M greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(packets.begin(), packets.end());
        int n = packets.size();
        
        if (M > n) {
            return -1;
        }
        
        int min_diff = INT_MAX;
        
        for (int i = 0; i <= n - M; i++) {
            int diff = packets[i + M - 1] - packets[i];
            min_diff = min(min_diff, diff);
        }
        
        return min_diff;
    }
};

void Test_Chocolate_Distribution() {
    Solution solution;
    
    vector<int> packets1 = {7, 3, 2, 4, 9, 12, 56};
    cout << "Test 1: " << solution.Chocolate_Distribution_Sort_Sliding_Window(packets1, 3) << endl;
    
    vector<int> packets2 = {3, 4, 1, 9, 56, 7, 9, 12};
    cout << "Test 2: " << solution.Chocolate_Distribution_Sort_Sliding_Window(packets2, 5) << endl;
    
    vector<int> packets3 = {12, 4, 7, 9, 2, 23, 25, 41, 30, 40, 28, 42, 30, 44, 48, 43, 50};
    cout << "Test 3: " << solution.Chocolate_Distribution_Sort_Sliding_Window(packets3, 7) << endl;
}

int main() {
    Test_Chocolate_Distribution();
    return 0;
}
