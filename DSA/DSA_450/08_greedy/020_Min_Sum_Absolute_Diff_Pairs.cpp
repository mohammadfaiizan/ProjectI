/*
Problem: Minimum Sum Absolute Difference Pairs
URL: https://www.geeksforgeeks.org/minimum-sum-absolute-difference-pairs-two-arrays/

Problem Statement:
Given two arrays, pair elements to minimize sum of absolute differences.

Sample Input/Output:
Input: a[] = {4, 1, 8, 7}, b[] = {2, 3, 6, 5}
Output: 6
Explanation: Pair (1,2), (4,3), (7,5), (8,6). Sum = |1-2| + |4-3| + |7-5| + |8-6| = 1 + 1 + 2 + 2 = 6
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Min_Sum_Absolute_Diff_Pairs_Sort_Both(vector<int>& a, vector<int>& b) {
        /*
        Sort both + pair corresponding greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(a.begin(), a.end());
        sort(b.begin(), b.end());
        
        long long sum = 0;
        int n = a.size();
        
        for (int i = 0; i < n; i++) {
            sum += abs(a[i] - b[i]);
        }
        
        return sum;
    }
};

void Test_Min_Sum_Absolute_Diff_Pairs() {
    Solution solution;
    
    vector<int> a1 = {4, 1, 8, 7};
    vector<int> b1 = {2, 3, 6, 5};
    cout << "Test 1: " << solution.Min_Sum_Absolute_Diff_Pairs_Sort_Both(a1, b1) << endl;
    
    vector<int> a2 = {4, 1, 2};
    vector<int> b2 = {2, 4, 1};
    cout << "Test 2: " << solution.Min_Sum_Absolute_Diff_Pairs_Sort_Both(a2, b2) << endl;
    
    vector<int> a3 = {1, 2, 3};
    vector<int> b3 = {3, 2, 1};
    cout << "Test 3: " << solution.Min_Sum_Absolute_Diff_Pairs_Sort_Both(a3, b3) << endl;
}

int main() {
    Test_Min_Sum_Absolute_Diff_Pairs();
    return 0;
}
