/*
Problem: Maximum Sum Consecutive Difference Circular
URL: https://practice.geeksforgeeks.org/problems/swap-and-maximize5859/1

Problem Statement:
Rearrange circular array to maximize sum of |arr[i]-arr[i+1]|.

Sample Input/Output:
Input: arr[] = {4, 2, 1, 8}
Output: 18
Explanation: Rearrange to {1, 8, 2, 4}. Sum = |1-8| + |8-2| + |2-4| + |4-1| = 18
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Max_Sum_Consecutive_Diff_Circular_Sort(vector<int>& arr) {
        /*
        Sort + sum 2*(large-small) greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(arr.begin(), arr.end());
        int n = arr.size();
        long long sum = 0;
        
        for (int i = 0; i < n / 2; i++) {
            sum += 2 * (arr[n - 1 - i] - arr[i]);
        }
        
        return sum;
    }
};

void Test_Max_Sum_Consecutive_Diff_Circular() {
    Solution solution;
    
    vector<int> arr1 = {4, 2, 1, 8};
    cout << "Test 1: " << solution.Max_Sum_Consecutive_Diff_Circular_Sort(arr1) << endl;
    
    vector<int> arr2 = {1, 2, 3, 4, 5};
    cout << "Test 2: " << solution.Max_Sum_Consecutive_Diff_Circular_Sort(arr2) << endl;
    
    vector<int> arr3 = {10, 12};
    cout << "Test 3: " << solution.Max_Sum_Consecutive_Diff_Circular_Sort(arr3) << endl;
}

int main() {
    Test_Max_Sum_Consecutive_Diff_Circular();
    return 0;
}
