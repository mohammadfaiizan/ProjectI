/*
Problem: Smallest Sum Contiguous Subarray
URL: https://www.geeksforgeeks.org/smallest-sum-contiguous-subarray/

Problem Statement:
Given an array containing n integers. The problem is to find the sum of the elements of the contiguous subarray having the smallest sum.

Sample Input/Output:
Input: [3,-4,2,-3,-1,7,-5]
Output: -6
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Subarray_Kadane(vector<int>& arr) {
        /*
        Modified Kadane's Algorithm
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        int minSum = arr[0];
        int currentSum = arr[0];
        
        for (int i = 1; i < n; i++) {
            currentSum = min(arr[i], currentSum + arr[i]);
            minSum = min(minSum, currentSum);
        }
        
        return minSum;
    }
};

void Test_Min_Subarray_Kadane() {
    Solution solution;
    vector<int> arr = {3, -4, 2, -3, -1, 7, -5};
    assert(solution.Min_Subarray_Kadane(arr) == -6);
}

int main() {
    Test_Min_Subarray_Kadane();
    return 0;
}
