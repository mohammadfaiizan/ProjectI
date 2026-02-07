/*
Problem: Zero Sum Subarrays
URL: https://practice.geeksforgeeks.org/problems/zero-sum-subarrays1825/1

Problem Statement:
You are given an array arr[] of size n. Find the total count of sub-arrays having their sum equal to 0.

Sample Input/Output:
Input: n = 6, arr[] = {0, 0, 5, 5, 0, 0}
Output: 6

Input: n = 10, arr[] = {6, -1, -3, 4, -2, 2, 4, 6, -12, -7}
Output: 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Subarray_Sum_Zero_Prefix_HashMap(vector<int>& arr, int n) {
        /*
        Use prefix sum and hash map to count subarrays with zero sum
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_map<long long, int> prefixSum;
        long long sum = 0;
        long long count = 0;
        
        prefixSum[0] = 1;
        
        for (int i = 0; i < n; i++) {
            sum += arr[i];
            if (prefixSum.find(sum) != prefixSum.end()) {
                count += prefixSum[sum];
            }
            prefixSum[sum]++;
        }
        
        return count;
    }

    long long Subarray_Sum_Zero_Brute_Force(vector<int>& arr, int n) {
        /*
        Check all possible subarrays and count those with zero sum
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        long long count = 0;
        
        for (int i = 0; i < n; i++) {
            long long sum = 0;
            for (int j = i; j < n; j++) {
                sum += arr[j];
                if (sum == 0) {
                    count++;
                }
            }
        }
        
        return count;
    }
};

void Test_Subarray_With_Sum_Zero() {
    Solution sol;
    vector<vector<int>> tests = {
        {0, 0, 5, 5, 0, 0},
        {6, -1, -3, 4, -2, 2, 4, 6, -12, -7},
        {1, -1, 1, -1},
        {0},
        {1, 2, 3}
    };

    for (auto& arr : tests) {
        int n = arr.size();
        cout << "Array: ";
        for (int num : arr) cout << num << " ";
        cout << endl;
        
        long long res1 = sol.Subarray_Sum_Zero_Prefix_HashMap(arr, n);
        long long res2 = sol.Subarray_Sum_Zero_Brute_Force(arr, n);
        
        cout << "Prefix Sum + HashMap: " << res1 << endl;
        cout << "Brute Force: " << res2 << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Subarray_With_Sum_Zero();
    return 0;
}
