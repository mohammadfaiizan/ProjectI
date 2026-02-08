/*
Problem: Maximum Sum Pairs with Specific Difference
URL: https://www.geeksforgeeks.org/maximum-sum-pairs-specific-difference/

Problem Statement:
Given an array of integers and a number k, find the maximum sum of pairs such that the difference between elements in each pair is less than k.

Sample Input/Output:
Input: arr = [3,5,10,15,17,12,9], k = 4
Output: 62
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Pairs_DP(vector<int>& arr, int k) {
        /*
        DP approach after sorting
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        int n = arr.size();
        sort(arr.begin(), arr.end());
        
        vector<int> dp(n);
        dp[0] = 0;
        
        for (int i = 1; i < n; i++) {
            dp[i] = dp[i - 1];
            
            if (arr[i] - arr[i - 1] < k) {
                int prev = (i >= 2) ? dp[i - 2] : 0;
                dp[i] = max(dp[i], prev + arr[i] + arr[i - 1]);
            }
        }
        
        return dp[n - 1];
    }
};

void Test_Max_Pairs() {
    Solution solution;
    
    vector<int> arr = {3, 5, 10, 15, 17, 12, 9};
    int k = 4;
    cout << "Array: ";
    for (int x : arr) cout << x << " ";
    cout << ", k=" << k << " -> " << solution.Max_Pairs_DP(arr, k) << endl;
}

int main() {
    Test_Max_Pairs();
    return 0;
}
