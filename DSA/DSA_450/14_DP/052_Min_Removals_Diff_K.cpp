/*
Problem: Minimum Removals to Make Max-Min <= K
URL: https://www.geeksforgeeks.org/minimum-removals-array-make-max-min-k/

Problem Statement:
Given an array and a number k, find the minimum number of elements to remove so that the difference between maximum and minimum remaining elements is at most k.

Sample Input/Output:
Input: arr = [1,3,4,9,10,11,12,17,20], k = 4
Output: 5
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Remove_Memo(vector<int>& arr, int i, int j, int k, vector<vector<int>>& dp) {
        /*
        Memoization approach
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        if (i >= j) return 0;
        if (arr[j] - arr[i] <= k) return 0;
        if (dp[i][j] != -1) return dp[i][j];
        
        dp[i][j] = 1 + min(Min_Remove_Memo(arr, i + 1, j, k, dp),
                          Min_Remove_Memo(arr, i, j - 1, k, dp));
        return dp[i][j];
    }
    
    int Min_Remove_Binary_Search(vector<int>& arr, int k) {
        /*
        Binary search approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        sort(arr.begin(), arr.end());
        
        int minRemovals = n - 1;
        
        for (int i = 0; i < n; i++) {
            int left = i, right = n - 1;
            int maxIdx = i;
            
            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (arr[mid] - arr[i] <= k) {
                    maxIdx = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            
            minRemovals = min(minRemovals, n - (maxIdx - i + 1));
        }
        
        return minRemovals;
    }
};

void Test_Min_Remove() {
    Solution solution;
    
    vector<int> arr = {1, 3, 4, 9, 10, 11, 12, 17, 20};
    int k = 4;
    vector<vector<int>> dp(arr.size(), vector<int>(arr.size(), -1));
    
    cout << "Memo: ";
    for (int x : arr) cout << x << " ";
    cout << ", k=" << k << " -> " 
         << solution.Min_Remove_Memo(arr, 0, arr.size() - 1, k, dp) << endl;
    
    vector<int> arr2 = {1, 3, 4, 9, 10, 11, 12, 17, 20};
    cout << "Binary Search: ";
    for (int x : arr2) cout << x << " ";
    cout << ", k=" << k << " -> " 
         << solution.Min_Remove_Binary_Search(arr2, k) << endl;
}

int main() {
    Test_Min_Remove();
    return 0;
}
