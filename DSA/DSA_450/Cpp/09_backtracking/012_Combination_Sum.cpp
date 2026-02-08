/*
Problem: Combination Sum
URL: https://practice.geeksforgeeks.org/problems/combination-sum-1587115620/1

Problem Statement:
Given an array of distinct integers and a target sum, find all unique combinations that sum to the target. The same number can be used unlimited times.

Sample Input/Output:
Input: arr = [2, 3, 6, 7], target = 7
Output: [[2, 2, 3], [7]]
Explanation: 2+2+3=7 and 7=7
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> Combination_Sum_Backtracking(vector<int>& arr, int target) {
        /*
        Backtracking with index tracking
        Time Complexity: O(2^t * k) where t=target, k=avg combo length
        Space Complexity: O(k)
        */
        sort(arr.begin(), arr.end());
        vector<vector<int>> result;
        vector<int> current;
        
        function<void(int, int)> backtrack = [&](int idx, int remaining) {
            if (remaining == 0) {
                result.push_back(current);
                return;
            }
            
            for (int i = idx; i < arr.size(); i++) {
                if (arr[i] > remaining) break;
                current.push_back(arr[i]);
                backtrack(i, remaining - arr[i]);
                current.pop_back();
            }
        };
        
        backtrack(0, target);
        return result;
    }
};

void Test_Combination_Sum() {
    Solution solution;
    vector<int> arr = {2, 3, 6, 7};
    int target = 7;
    vector<vector<int>> result = solution.Combination_Sum_Backtracking(arr, target);
    cout << "Combinations for target " << target << ":" << endl;
    for (auto& combo : result) {
        for (int num : combo) {
            cout << num << " ";
        }
        cout << endl;
    }
}

int main() {
    Test_Combination_Sum();
    return 0;
}
