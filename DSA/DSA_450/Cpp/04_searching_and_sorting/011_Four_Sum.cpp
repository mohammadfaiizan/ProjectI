/*
Problem: Find All Four Sum Numbers
URL: https://practice.geeksforgeeks.org/problems/find-all-four-sum-numbers1732/1

Problem Statement:
Given an array of integers arr[] and a target X, find all unique quadruplets (a, b, c, d) such that a + b + c + d = X.

Sample Input/Output:
Input: arr[] = {0,0,2,1,1}, X = 3
Output: 0 0 1 2

Input: arr[] = {10,2,3,4,5,7,8}, X = 23
Output: 2 3 7 8 2 4 5 10 3 5 7 8
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> Four_Sum_Sorting_Two_Pointer(vector<int>& arr, int n, int X) {
        /*
        Sort array and use nested loops with two pointers for last two elements
        Time Complexity: O(n^3)
        Space Complexity: O(1)
        */
        vector<vector<int>> result;
        sort(arr.begin(), arr.end());
        
        for (int i = 0; i < n - 3; i++) {
            if (i > 0 && arr[i] == arr[i - 1]) continue;
            
            for (int j = i + 1; j < n - 2; j++) {
                if (j > i + 1 && arr[j] == arr[j - 1]) continue;
                
                int left = j + 1, right = n - 1;
                while (left < right) {
                    int sum = arr[i] + arr[j] + arr[left] + arr[right];
                    if (sum == X) {
                        result.push_back({arr[i], arr[j], arr[left], arr[right]});
                        while (left < right && arr[left] == arr[left + 1]) left++;
                        while (left < right && arr[right] == arr[right - 1]) right--;
                        left++;
                        right--;
                    } else if (sum < X) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
        }
        return result;
    }

    vector<vector<int>> Four_Sum_Hashing(vector<int>& arr, int n, int X) {
        /*
        Use hash map to store sum of pairs and find complement pairs
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        vector<vector<int>> result;
        unordered_map<int, vector<pair<int, int>>> pairSum;
        
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                int sum = arr[i] + arr[j];
                int complement = X - sum;
                
                if (pairSum.find(complement) != pairSum.end()) {
                    for (auto& p : pairSum[complement]) {
                        if (p.first != i && p.first != j && p.second != i && p.second != j) {
                            vector<int> quad = {arr[p.first], arr[p.second], arr[i], arr[j]};
                            sort(quad.begin(), quad.end());
                            result.push_back(quad);
                        }
                    }
                }
                pairSum[sum].push_back({i, j});
            }
        }
        
        sort(result.begin(), result.end());
        result.erase(unique(result.begin(), result.end()), result.end());
        return result;
    }
};

void Test_Four_Sum() {
    Solution sol;
    vector<pair<vector<int>, int>> tests = {
        {{0, 0, 2, 1, 1}, 3},
        {{10, 2, 3, 4, 5, 7, 8}, 23},
        {{1, 0, -1, 0, -2, 2}, 0},
        {{2, 2, 2, 2, 2}, 8}
    };

    for (auto& test : tests) {
        vector<int> arr = test.first;
        int X = test.second;
        int n = arr.size();
        
        cout << "Array: ";
        for (int num : arr) cout << num << " ";
        cout << ", X = " << X << endl;
        
        vector<int> arr1 = arr, arr2 = arr;
        vector<vector<int>> res1 = sol.Four_Sum_Sorting_Two_Pointer(arr1, n, X);
        vector<vector<int>> res2 = sol.Four_Sum_Hashing(arr2, n, X);
        
        cout << "Sorting + Two Pointer: ";
        for (auto& quad : res1) {
            for (int num : quad) cout << num << " ";
        }
        cout << endl;
        
        cout << "Hashing: ";
        for (auto& quad : res2) {
            for (int num : quad) cout << num << " ";
        }
        cout << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Four_Sum();
    return 0;
}
