/*
Problem: Find Missing and Repeating
URL: https://practice.geeksforgeeks.org/problems/find-missing-and-repeating2512/1

Problem Statement:
Given an unsorted array Arr of size N of positive integers. One number 'A' from set {1, 2, …N} is missing and one number 'B' occurs twice in array. Find these two numbers.

Sample Input/Output:
Input: N = 2, Arr[] = {2, 2}
Output: 2 1

Input: N = 3, Arr[] = {1, 3, 3}
Output: 3 2
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    pair<int, int> Find_Repeating_Missing_Count_Array(vector<int>& arr, int n) {
        /*
        Using count array to track occurrences
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> count(n + 1, 0);
        int repeating = -1, missing = -1;
        
        for (int i = 0; i < n; i++) {
            count[arr[i]]++;
        }
        
        for (int i = 1; i <= n; i++) {
            if (count[i] == 0) {
                missing = i;
            } else if (count[i] == 2) {
                repeating = i;
            }
        }
        
        return {repeating, missing};
    }

    pair<int, int> Find_Repeating_Missing_Sign_Marking(vector<int>& arr, int n) {
        /*
        Using sign marking to identify repeating and missing
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int repeating = -1, missing = -1;
        
        for (int i = 0; i < n; i++) {
            int index = abs(arr[i]) - 1;
            if (arr[index] < 0) {
                repeating = abs(arr[i]);
            } else {
                arr[index] = -arr[index];
            }
        }
        
        for (int i = 0; i < n; i++) {
            if (arr[i] > 0) {
                missing = i + 1;
                break;
            }
        }
        
        return {repeating, missing};
    }

    pair<int, int> Find_Repeating_Missing_Math(vector<int>& arr, int n) {
        /*
        Using mathematical formulas (sum and sum of squares)
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        long long sum = 0, sumSq = 0;
        long long expectedSum = (long long)n * (n + 1) / 2;
        long long expectedSumSq = (long long)n * (n + 1) * (2 * n + 1) / 6;
        
        for (int i = 0; i < n; i++) {
            sum += arr[i];
            sumSq += (long long)arr[i] * arr[i];
        }
        
        long long diff = sum - expectedSum;
        long long diffSq = sumSq - expectedSumSq;
        
        long long sumBoth = diffSq / diff;
        int repeating = (int)(diff + sumBoth) / 2;
        int missing = (int)(sumBoth - repeating);
        
        return {repeating, missing};
    }
};

void Test_Find_Repeating_And_Missing() {
    Solution sol;
    vector<vector<int>> tests = {
        {2, 2},
        {1, 3, 3},
        {1, 2, 2, 4},
        {4, 3, 6, 2, 1, 1}
    };

    for (auto& arr : tests) {
        int n = arr.size();
        vector<int> arr1 = arr, arr2 = arr;
        
        cout << "Array: ";
        for (int num : arr) cout << num << " ";
        cout << endl;
        
        pair<int, int> res1 = sol.Find_Repeating_Missing_Count_Array(arr, n);
        cout << "Count Array: Repeating = " << res1.first << ", Missing = " << res1.second << endl;
        
        pair<int, int> res2 = sol.Find_Repeating_Missing_Sign_Marking(arr1, n);
        cout << "Sign Marking: Repeating = " << res2.first << ", Missing = " << res2.second << endl;
        
        pair<int, int> res3 = sol.Find_Repeating_Missing_Math(arr2, n);
        cout << "Math: Repeating = " << res3.first << ", Missing = " << res3.second << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Find_Repeating_And_Missing();
    return 0;
}
