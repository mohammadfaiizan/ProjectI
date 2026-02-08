/*
Problem: Merge Two Sorted Arrays Without Extra Space
URL: https://practice.geeksforgeeks.org/problems/merge-two-sorted-arrays5135/1

Problem Statement:
Given two sorted arrays arr1[] and arr2[] of sizes n and m in non-decreasing order. Merge them in sorted order without using any extra space.

Sample Input/Output:
Input: n = 4, arr1[] = {1, 3, 5, 7}, m = 5, arr2[] = {0, 2, 6, 8, 9}
Output: arr1[] = {0, 1, 2, 3}, arr2[] = {5, 6, 7, 8, 9}

Input: n = 2, arr1[] = {10, 12}, m = 3, arr2[] = {5, 18, 20}
Output: arr1[] = {5, 10}, arr2[] = {12, 18, 20}
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Merge_Gap_Method(vector<int>& arr1, vector<int>& arr2, int n, int m) {
        /*
        Gap method based on shell sort - compare elements at gap distance and swap if needed
        Time Complexity: O((n+m) * log(n+m))
        Space Complexity: O(1)
        */
        int total = n + m;
        int gap = (total + 1) / 2;
        
        while (gap > 0) {
            int i = 0, j = gap;
            
            while (j < total) {
                if (i < n && j < n) {
                    if (arr1[i] > arr1[j]) {
                        swap(arr1[i], arr1[j]);
                    }
                } else if (i < n && j >= n) {
                    if (arr1[i] > arr2[j - n]) {
                        swap(arr1[i], arr2[j - n]);
                    }
                } else {
                    if (arr2[i - n] > arr2[j - n]) {
                        swap(arr2[i - n], arr2[j - n]);
                    }
                }
                i++;
                j++;
            }
            
            if (gap == 1) break;
            gap = (gap + 1) / 2;
        }
    }

    void Merge_Insertion_Based(vector<int>& arr1, vector<int>& arr2, int n, int m) {
        /*
        Compare last element of arr1 with first element of arr2 and insert appropriately
        Time Complexity: O(n * m)
        Space Complexity: O(1)
        */
        for (int i = n - 1; i >= 0; i--) {
            int last = arr1[i];
            int j = m - 2;
            
            while (j >= 0 && arr2[j] > last) {
                arr2[j + 1] = arr2[j];
                j--;
            }
            
            if (j != m - 2 || arr2[j + 1] > last) {
                arr2[j + 1] = last;
                arr1[i] = arr2[0];
                
                int first = arr2[0];
                int k = 1;
                while (k < m && arr2[k] < first) {
                    arr2[k - 1] = arr2[k];
                    k++;
                }
                arr2[k - 1] = first;
            }
        }
    }
};

void Test_Merge_Two_Sorted_Arrays() {
    Solution sol;
    vector<pair<pair<vector<int>, vector<int>>, pair<int, int>>> tests = {
        {{{1, 3, 5, 7}, {0, 2, 6, 8, 9}}, {4, 5}},
        {{{10, 12}, {5, 18, 20}}, {2, 3}},
        {{{1, 2}, {3, 4}}, {2, 2}},
        {{{1}, {1}}, {1, 1}}
    };

    for (auto& test : tests) {
        vector<int> arr1 = test.first.first;
        vector<int> arr2 = test.first.second;
        int n = test.second.first;
        int m = test.second.second;
        
        cout << "arr1: ";
        for (int num : arr1) cout << num << " ";
        cout << ", arr2: ";
        for (int num : arr2) cout << num << " ";
        cout << endl;
        
        vector<int> arr1a = arr1, arr2a = arr2;
        vector<int> arr1b = arr1, arr2b = arr2;
        
        sol.Merge_Gap_Method(arr1a, arr2a, n, m);
        cout << "Gap Method - arr1: ";
        for (int num : arr1a) cout << num << " ";
        cout << ", arr2: ";
        for (int num : arr2a) cout << num << " ";
        cout << endl;
        
        sol.Merge_Insertion_Based(arr1b, arr2b, n, m);
        cout << "Insertion Based - arr1: ";
        for (int num : arr1b) cout << num << " ";
        cout << ", arr2: ";
        for (int num : arr2b) cout << num << " ";
        cout << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Merge_Two_Sorted_Arrays();
    return 0;
}
