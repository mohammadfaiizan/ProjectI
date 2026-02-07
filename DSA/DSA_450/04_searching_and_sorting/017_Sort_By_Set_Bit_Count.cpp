/*
Problem: Sort Array by Set Bit Count
URL: https://practice.geeksforgeeks.org/problems/sort-by-set-bit-count1153/1

Problem Statement:
Given an array of integers, sort the array (in decreasing order) according to count of set bits in binary representation of array elements.

Sample Input/Output:
Input: arr[] = {5, 2, 3, 9, 4, 6, 7, 15, 32}
Output: 15 7 5 3 9 6 2 4 32

Input: arr[] = {1, 2, 3, 4, 5, 6}
Output: 3 5 6 1 2 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Sort_By_Set_Bit_Custom_Comparator(vector<int>& arr, int n) {
        /*
        Use custom comparator with stable_sort to sort by set bit count
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        auto countBits = [](int num) -> int {
            int count = 0;
            while (num) {
                count += num & 1;
                num >>= 1;
            }
            return count;
        };
        
        stable_sort(arr.begin(), arr.end(), [&](int a, int b) {
            int bitsA = countBits(a);
            int bitsB = countBits(b);
            if (bitsA == bitsB) {
                return false;
            }
            return bitsA > bitsB;
        });
    }

    void Sort_By_Set_Bit_Builtin_Popcount(vector<int>& arr, int n) {
        /*
        Use __builtin_popcount to count set bits efficiently
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        stable_sort(arr.begin(), arr.end(), [](int a, int b) {
            int bitsA = __builtin_popcount(a);
            int bitsB = __builtin_popcount(b);
            if (bitsA == bitsB) {
                return false;
            }
            return bitsA > bitsB;
        });
    }
};

void Test_Sort_By_Set_Bit_Count() {
    Solution sol;
    vector<vector<int>> tests = {
        {5, 2, 3, 9, 4, 6, 7, 15, 32},
        {1, 2, 3, 4, 5, 6},
        {1024, 512, 256, 128, 64},
        {7, 8, 9, 10, 11}
    };

    for (auto& arr : tests) {
        int n = arr.size();
        cout << "Original Array: ";
        for (int num : arr) cout << num << " ";
        cout << endl;
        
        vector<int> arr1 = arr, arr2 = arr;
        
        sol.Sort_By_Set_Bit_Custom_Comparator(arr1, n);
        cout << "Custom Comparator: ";
        for (int num : arr1) cout << num << " ";
        cout << endl;
        
        sol.Sort_By_Set_Bit_Builtin_Popcount(arr2, n);
        cout << "Builtin Popcount: ";
        for (int num : arr2) cout << num << " ";
        cout << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Sort_By_Set_Bit_Count();
    return 0;
}
