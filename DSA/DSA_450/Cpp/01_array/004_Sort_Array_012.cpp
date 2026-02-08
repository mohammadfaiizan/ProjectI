/*
Problem: Sort an Array of 0s, 1s and 2s
URL: https://practice.geeksforgeeks.org/problems/sort-an-array-of-0s-1s-and-2s4231/1

Problem Statement:
Given an array of size N containing only 0s, 1s, and 2s, sort the array in ascending order
without using any sorting algorithm (Dutch National Flag Problem).

Sample Input/Output:
Input: arr = [0, 2, 1, 2, 0]
Output: [0, 0, 1, 2, 2]

Input: arr = [0, 1, 0, 1, 2, 1, 2, 0]
Output: [0, 0, 0, 1, 1, 1, 2, 2]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Sort_012_Dutch_National_Flag_Optimal(vector<int>& arr) {
        /*
        Dutch National Flag - Three pointer approach (low, mid, high)
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int low = 0, mid = 0, high = arr.size() - 1;
        while (mid <= high) {
            if (arr[mid] == 0) swap(arr[mid++], arr[low++]);
            else if (arr[mid] == 1) mid++;
            else swap(arr[mid], arr[high--]);
        }
    }

    void Sort_012_Counting(vector<int>& arr) {
        /*
        Counting Approach - Count 0s, 1s, 2s and overwrite
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int c0 = 0, c1 = 0, c2 = 0;
        for (int x : arr) {
            if (x == 0) c0++;
            else if (x == 1) c1++;
            else c2++;
        }
        int i = 0;
        while (c0--) arr[i++] = 0;
        while (c1--) arr[i++] = 1;
        while (c2--) arr[i++] = 2;
    }
};

void Test_Sort_Array_012() {
    Solution solution;

    vector<vector<int>> test_cases = {
        {0, 2, 1, 2, 0},
        {0, 1, 0, 1, 2, 1, 2, 0},
        {2, 2, 2, 0, 0, 0, 1, 1},
        {0},
        {1, 0}
    };

    for (auto& arr : test_cases) {
        cout << "Original: ";
        for (int x : arr) cout << x << " ";
        cout << endl;

        vector<int> arr1 = arr, arr2 = arr;

        solution.Sort_012_Dutch_National_Flag_Optimal(arr1);
        cout << "Dutch National Flag: ";
        for (int x : arr1) cout << x << " ";
        cout << endl;

        solution.Sort_012_Counting(arr2);
        cout << "Counting: ";
        for (int x : arr2) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Sort_Array_012();
    return 0;
}
