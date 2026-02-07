/*
Problem: Reverse an Array
URL: https://www.geeksforgeeks.org/write-a-program-to-reverse-an-array-or-string/

Problem Statement:
Given an array (or string), the task is to reverse the array/string.

Sample Input/Output:
Input: arr = [1, 2, 3, 4, 5]
Output: [5, 4, 3, 2, 1]
Explanation: Array elements are reversed in-place.

Input: arr = [4, 5, 1, 2]
Output: [2, 1, 5, 4]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Reverse_Array_Two_Pointer_Optimal(vector<int>& arr) {
        /*
        Two Pointer Iterative - Swap elements from both ends
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int start = 0, end = arr.size() - 1;
        while (start < end) {
            swap(arr[start], arr[end]);
            start++;
            end--;
        }
    }

    void Reverse_Array_Recursive(vector<int>& arr, int start, int end) {
        /*
        Recursive Approach - Recursively swap endpoints
        Time Complexity: O(n)
        Space Complexity: O(n) - recursion stack
        */
        if (start >= end) return;
        swap(arr[start], arr[end]);
        Reverse_Array_Recursive(arr, start + 1, end - 1);
    }

    void Reverse_Array_STL(vector<int>& arr) {
        /*
        STL Reverse - Using built-in reverse function
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        reverse(arr.begin(), arr.end());
    }
};

void Test_Reverse_Array() {
    Solution solution;

    vector<vector<int>> test_cases = {
        {1, 2, 3, 4, 5},
        {4, 5, 1, 2},
        {1},
        {1, 2}
    };

    for (auto& arr : test_cases) {
        cout << "Original: ";
        for (int x : arr) cout << x << " ";
        cout << endl;

        vector<int> arr1 = arr, arr2 = arr, arr3 = arr;

        solution.Reverse_Array_Two_Pointer_Optimal(arr1);
        cout << "Two Pointer: ";
        for (int x : arr1) cout << x << " ";
        cout << endl;

        solution.Reverse_Array_Recursive(arr2, 0, arr2.size() - 1);
        cout << "Recursive: ";
        for (int x : arr2) cout << x << " ";
        cout << endl;

        solution.Reverse_Array_STL(arr3);
        cout << "STL: ";
        for (int x : arr3) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Reverse_Array();
    return 0;
}
