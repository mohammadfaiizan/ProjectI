/*
Problem: Cyclically Rotate an Array by One
URL: https://practice.geeksforgeeks.org/problems/cyclically-rotate-an-array-by-one2614/1

Problem Statement:
Given an array, rotate the array by one position in clock-wise direction.
The last element becomes the first element.

Sample Input/Output:
Input: arr = [1, 2, 3, 4, 5]
Output: [5, 1, 2, 3, 4]

Input: arr = [9, 8, 7, 6, 4, 2, 1, 3]
Output: [3, 9, 8, 7, 6, 4, 2, 1]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Rotate_By_One_Shift_Optimal(vector<int>& arr) {
        /*
        Shift Based - Store last element and shift all right
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        int temp = arr[n - 1];
        for (int i = n - 1; i > 0; i--) {
            arr[i] = arr[i - 1];
        }
        arr[0] = temp;
    }

    void Rotate_By_One_STL(vector<int>& arr) {
        /*
        STL Rotate - Using rotate with reverse iterators
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        rotate(arr.rbegin(), arr.rbegin() + 1, arr.rend());
    }
};

void Test_Cyclically_Rotate() {
    Solution solution;

    vector<vector<int>> test_cases = {
        {1, 2, 3, 4, 5},
        {9, 8, 7, 6, 4, 2, 1, 3},
        {1},
        {1, 2}
    };

    for (auto& arr : test_cases) {
        cout << "Original: ";
        for (int x : arr) cout << x << " ";
        cout << endl;

        vector<int> arr1 = arr, arr2 = arr;

        solution.Rotate_By_One_Shift_Optimal(arr1);
        cout << "Shift: ";
        for (int x : arr1) cout << x << " ";
        cout << endl;

        solution.Rotate_By_One_STL(arr2);
        cout << "STL: ";
        for (int x : arr2) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Cyclically_Rotate();
    return 0;
}
