/*
Problem: Move All Negative Elements to One Side
URL: https://www.geeksforgeeks.org/move-negative-numbers-beginning-positive-end-constant-extra-space/

Problem Statement:
Given an unsorted array of both negative and positive integers, move all negative numbers
to the beginning and all positive numbers to the end. Order of elements is not important.

Sample Input/Output:
Input: arr = [-12, 11, -13, -5, 6, -7, 5, -3, -6]
Output: [-12, -13, -5, -7, -3, -6, 11, 6, 5] (one possible output)

Input: arr = [1, -1, 3, 2, -7, -5, 11, 6]
Output: [-1, -7, -5, 3, 2, 1, 11, 6] (one possible output)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Negative_One_Side_Two_Pointer_Optimal(vector<int>& arr) {
        /*
        Two Pointer Approach - Pointers from both ends
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int left = 0, right = arr.size() - 1;
        while (left <= right) {
            if (arr[left] < 0 && arr[right] < 0) {
                left++;
            } else if (arr[left] > 0 && arr[right] < 0) {
                swap(arr[left], arr[right]);
                left++;
                right--;
            } else if (arr[left] > 0 && arr[right] > 0) {
                right--;
            } else {
                left++;
                right--;
            }
        }
    }

    void Negative_One_Side_Partition(vector<int>& arr) {
        /*
        Partition Based - Similar to quicksort partition around 0
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int j = 0;
        for (int i = 0; i < (int)arr.size(); i++) {
            if (arr[i] < 0) {
                if (i != j) swap(arr[i], arr[j]);
                j++;
            }
        }
    }
};

void Test_Negative_Elements_One_Side() {
    Solution solution;

    vector<vector<int>> test_cases = {
        {-12, 11, -13, -5, 6, -7, 5, -3, -6},
        {1, -1, 3, 2, -7, -5, 11, 6},
        {-1, -2, -3},
        {1, 2, 3}
    };

    for (auto& arr : test_cases) {
        cout << "Original: ";
        for (int x : arr) cout << x << " ";
        cout << endl;

        vector<int> arr1 = arr, arr2 = arr;

        solution.Negative_One_Side_Two_Pointer_Optimal(arr1);
        cout << "Two Pointer: ";
        for (int x : arr1) cout << x << " ";
        cout << endl;

        solution.Negative_One_Side_Partition(arr2);
        cout << "Partition: ";
        for (int x : arr2) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Negative_Elements_One_Side();
    return 0;
}
