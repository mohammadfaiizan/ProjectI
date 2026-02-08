/*
Problem: Count Inversions
URL: https://practice.geeksforgeeks.org/problems/inversion-of-array-1587115620/1

Problem Statement:
Given an array of N integers, count the number of inversions in the array.
An inversion occurs when arr[i] > arr[j] and i < j.

Sample Input/Output:
Input: arr = [2, 4, 1, 3, 5]
Output: 3
Explanation: Inversions are (2,1), (4,1), (4,3).

Input: arr = [5, 4, 3, 2, 1]
Output: 10
Explanation: Every pair is an inversion in a reverse-sorted array.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Count_Inversions_Merge_Sort_Optimal(vector<int>& arr) {
        /*
        Merge Sort Based - Count inversions during merge step
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        vector<int> temp(arr.size());
        return Merge_Sort_Count(arr, temp, 0, arr.size() - 1);
    }

    long long Count_Inversions_Brute_Force(vector<int>& arr) {
        /*
        Brute Force - Check all pairs
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        long long count = 0;
        int n = arr.size();
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (arr[i] > arr[j]) count++;
        return count;
    }

private:
    long long Merge_Sort_Count(vector<int>& arr, vector<int>& temp, int left, int right) {
        long long inv_count = 0;
        if (left < right) {
            int mid = (left + right) / 2;
            inv_count += Merge_Sort_Count(arr, temp, left, mid);
            inv_count += Merge_Sort_Count(arr, temp, mid + 1, right);
            inv_count += Merge_Count(arr, temp, left, mid, right);
        }
        return inv_count;
    }

    long long Merge_Count(vector<int>& arr, vector<int>& temp, int left, int mid, int right) {
        int i = left, j = mid + 1, k = left;
        long long inv_count = 0;
        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
                inv_count += (mid - i + 1);
            }
        }
        while (i <= mid) temp[k++] = arr[i++];
        while (j <= right) temp[k++] = arr[j++];
        for (i = left; i <= right; i++) arr[i] = temp[i];
        return inv_count;
    }
};

void Test_Count_Inversions() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        long long expected;
    };

    vector<TestCase> test_cases = {
        {{2, 4, 1, 3, 5}, 3},
        {{5, 4, 3, 2, 1}, 10},
        {{1, 20, 6, 4, 5}, 5},
        {{1, 2, 3, 4, 5}, 0}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        vector<int> arr1 = tc.arr;
        cout << "Merge Sort: " << solution.Count_Inversions_Merge_Sort_Optimal(arr1) << endl;
        cout << "Brute Force: " << solution.Count_Inversions_Brute_Force(tc.arr) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Count_Inversions();
    return 0;
}
