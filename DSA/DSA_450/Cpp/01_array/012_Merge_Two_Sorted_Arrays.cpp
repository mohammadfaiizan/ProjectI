/*
Problem: Merge Two Sorted Arrays Without Extra Space
URL: https://practice.geeksforgeeks.org/problems/merge-two-sorted-arrays5135/1

Problem Statement:
Given two sorted arrays arr1[] of size N and arr2[] of size M, merge both arrays without
using extra space. Modify arr1 to contain first N smallest and arr2 to contain remaining
M elements in sorted order.

Sample Input/Output:
Input: arr1 = [1, 3, 5, 7], arr2 = [0, 2, 6, 8, 9]
Output: arr1 = [0, 1, 2, 3], arr2 = [5, 6, 7, 8, 9]

Input: arr1 = [10, 12], arr2 = [5, 18, 20]
Output: arr1 = [5, 10], arr2 = [12, 18, 20]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Merge_Gap_Method_Optimal(vector<int>& arr1, vector<int>& arr2) {
        /*
        Gap Method (Shell Sort Variant) - Compare elements at gap distance
        Time Complexity: O((n+m) * log(n+m))
        Space Complexity: O(1)
        */
        int n = arr1.size(), m = arr2.size();
        int gap = (n + m + 1) / 2;
        while (gap > 0) {
            int i = 0, j = gap;
            while (j < n + m) {
                int& val_i = (i < n) ? arr1[i] : arr2[i - n];
                int& val_j = (j < n) ? arr1[j] : arr2[j - n];
                if (val_i > val_j) swap(val_i, val_j);
                i++;
                j++;
            }
            if (gap == 1) break;
            gap = (gap + 1) / 2;
        }
    }

    void Merge_Compare_And_Sort(vector<int>& arr1, vector<int>& arr2) {
        /*
        Compare and Sort - Swap larger of arr1 with smaller of arr2
        Time Complexity: O(n * m)
        Space Complexity: O(1)
        */
        int n = arr1.size(), m = arr2.size();
        for (int i = n - 1; i >= 0; i--) {
            if (arr1[i] > arr2[0]) {
                swap(arr1[i], arr2[0]);
                int first = arr2[0], k;
                for (k = 1; k < m && arr2[k] < first; k++)
                    arr2[k - 1] = arr2[k];
                arr2[k - 1] = first;
            }
        }
    }

    vector<int> Merge_Extra_Space(vector<int>& arr1, vector<int>& arr2) {
        /*
        Extra Space Merge - Standard merge into new array
        Time Complexity: O(n + m)
        Space Complexity: O(n + m)
        */
        vector<int> result;
        int i = 0, j = 0;
        while (i < (int)arr1.size() && j < (int)arr2.size()) {
            if (arr1[i] <= arr2[j]) result.push_back(arr1[i++]);
            else result.push_back(arr2[j++]);
        }
        while (i < (int)arr1.size()) result.push_back(arr1[i++]);
        while (j < (int)arr2.size()) result.push_back(arr2[j++]);
        return result;
    }
};

void Test_Merge_Two_Sorted_Arrays() {
    Solution solution;

    struct TestCase {
        vector<int> arr1, arr2;
    };

    vector<TestCase> test_cases = {
        {{1, 3, 5, 7}, {0, 2, 6, 8, 9}},
        {{10, 12}, {5, 18, 20}},
        {{1, 2, 3}, {4, 5, 6}},
        {{2, 4, 6}, {1, 3, 5}}
    };

    for (auto& tc : test_cases) {
        cout << "arr1: ";
        for (int x : tc.arr1) cout << x << " ";
        cout << ", arr2: ";
        for (int x : tc.arr2) cout << x << " ";
        cout << endl;

        vector<int> a1 = tc.arr1, a2 = tc.arr2;
        solution.Merge_Gap_Method_Optimal(a1, a2);
        cout << "Gap Method - arr1: ";
        for (int x : a1) cout << x << " ";
        cout << ", arr2: ";
        for (int x : a2) cout << x << " ";
        cout << endl;

        a1 = tc.arr1; a2 = tc.arr2;
        solution.Merge_Compare_And_Sort(a1, a2);
        cout << "Compare&Sort - arr1: ";
        for (int x : a1) cout << x << " ";
        cout << ", arr2: ";
        for (int x : a2) cout << x << " ";
        cout << endl;

        a1 = tc.arr1; a2 = tc.arr2;
        auto merged = solution.Merge_Extra_Space(a1, a2);
        cout << "Extra Space: ";
        for (int x : merged) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Merge_Two_Sorted_Arrays();
    return 0;
}
