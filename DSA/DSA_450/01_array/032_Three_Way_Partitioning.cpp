/*
Problem: Three Way Partitioning
URL: https://practice.geeksforgeeks.org/problems/three-way-partitioning/1

Problem Statement:
Given an array and a range [a, b], partition the array such that elements < a come first,
then elements in range [a, b], and finally elements > b.

Sample Input/Output:
Input: arr = [1, 14, 5, 20, 4, 2, 54, 20, 87, 98, 3, 1, 32], a = 14, b = 20
Output: [1, 5, 4, 2, 3, 1, 14, 20, 20, 54, 87, 98, 32] (one possible output)

Input: arr = [1, 2, 3, 3, 4], a = 1, b = 2
Output: [1, 2, 3, 3, 4]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Three_Way_Partition_Dutch_Flag_Optimal(vector<int>& arr, int a, int b) {
        /*
        Dutch National Flag Variant - Three pointer approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int low = 0, mid = 0, high = arr.size() - 1;
        while (mid <= high) {
            if (arr[mid] < a) {
                swap(arr[mid++], arr[low++]);
            } else if (arr[mid] > b) {
                swap(arr[mid], arr[high--]);
            } else {
                mid++;
            }
        }
    }
};

void Test_Three_Way_Partitioning() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int a, b;
    };

    vector<TestCase> test_cases = {
        {{1, 14, 5, 20, 4, 2, 54, 20, 87, 98, 3, 1, 32}, 14, 20},
        {{1, 2, 3, 3, 4}, 1, 2},
        {{87, 78, 16, 94}, 16, 78}
    };

    for (auto& tc : test_cases) {
        cout << "Original: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", Range=[" << tc.a << ", " << tc.b << "]" << endl;

        vector<int> arr1 = tc.arr;
        solution.Three_Way_Partition_Dutch_Flag_Optimal(arr1, tc.a, tc.b);
        cout << "Dutch Flag: ";
        for (int x : arr1) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Three_Way_Partitioning();
    return 0;
}
