/*
Problem: Find All Elements Appearing More Than N/K Times
URL: https://www.geeksforgeeks.org/given-an-array-of-of-size-n-finds-all-the-elements-that-appear-more-than-nk-times/

Problem Statement:
Given an array of size n, find all elements that appear more than n/k times.

Sample Input/Output:
Input: arr = [3, 1, 2, 2, 1, 2, 3, 3], K = 4
Output: [2, 3]
Explanation: Elements 2 and 3 appear more than 8/4 = 2 times.

Input: arr = [9, 8, 7, 9, 2, 9, 7], K = 3
Output: [9]
Explanation: Only 9 appears more than 7/3 = 2 times.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Elements_N_By_K_Hashing_Optimal(vector<int>& arr, int k) {
        /*
        Hashing Approach - Count frequency using map
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_map<int, int> freq;
        for (int x : arr) freq[x]++;
        int threshold = arr.size() / k;
        vector<int> result;
        for (auto& [val, count] : freq) {
            if (count > threshold) result.push_back(val);
        }
        return result;
    }

    vector<int> Elements_N_By_K_Sorting(vector<int> arr, int k) {
        /*
        Sorting Approach - Sort and check consecutive count
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(arr.begin(), arr.end());
        int n = arr.size();
        int threshold = n / k;
        vector<int> result;
        int i = 0;
        while (i < n) {
            int count = 1;
            while (i + count < n && arr[i + count] == arr[i]) count++;
            if (count > threshold) result.push_back(arr[i]);
            i += count;
        }
        return result;
    }
};

void Test_Elements_More_Than_N_By_K() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int k;
    };

    vector<TestCase> test_cases = {
        {{3, 1, 2, 2, 1, 2, 3, 3}, 4},
        {{9, 8, 7, 9, 2, 9, 7}, 3},
        {{1, 1, 2, 2, 3, 5, 4, 2, 2, 3, 1, 1, 1}, 3}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", K=" << tc.k << endl;

        auto r1 = solution.Elements_N_By_K_Hashing_Optimal(tc.arr, tc.k);
        cout << "Hashing: ";
        for (int x : r1) cout << x << " ";
        cout << endl;

        auto r2 = solution.Elements_N_By_K_Sorting(tc.arr, tc.k);
        cout << "Sorting: ";
        for (int x : r2) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Elements_More_Than_N_By_K();
    return 0;
}
