/*
Problem: Count Pairs with Given Sum
URL: https://practice.geeksforgeeks.org/problems/count-pairs-with-given-sum5022/1

Problem Statement:
Given an array of N integers, and an integer K, find the number of pairs of elements
in the array whose sum is equal to K.

Sample Input/Output:
Input: arr = [1, 5, 7, -1, 5], K = 6
Output: 3
Explanation: Pairs are (1,5), (7,-1), (1,5).

Input: arr = [1, 1, 1, 1], K = 2
Output: 6
Explanation: All C(4,2) = 6 pairs give sum 2.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Count_Pairs_Hashing_Optimal(vector<int>& arr, int k) {
        /*
        Hashing Approach - Count complements as we iterate
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_map<int, int> freq;
        int count = 0;
        for (int x : arr) {
            count += freq[k - x];
            freq[x]++;
        }
        return count;
    }

    int Count_Pairs_Brute_Force(vector<int>& arr, int k) {
        /*
        Brute Force - Check all pairs
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int count = 0, n = arr.size();
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (arr[i] + arr[j] == k) count++;
        return count;
    }
};

void Test_Count_Pairs_With_Given_Sum() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        int k;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{1, 5, 7, -1, 5}, 6, 3},
        {{1, 1, 1, 1}, 2, 6},
        {{10, 12, 10, 15, -1, 7, 6, 5, 4, 2, 1, 1, 1}, 11, 9},
        {{1, 2, 3, 4}, 10, 0}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", K=" << tc.k << ", Expected=" << tc.expected << endl;

        cout << "Hashing: " << solution.Count_Pairs_Hashing_Optimal(tc.arr, tc.k) << endl;
        cout << "Brute Force: " << solution.Count_Pairs_Brute_Force(tc.arr, tc.k) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Count_Pairs_With_Given_Sum();
    return 0;
}
