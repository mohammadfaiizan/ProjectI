/*
Problem: Maximum Product Subarray
URL: https://practice.geeksforgeeks.org/problems/maximum-product-subarray3604/1

Problem Statement:
Given an array arr[] that contains positive and negative integers (may contain 0),
find the maximum product subarray.

Sample Input/Output:
Input: arr = [6, -3, -10, 0, 2]
Output: 180
Explanation: Subarray [6, -3, -10] has product 180.

Input: arr = [-1, -3, -10, 0, 60]
Output: 60
Explanation: Subarray [60] has product 60.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Max_Product_DP_Optimal(vector<int>& arr) {
        /*
        DP Approach - Track min and max product ending at each index
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        long long max_val = arr[0], min_val = arr[0], max_product = arr[0];
        for (int i = 1; i < (int)arr.size(); i++) {
            if (arr[i] < 0) swap(max_val, min_val);
            max_val = max((long long)arr[i], max_val * arr[i]);
            min_val = min((long long)arr[i], min_val * arr[i]);
            max_product = max(max_product, max_val);
        }
        return max_product;
    }

    long long Max_Product_Prefix_Suffix(vector<int>& arr) {
        /*
        Prefix-Suffix Product - Compute products from both ends
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = arr.size();
        long long max_product = LLONG_MIN;
        long long prefix = 1, suffix = 1;
        for (int i = 0; i < n; i++) {
            prefix *= arr[i];
            suffix *= arr[n - 1 - i];
            max_product = max({max_product, prefix, suffix});
            if (prefix == 0) prefix = 1;
            if (suffix == 0) suffix = 1;
        }
        return max_product;
    }

    long long Max_Product_Brute_Force(vector<int>& arr) {
        /*
        Brute Force - Check all subarrays
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int n = arr.size();
        long long result = arr[0];
        for (int i = 0; i < n; i++) {
            long long product = 1;
            for (int j = i; j < n; j++) {
                product *= arr[j];
                result = max(result, product);
            }
        }
        return result;
    }
};

void Test_Maximum_Product_Subarray() {
    Solution solution;

    struct TestCase {
        vector<int> arr;
        long long expected;
    };

    vector<TestCase> test_cases = {
        {{6, -3, -10, 0, 2}, 180},
        {{-1, -3, -10, 0, 60}, 60},
        {{2, 3, -2, 4}, 6},
        {{-2, 0, -1}, 0}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.arr) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        cout << "DP: " << solution.Max_Product_DP_Optimal(tc.arr) << endl;
        cout << "Prefix-Suffix: " << solution.Max_Product_Prefix_Suffix(tc.arr) << endl;
        cout << "Brute Force: " << solution.Max_Product_Brute_Force(tc.arr) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Maximum_Product_Subarray();
    return 0;
}
