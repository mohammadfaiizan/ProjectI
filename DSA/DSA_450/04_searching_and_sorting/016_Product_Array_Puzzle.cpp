/*
Problem: Product Array Puzzle
URL: https://practice.geeksforgeeks.org/problems/product-array-puzzle4525/1

Problem Statement:
Given an array nums[] of size n, construct a Product Array P (of same size n) such that P[i] is equal to the product of all the elements of nums except nums[i].

Sample Input/Output:
Input: n = 5, nums[] = {10, 3, 5, 6, 2}
Output: 180 600 360 300 900

Input: n = 2, nums[] = {12, 0}
Output: 0 12
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<long long> Product_Array_Left_Right(vector<int>& nums, int n) {
        /*
        Calculate left product and right product arrays, then multiply corresponding elements
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        vector<long long> result(n, 1);
        
        long long left = 1;
        for (int i = 0; i < n; i++) {
            result[i] = left;
            left *= nums[i];
        }
        
        long long right = 1;
        for (int i = n - 1; i >= 0; i--) {
            result[i] *= right;
            right *= nums[i];
        }
        
        return result;
    }

    vector<long long> Product_Array_Zero_Counting(vector<int>& nums, int n) {
        /*
        Count zeros and handle division with zero cases
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        long long product = 1;
        int zeroCount = 0;
        int zeroIndex = -1;
        
        for (int i = 0; i < n; i++) {
            if (nums[i] == 0) {
                zeroCount++;
                zeroIndex = i;
            } else {
                product *= nums[i];
            }
        }
        
        vector<long long> result(n, 0);
        
        if (zeroCount > 1) {
            return result;
        } else if (zeroCount == 1) {
            result[zeroIndex] = product;
            return result;
        } else {
            for (int i = 0; i < n; i++) {
                result[i] = product / nums[i];
            }
            return result;
        }
    }
};

void Test_Product_Array_Puzzle() {
    Solution sol;
    vector<vector<int>> tests = {
        {10, 3, 5, 6, 2},
        {12, 0},
        {1, 2, 3, 4},
        {0, 0, 1, 2},
        {1, 0, 3, 4}
    };

    for (auto& nums : tests) {
        int n = nums.size();
        cout << "Array: ";
        for (int num : nums) cout << num << " ";
        cout << endl;
        
        vector<long long> res1 = sol.Product_Array_Left_Right(nums, n);
        vector<long long> res2 = sol.Product_Array_Zero_Counting(nums, n);
        
        cout << "Left-Right Product: ";
        for (long long val : res1) cout << val << " ";
        cout << endl;
        
        cout << "Zero Counting: ";
        for (long long val : res2) cout << val << " ";
        cout << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Product_Array_Puzzle();
    return 0;
}
