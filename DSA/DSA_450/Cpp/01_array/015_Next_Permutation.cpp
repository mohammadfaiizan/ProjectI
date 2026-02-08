/*
Problem: Next Permutation
URL: https://leetcode.com/problems/next-permutation/

Problem Statement:
Given an array of integers nums, find the next permutation in lexicographical order.
If no such arrangement exists, rearrange to the lowest possible order (sorted ascending).

Sample Input/Output:
Input: nums = [1, 2, 3]
Output: [1, 3, 2]

Input: nums = [3, 2, 1]
Output: [1, 2, 3]

Input: nums = [1, 1, 5]
Output: [1, 5, 1]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Next_Permutation_Optimal(vector<int>& nums) {
        /*
        Optimal Approach - Find rightmost ascending pair, swap and reverse
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = nums.size();
        int i = n - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) i--;
        if (i >= 0) {
            int j = n - 1;
            while (nums[j] <= nums[i]) j--;
            swap(nums[i], nums[j]);
        }
        reverse(nums.begin() + i + 1, nums.end());
    }

    void Next_Permutation_STL(vector<int>& nums) {
        /*
        STL Approach - Using built-in next_permutation
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        next_permutation(nums.begin(), nums.end());
    }
};

void Test_Next_Permutation() {
    Solution solution;

    vector<vector<int>> test_cases = {
        {1, 2, 3},
        {3, 2, 1},
        {1, 1, 5},
        {1, 3, 2}
    };

    for (auto& nums : test_cases) {
        cout << "Input: ";
        for (int x : nums) cout << x << " ";
        cout << endl;

        vector<int> nums1 = nums, nums2 = nums;

        solution.Next_Permutation_Optimal(nums1);
        cout << "Optimal: ";
        for (int x : nums1) cout << x << " ";
        cout << endl;

        solution.Next_Permutation_STL(nums2);
        cout << "STL: ";
        for (int x : nums2) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Next_Permutation();
    return 0;
}
