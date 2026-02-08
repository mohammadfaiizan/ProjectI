/*
Problem: Next Permutation (Next Greater Number with Same Set of Digits)
URL: https://practice.geeksforgeeks.org/problems/next-permutation5226/1

Problem Statement:
Given a number represented as an array of digits, find the next greater number
using the same set of digits. If no greater permutation exists, return the
smallest permutation.

Sample Input/Output:
Input: [1, 2, 3]
Output: [1, 3, 2]

Input: [3, 2, 1]
Output: [1, 2, 3]

Input: [1, 1, 5]
Output: [1, 5, 1]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Next_Permutation_Optimal(vector<int>& nums) {
        /*
        1. Find rightmost element smaller than its next
        2. Swap with smallest element larger than it on right
        3. Reverse the suffix
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
        Using STL next_permutation
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        next_permutation(nums.begin(), nums.end());
    }

    string Next_Greater_Number_String(string num) {
        /*
        String version of next permutation
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int n = num.size();
        int i = n - 2;
        while (i >= 0 && num[i] >= num[i + 1]) i--;

        if (i < 0) {
            reverse(num.begin(), num.end());
            return num;
        }

        int j = n - 1;
        while (num[j] <= num[i]) j--;
        swap(num[i], num[j]);
        reverse(num.begin() + i + 1, num.end());
        return num;
    }
};

void Test_Next_Permutation() {
    Solution sol;
    vector<vector<int>> tests = {
        {1, 2, 3},
        {3, 2, 1},
        {1, 1, 5},
        {1, 3, 5, 4, 2},
        {5, 4, 3, 2, 1}
    };

    for (auto nums : tests) {
        cout << "Input: ";
        for (int x : nums) cout << x << " ";
        cout << endl;

        vector<int> v1 = nums, v2 = nums;
        sol.Next_Permutation_Optimal(v1);
        cout << "Optimal: ";
        for (int x : v1) cout << x << " ";
        cout << endl;

        sol.Next_Permutation_STL(v2);
        cout << "STL: ";
        for (int x : v2) cout << x << " ";
        cout << endl;

        cout << string(50, '-') << endl;
    }

    vector<string> str_tests = {"1234", "4321", "534976"};
    for (auto& s : str_tests) {
        cout << "String Input: " << s << endl;
        cout << "Next Greater: " << sol.Next_Greater_Number_String(s) << endl;
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Next_Permutation();
    return 0;
}
