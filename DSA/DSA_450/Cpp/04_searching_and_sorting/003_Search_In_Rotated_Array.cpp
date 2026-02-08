/*
Problem: Search in Rotated Sorted Array
URL: https://leetcode.com/problems/search-in-rotated-sorted-array/

Problem Statement:
There is an integer array nums sorted in ascending order (with distinct values). Prior to being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed).

Sample Input/Output:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Search_Rotated_Find_Pivot_Then_Search(vector<int>& nums, int target) {
        /*
        Find pivot point first, then binary search in appropriate half
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        int n = nums.size();
        int left = 0, right = n - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        int pivot = left;
        left = 0, right = n - 1;
        
        if (target >= nums[pivot] && target <= nums[n - 1]) {
            left = pivot;
        } else {
            right = pivot - 1;
        }
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }

    int Search_Rotated_Single_Pass(vector<int>& nums, int target) {
        /*
        Single pass binary search accounting for rotation
        Time Complexity: O(log n)
        Space Complexity: O(1)
        */
        int left = 0, right = nums.size() - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target) {
                return mid;
            }
            
            if (nums[left] <= nums[mid]) {
                if (target >= nums[left] && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        
        return -1;
    }
};

void Test_Search_In_Rotated_Array() {
    Solution sol;
    vector<pair<vector<int>, int>> tests = {
        {{4, 5, 6, 7, 0, 1, 2}, 0},
        {{4, 5, 6, 7, 0, 1, 2}, 3},
        {{1}, 0},
        {{1, 3}, 3},
        {{3, 1}, 1}
    };

    for (auto& test : tests) {
        vector<int> nums = test.first;
        int target = test.second;
        
        cout << "Array: ";
        for (int num : nums) cout << num << " ";
        cout << ", target = " << target << endl;
        
        int res1 = sol.Search_Rotated_Find_Pivot_Then_Search(nums, target);
        cout << "Find Pivot Then Search: " << res1 << endl;
        
        int res2 = sol.Search_Rotated_Single_Pass(nums, target);
        cout << "Single Pass: " << res2 << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Search_In_Rotated_Array();
    return 0;
}
