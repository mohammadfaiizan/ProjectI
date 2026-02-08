/*
Problem: Find the Duplicate Number
URL: https://leetcode.com/problems/find-the-duplicate-number/

Problem Statement:
Given an array of integers nums containing n + 1 integers where each integer is in the
range [1, n] inclusive. There is only one repeated number, return this repeated number.

Sample Input/Output:
Input: nums = [1, 3, 4, 2, 2]
Output: 2

Input: nums = [3, 1, 3, 4, 2]
Output: 3
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Find_Duplicate_Floyd_Cycle_Optimal(vector<int>& nums) {
        /*
        Floyd's Tortoise and Hare - Cycle detection in linked list
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int slow = nums[0], fast = nums[0];
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        fast = nums[0];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    int Find_Duplicate_Negative_Marking(vector<int> nums) {
        /*
        Negative Marking - Mark visited indices as negative
        Time Complexity: O(n)
        Space Complexity: O(1) - modifies input
        */
        for (int i = 0; i < (int)nums.size(); i++) {
            int idx = abs(nums[i]) - 1;
            nums[idx] = -nums[idx];
            if (nums[idx] > 0) return idx + 1;
        }
        return -1;
    }

    int Find_Duplicate_Hashing(vector<int>& nums) {
        /*
        Hashing Approach - Use set to detect duplicate
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        unordered_set<int> seen;
        for (int x : nums) {
            if (seen.count(x)) return x;
            seen.insert(x);
        }
        return -1;
    }
};

void Test_Find_Duplicate() {
    Solution solution;

    struct TestCase {
        vector<int> nums;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{1, 3, 4, 2, 2}, 2},
        {{3, 1, 3, 4, 2}, 3},
        {{1, 1}, 1},
        {{2, 5, 9, 6, 9, 3, 8, 9, 7, 1}, 9}
    };

    for (auto& tc : test_cases) {
        cout << "Array: ";
        for (int x : tc.nums) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        cout << "Floyd's Cycle: " << solution.Find_Duplicate_Floyd_Cycle_Optimal(tc.nums) << endl;
        cout << "Negative Marking: " << solution.Find_Duplicate_Negative_Marking(tc.nums) << endl;
        cout << "Hashing: " << solution.Find_Duplicate_Hashing(tc.nums) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Find_Duplicate();
    return 0;
}
