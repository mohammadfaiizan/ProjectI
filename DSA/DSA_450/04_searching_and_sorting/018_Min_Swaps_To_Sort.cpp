/*
Problem: Minimum Swaps to Sort
URL: https://practice.geeksforgeeks.org/problems/minimum-swaps/1

Problem Statement:
Given an array of n distinct elements. Find the minimum number of swaps required to sort the array in strictly increasing order.

Sample Input/Output:
Input: nums[] = {2, 8, 5, 4}
Output: 1

Input: nums[] = {10, 19, 6, 3, 5}
Output: 2
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Swaps_Graph_Cycle_Detection(vector<int>& nums, int n) {
        /*
        Create graph of cycles where each element should be at its sorted position
        Count cycles and swaps needed = n - number of cycles
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        vector<pair<int, int>> arrPos;
        for (int i = 0; i < n; i++) {
            arrPos.push_back({nums[i], i});
        }
        
        sort(arrPos.begin(), arrPos.end());
        
        vector<bool> visited(n, false);
        int swaps = 0;
        
        for (int i = 0; i < n; i++) {
            if (visited[i] || arrPos[i].second == i) {
                continue;
            }
            
            int cycleSize = 0;
            int j = i;
            
            while (!visited[j]) {
                visited[j] = true;
                j = arrPos[j].second;
                cycleSize++;
            }
            
            if (cycleSize > 0) {
                swaps += (cycleSize - 1);
            }
        }
        
        return swaps;
    }

    int Min_Swaps_HashMap_Tracking(vector<int>& nums, int n) {
        /*
        Use hash map to track correct positions and count swaps needed
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        unordered_map<int, int> posMap;
        for (int i = 0; i < n; i++) {
            posMap[nums[i]] = i;
        }
        
        vector<int> sortedNums = nums;
        sort(sortedNums.begin(), sortedNums.end());
        
        int swaps = 0;
        vector<bool> visited(n, false);
        
        for (int i = 0; i < n; i++) {
            if (visited[i] || nums[i] == sortedNums[i]) {
                continue;
            }
            
            int cycleSize = 0;
            int j = i;
            
            while (!visited[j]) {
                visited[j] = true;
                j = posMap[sortedNums[j]];
                cycleSize++;
            }
            
            swaps += (cycleSize - 1);
        }
        
        return swaps;
    }
};

void Test_Min_Swaps_To_Sort() {
    Solution sol;
    vector<vector<int>> tests = {
        {2, 8, 5, 4},
        {10, 19, 6, 3, 5},
        {1, 5, 4, 3, 2},
        {1, 2, 3, 4, 5},
        {4, 3, 2, 1}
    };

    for (auto& nums : tests) {
        int n = nums.size();
        cout << "Array: ";
        for (int num : nums) cout << num << " ";
        cout << endl;
        
        vector<int> nums1 = nums, nums2 = nums;
        int res1 = sol.Min_Swaps_Graph_Cycle_Detection(nums1, n);
        int res2 = sol.Min_Swaps_HashMap_Tracking(nums2, n);
        
        cout << "Graph Cycle Detection: " << res1 << endl;
        cout << "HashMap Tracking: " << res2 << endl;
        
        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Min_Swaps_To_Sort();
    return 0;
}
