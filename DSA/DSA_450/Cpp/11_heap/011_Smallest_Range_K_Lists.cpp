/*
Problem: Smallest Range Covering Elements from K Lists
URL: https://practice.geeksforgeeks.org/problems/find-smallest-range-containing-elements-from-k-lists/1

Problem Statement:
Given K sorted lists, find the smallest range [a, b] such that at least one element from each list falls in the range.

Sample Input/Output:
Input: [[4,10,15,24,26],[0,9,12,20],[5,18,22,30]]
Output: [20,24]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    pair<int, int> Smallest_Range_Min_Heap(vector<vector<int>>& nums) {
        /*
        Min Heap Approach
        Time Complexity: O(n*k log k)
        Space Complexity: O(k)
        */
        int k = nums.size();
        priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>, greater<pair<int, pair<int, int>>>> pq;
        
        int maxVal = INT_MIN;
        for (int i = 0; i < k; i++) {
            pq.push({nums[i][0], {i, 0}});
            maxVal = max(maxVal, nums[i][0]);
        }
        
        pair<int, int> result = {pq.top().first, maxVal};
        int minRange = maxVal - pq.top().first;
        
        while (true) {
            auto [val, pos] = pq.top();
            int row = pos.first;
            int col = pos.second;
            pq.pop();
            
            if (col + 1 < nums[row].size()) {
                int nextVal = nums[row][col + 1];
                pq.push({nextVal, {row, col + 1}});
                maxVal = max(maxVal, nextVal);
                
                int currentRange = maxVal - pq.top().first;
                if (currentRange < minRange) {
                    minRange = currentRange;
                    result = {pq.top().first, maxVal};
                }
            } else {
                break;
            }
        }
        
        return result;
    }
    
    pair<int, int> Smallest_Range_Pointers(vector<vector<int>>& nums) {
        /*
        Pointer-based Approach
        Time Complexity: O(n*k^2)
        Space Complexity: O(k)
        */
        int k = nums.size();
        vector<int> pointers(k, 0);
        pair<int, int> result = {0, INT_MAX};
        int minRange = INT_MAX;
        
        while (true) {
            int minVal = INT_MAX;
            int maxVal = INT_MIN;
            int minIdx = -1;
            
            for (int i = 0; i < k; i++) {
                if (pointers[i] < nums[i].size()) {
                    if (nums[i][pointers[i]] < minVal) {
                        minVal = nums[i][pointers[i]];
                        minIdx = i;
                    }
                    maxVal = max(maxVal, nums[i][pointers[i]]);
                }
            }
            
            if (minIdx == -1) break;
            
            int currentRange = maxVal - minVal;
            if (currentRange < minRange) {
                minRange = currentRange;
                result = {minVal, maxVal};
            }
            
            pointers[minIdx]++;
        }
        
        return result;
    }
};

void Test_Smallest_Range() {
    Solution solution;
    
    vector<vector<int>> nums1 = {{4,10,15,24,26}, {0,9,12,20}, {5,18,22,30}};
    pair<int, int> result1 = solution.Smallest_Range_Min_Heap(nums1);
    cout << "Min Heap Result: [" << result1.first << ", " << result1.second << "]" << endl;
    
    vector<vector<int>> nums2 = {{4,10,15,24,26}, {0,9,12,20}, {5,18,22,30}};
    pair<int, int> result2 = solution.Smallest_Range_Pointers(nums2);
    cout << "Pointers Result: [" << result2.first << ", " << result2.second << "]" << endl;
    
    vector<vector<int>> nums3 = {{1,2,3}, {1,2,3}, {1,2,3}};
    pair<int, int> result3 = solution.Smallest_Range_Min_Heap(nums3);
    cout << "Test 2 Result: [" << result3.first << ", " << result3.second << "]" << endl;
}

int main() {
    Test_Smallest_Range();
    return 0;
}
