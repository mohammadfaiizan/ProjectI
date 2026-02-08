/*
Problem: Largest Rectangular Submatrix with Sum 0
URL: https://www.geeksforgeeks.org/largest-rectangular-sub-matrix-whose-sum-0/

Problem Statement:
Given a 2D matrix, find the largest rectangular submatrix whose sum is zero.

Sample Input/Output:
Input: matrix with positive and negative values
Output: Size of largest rectangular submatrix with sum 0
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Largest_Rect_Zero(vector<vector<int>>& matrix) {
        /*
        Fix columns and use prefix sums with hashmap
        Time Complexity: O(n^2*m)
        Space Complexity: O(m)
        */
        int m = matrix.size();
        if (m == 0) return 0;
        int n = matrix[0].size();
        if (n == 0) return 0;
        
        int maxArea = 0;
        
        for (int left = 0; left < n; left++) {
            vector<int> temp(m, 0);
            
            for (int right = left; right < n; right++) {
                for (int i = 0; i < m; i++) {
                    temp[i] += matrix[i][right];
                }
                
                unordered_map<int, int> prefixSum;
                prefixSum[0] = -1;
                int sum = 0;
                
                for (int i = 0; i < m; i++) {
                    sum += temp[i];
                    
                    if (prefixSum.find(sum) != prefixSum.end()) {
                        int height = i - prefixSum[sum];
                        int width = right - left + 1;
                        maxArea = max(maxArea, height * width);
                    } else {
                        prefixSum[sum] = i;
                    }
                }
            }
        }
        
        return maxArea;
    }
};

void Test_Largest_Rect_Zero() {
    Solution solution;
    
    vector<vector<int>> matrix = {
        {9, 7, 16, 5},
        {1, -6, -7, 3},
        {1, 8, 7, 9},
        {7, -2, 0, 10}
    };
    
    cout << "Largest rectangular submatrix with sum 0: " 
         << solution.Largest_Rect_Zero(matrix) << endl;
}

int main() {
    Test_Largest_Rect_Zero();
    return 0;
}
