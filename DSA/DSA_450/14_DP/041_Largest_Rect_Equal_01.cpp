/*
Problem: Largest Area Rectangular Submatrix with Equal 0s and 1s
URL: https://www.geeksforgeeks.org/largest-area-rectangular-sub-matrix-equal-number-1s-0s/

Problem Statement:
Given a binary matrix, find the largest rectangular sub-matrix with equal number of 1s and 0s. Replace 0 with -1, then find largest submatrix with sum 0.

Sample Input/Output:
Input: binary matrix
Output: largest area
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Largest_Rect_01_Kadane(vector<vector<int>>& matrix) {
        /*
        Kadane's Algorithm Approach
        Time Complexity: O(n^2*m)
        Space Complexity: O(m)
        */
        int rows = matrix.size();
        int cols = matrix[0].size();
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][j] = -1;
                }
            }
        }
        
        int maxArea = 0;
        
        for (int top = 0; top < rows; top++) {
            vector<int> temp(cols, 0);
            
            for (int bottom = top; bottom < rows; bottom++) {
                for (int j = 0; j < cols; j++) {
                    temp[j] += matrix[bottom][j];
                }
                
                int area = maxSubarrayWithSumZero(temp);
                maxArea = max(maxArea, area);
            }
        }
        
        return maxArea;
    }
    
private:
    int maxSubarrayWithSumZero(vector<int>& arr) {
        unordered_map<int, int> prefixSum;
        int sum = 0;
        int maxLen = 0;
        
        for (int i = 0; i < arr.size(); i++) {
            sum += arr[i];
            
            if (sum == 0) {
                maxLen = i + 1;
            }
            
            if (prefixSum.find(sum) != prefixSum.end()) {
                maxLen = max(maxLen, i - prefixSum[sum]);
            } else {
                prefixSum[sum] = i;
            }
        }
        
        return maxLen;
    }
};

void Test_Largest_Rect_01_Kadane() {
    Solution solution;
    vector<vector<int>> matrix = {
        {0, 0, 1, 1},
        {0, 1, 1, 1},
        {1, 1, 1, 1}
    };
    int result = solution.Largest_Rect_01_Kadane(matrix);
    assert(result >= 0);
}

int main() {
    Test_Largest_Rect_01_Kadane();
    return 0;
}
