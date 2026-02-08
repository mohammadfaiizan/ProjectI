/*
Problem: Maximum Sum Rectangle
URL: https://practice.geeksforgeeks.org/problems/maximum-sum-rectangle2948/1

Problem Statement:
Given a 2D matrix, find the maximum sum rectangle in it.

Sample Input/Output:
Input: 4x5 matrix
Output: maximum sum rectangle
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Rect_Kadane(vector<vector<int>>& matrix) {
        /*
        Kadane's Algorithm for 2D
        Time Complexity: O(n^2*m)
        Space Complexity: O(m)
        */
        int rows = matrix.size();
        int cols = matrix[0].size();
        int maxSum = INT_MIN;
        
        for (int left = 0; left < cols; left++) {
            vector<int> temp(rows, 0);
            
            for (int right = left; right < cols; right++) {
                for (int i = 0; i < rows; i++) {
                    temp[i] += matrix[i][right];
                }
                
                int currentSum = kadane(temp);
                maxSum = max(maxSum, currentSum);
            }
        }
        
        return maxSum;
    }
    
private:
    int kadane(vector<int>& arr) {
        int maxSum = arr[0];
        int currentSum = arr[0];
        
        for (int i = 1; i < arr.size(); i++) {
            currentSum = max(arr[i], currentSum + arr[i]);
            maxSum = max(maxSum, currentSum);
        }
        
        return maxSum;
    }
};

void Test_Max_Rect_Kadane() {
    Solution solution;
    vector<vector<int>> matrix = {
        {1, 2, -1, -4, -20},
        {-8, -3, 4, 2, 1},
        {3, 8, 10, 1, 3},
        {-4, -1, 1, 7, -6}
    };
    int result = solution.Max_Rect_Kadane(matrix);
    assert(result > 0);
}

int main() {
    Test_Max_Rect_Kadane();
    return 0;
}
