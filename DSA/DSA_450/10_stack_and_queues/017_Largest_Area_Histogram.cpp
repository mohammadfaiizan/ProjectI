/*
Problem: Largest Rectangular Area in a Histogram
URL: https://practice.geeksforgeeks.org/problems/maximum-rectangular-area-in-a-histogram-1587115620/1

Problem Statement:
Find the largest rectangular area possible in a histogram where bars have unit width.

Sample Input/Output:
Input: [6,2,5,4,5,1,6]
Output: 12
Input: [2,1,5,6,2,3]
Output: 10
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Largest_Area_Histogram_Stack(vector<long long>& heights) {
        stack<int> st;
        long long maxArea = 0;
        int n = heights.size();
        for (int i = 0; i <= n; i++) {
            while (!st.empty() && (i == n || heights[st.top()] >= heights[i])) {
                long long height = heights[st.top()];
                st.pop();
                long long width = st.empty() ? i : i - st.top() - 1;
                maxArea = max(maxArea, height * width);
            }
            st.push(i);
        }
        return maxArea;
    }

    long long Largest_Area_Histogram_Divide_Conquer(vector<long long>& heights, int left, int right) {
        if (left > right) return 0;
        if (left == right) return heights[left];
        int minIdx = left;
        for (int i = left; i <= right; i++) {
            if (heights[i] < heights[minIdx]) {
                minIdx = i;
            }
        }
        long long area = heights[minIdx] * (right - left + 1);
        long long leftArea = Largest_Area_Histogram_Divide_Conquer(heights, left, minIdx - 1);
        long long rightArea = Largest_Area_Histogram_Divide_Conquer(heights, minIdx + 1, right);
        return max({area, leftArea, rightArea});
    }
};

void Test_Largest_Area_Histogram() {
    Solution solution;
    
    cout << "=== Stack Approach ===" << endl;
    vector<long long> heights1 = {6, 2, 5, 4, 5, 1, 6};
    cout << "Input: ";
    for (long long h : heights1) cout << h << " ";
    cout << endl;
    cout << "Output: " << solution.Largest_Area_Histogram_Stack(heights1) << endl;
    
    vector<long long> heights2 = {2, 1, 5, 6, 2, 3};
    cout << "\nInput: ";
    for (long long h : heights2) cout << h << " ";
    cout << endl;
    cout << "Output: " << solution.Largest_Area_Histogram_Stack(heights2) << endl;
    
    vector<long long> heights3 = {1, 2, 3, 4, 5};
    cout << "\nInput: ";
    for (long long h : heights3) cout << h << " ";
    cout << endl;
    cout << "Output: " << solution.Largest_Area_Histogram_Stack(heights3) << endl;
    
    cout << "\n=== Divide and Conquer Approach ===" << endl;
    vector<long long> heights4 = {6, 2, 5, 4, 5, 1, 6};
    cout << "Input: ";
    for (long long h : heights4) cout << h << " ";
    cout << endl;
    cout << "Output: " << solution.Largest_Area_Histogram_Divide_Conquer(heights4, 0, heights4.size() - 1) << endl;
    
    vector<long long> heights5 = {2, 1, 5, 6, 2, 3};
    cout << "\nInput: ";
    for (long long h : heights5) cout << h << " ";
    cout << endl;
    cout << "Output: " << solution.Largest_Area_Histogram_Divide_Conquer(heights5, 0, heights5.size() - 1) << endl;
}

int main() {
    Test_Largest_Area_Histogram();
    return 0;
}
