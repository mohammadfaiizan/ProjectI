/*
Problem: Defence Of Kingdom
URL: https://www.spoj.com/problems/DEFKIN/

Problem Statement:
Given a rectangular grid of W x H with some fortified cells, find the largest undefended rectangular area.

Sample Input/Output:
Input: W=15, H=8, fortified cells: [(3,8), (11,2), (8,6)]
Output: 12
Explanation: Add boundary 0 and W+1/H+1, sort coordinates, find max gap in x and y, multiply.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Largest_Undefended_Area(int W, int H, vector<pair<int, int>>& fortified) {
        /*
        Sort + max gap approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        vector<int> x_coords, y_coords;
        x_coords.push_back(0);
        x_coords.push_back(W + 1);
        y_coords.push_back(0);
        y_coords.push_back(H + 1);
        
        for (auto& p : fortified) {
            x_coords.push_back(p.first);
            y_coords.push_back(p.second);
        }
        
        sort(x_coords.begin(), x_coords.end());
        sort(y_coords.begin(), y_coords.end());
        
        int max_x_gap = 0, max_y_gap = 0;
        
        for (int i = 1; i < x_coords.size(); i++) {
            max_x_gap = max(max_x_gap, x_coords[i] - x_coords[i-1] - 1);
        }
        
        for (int i = 1; i < y_coords.size(); i++) {
            max_y_gap = max(max_y_gap, y_coords[i] - y_coords[i-1] - 1);
        }
        
        return max_x_gap * max_y_gap;
    }
};

void Test_Defence_Of_Kingdom() {
    Solution solution;
    
    vector<pair<int, int>> fortified1 = {{3, 8}, {11, 2}, {8, 6}};
    cout << "Test 1: " << solution.Largest_Undefended_Area(15, 8, fortified1) << endl;
    
    vector<pair<int, int>> fortified2 = {{2, 2}};
    cout << "Test 2: " << solution.Largest_Undefended_Area(5, 5, fortified2) << endl;
}

int main() {
    Test_Defence_Of_Kingdom();
    return 0;
}
