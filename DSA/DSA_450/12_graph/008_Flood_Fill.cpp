/*
Problem: Flood Fill Algorithm
URL: https://leetcode.com/problems/flood-fill/

Problem Statement:
Given an image (2D grid), a starting pixel, and new color, perform flood fill.

Sample Input/Output:
Input: image=[[1,1,1],[1,1,0],[1,0,1]], sr=1, sc=1, color=2
Output: [[2,2,2],[2,2,0],[2,0,1]]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Flood_Fill_DFS_Helper(int row, int col, int oldColor, int newColor, vector<vector<int>>& image, int m, int n) {
        if (row < 0 || row >= m || col < 0 || col >= n || image[row][col] != oldColor || image[row][col] == newColor) {
            return;
        }
        
        image[row][col] = newColor;
        
        Flood_Fill_DFS_Helper(row + 1, col, oldColor, newColor, image, m, n);
        Flood_Fill_DFS_Helper(row - 1, col, oldColor, newColor, image, m, n);
        Flood_Fill_DFS_Helper(row, col + 1, oldColor, newColor, image, m, n);
        Flood_Fill_DFS_Helper(row, col - 1, oldColor, newColor, image, m, n);
    }

    vector<vector<int>> Flood_Fill_DFS(vector<vector<int>>& image, int sr, int sc, int color) {
        /*
        Recursive DFS
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        */
        int m = image.size();
        int n = image[0].size();
        int oldColor = image[sr][sc];
        
        if (oldColor != color) {
            Flood_Fill_DFS_Helper(sr, sc, oldColor, color, image, m, n);
        }
        
        return image;
    }

    vector<vector<int>> Flood_Fill_BFS(vector<vector<int>>& image, int sr, int sc, int color) {
        /*
        Iterative BFS
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        */
        int m = image.size();
        int n = image[0].size();
        int oldColor = image[sr][sc];
        
        if (oldColor == color) {
            return image;
        }
        
        queue<pair<int, int>> q;
        q.push({sr, sc});
        image[sr][sc] = color;
        
        int directions[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        
        while (!q.empty()) {
            auto current = q.front();
            q.pop();
            
            int row = current.first;
            int col = current.second;
            
            for (int i = 0; i < 4; i++) {
                int newRow = row + directions[i][0];
                int newCol = col + directions[i][1];
                
                if (newRow >= 0 && newRow < m && newCol >= 0 && newCol < n && 
                    image[newRow][newCol] == oldColor) {
                    image[newRow][newCol] = color;
                    q.push({newRow, newCol});
                }
            }
        }
        
        return image;
    }
};

void Test_Flood_Fill() {
    Solution solution;
    
    cout << "Test: Flood Fill" << endl;
    vector<vector<int>> image = {
        {1, 1, 1},
        {1, 1, 0},
        {1, 0, 1}
    };
    
    cout << "Original image:" << endl;
    for (const auto& row : image) {
        for (int pixel : row) {
            cout << pixel << " ";
        }
        cout << endl;
    }
    
    vector<vector<int>> result1 = solution.Flood_Fill_DFS(image, 1, 1, 2);
    cout << "\nAfter flood fill (DFS):" << endl;
    for (const auto& row : result1) {
        for (int pixel : row) {
            cout << pixel << " ";
        }
        cout << endl;
    }
    
    vector<vector<int>> image2 = {
        {1, 1, 1},
        {1, 1, 0},
        {1, 0, 1}
    };
    
    vector<vector<int>> result2 = solution.Flood_Fill_BFS(image2, 1, 1, 2);
    cout << "\nAfter flood fill (BFS):" << endl;
    for (const auto& row : result2) {
        for (int pixel : row) {
            cout << pixel << " ";
        }
        cout << endl;
    }
}

int main() {
    Test_Flood_Fill();
    return 0;
}
