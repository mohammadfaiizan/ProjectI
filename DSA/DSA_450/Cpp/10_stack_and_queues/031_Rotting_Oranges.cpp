/*
Problem: Minimum Time Required to Rot All Oranges
URL: https://practice.geeksforgeeks.org/problems/rotten-oranges2536/1

Problem Statement:
Given a grid of dimension nxm where each cell in the grid can have 3 values:
0: Empty cell
1: Cells have fresh oranges
2: Cells have rotten oranges
Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.
Return the minimum time in minutes until no cell has a fresh orange. If this is impossible, return -1.

Sample Input/Output:
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Rotting_Oranges_BFS(vector<vector<int>>& grid) {
        /*
        BFS level-by-level
        Time Complexity: O(R*C)
        Space Complexity: O(R*C)
        */
        int rows = grid.size();
        int cols = grid[0].size();
        queue<pair<int, int>> q;
        int fresh = 0;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 2) {
                    q.push({i, j});
                } else if (grid[i][j] == 1) {
                    fresh++;
                }
            }
        }
        
        if (fresh == 0) return 0;
        
        int time = 0;
        vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        while (!q.empty()) {
            int size = q.size();
            bool rotted = false;
            
            for (int i = 0; i < size; i++) {
                auto [x, y] = q.front();
                q.pop();
                
                for (auto [dx, dy] : directions) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && grid[nx][ny] == 1) {
                        grid[nx][ny] = 2;
                        q.push({nx, ny});
                        fresh--;
                        rotted = true;
                    }
                }
            }
            
            if (rotted) time++;
        }
        
        return fresh == 0 ? time : -1;
    }
    
    int Rotting_Oranges_Brute_Force(vector<vector<int>>& grid) {
        /*
        Brute force simulation
        Time Complexity: O(R*C * R*C)
        Space Complexity: O(R*C)
        */
        int rows = grid.size();
        int cols = grid[0].size();
        int time = 0;
        vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        while (true) {
            vector<vector<int>> next = grid;
            bool changed = false;
            
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (grid[i][j] == 2) {
                        for (auto [dx, dy] : directions) {
                            int nx = i + dx;
                            int ny = j + dy;
                            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && grid[nx][ny] == 1) {
                                next[nx][ny] = 2;
                                changed = true;
                            }
                        }
                    }
                }
            }
            
            if (!changed) break;
            grid = next;
            time++;
        }
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1) return -1;
            }
        }
        
        return time;
    }
};

void Test_Rotting_Oranges() {
    Solution solution;
    
    vector<vector<int>> grid1 = {{2,1,1},{1,1,0},{0,1,1}};
    vector<vector<int>> grid1_copy = grid1;
    cout << "Test 1 - BFS: " << solution.Rotting_Oranges_BFS(grid1) << endl;
    cout << "Test 1 - Brute Force: " << solution.Rotting_Oranges_Brute_Force(grid1_copy) << endl;
    
    vector<vector<int>> grid2 = {{2,1,1},{0,1,1},{1,0,1}};
    vector<vector<int>> grid2_copy = grid2;
    cout << "Test 2 - BFS: " << solution.Rotting_Oranges_BFS(grid2) << endl;
    cout << "Test 2 - Brute Force: " << solution.Rotting_Oranges_Brute_Force(grid2_copy) << endl;
    
    vector<vector<int>> grid3 = {{0,2}};
    vector<vector<int>> grid3_copy = grid3;
    cout << "Test 3 - BFS: " << solution.Rotting_Oranges_BFS(grid3) << endl;
    cout << "Test 3 - Brute Force: " << solution.Rotting_Oranges_Brute_Force(grid3_copy) << endl;
}

int main() {
    Test_Rotting_Oranges();
    return 0;
}
