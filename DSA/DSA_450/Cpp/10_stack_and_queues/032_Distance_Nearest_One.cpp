/*
Problem: Distance of Nearest Cell Having 1 in Binary Matrix
URL: https://practice.geeksforgeeks.org/problems/distance-of-nearest-cell-having-1-1587115620/1

Problem Statement:
Given a binary matrix of size N x M. For each cell of the matrix, find the distance of the nearest cell having 1 in the matrix.
Distance between two cells (x1, y1) and (x2, y2) is defined as |x1 - x2| + |y1 - y2|.

Sample Input/Output:
Input: grid = [[0,0,0],[0,1,0],[0,0,0]]
Output: [[2,1,2],[1,0,1],[2,1,2]]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> Distance_Nearest_One_BFS(vector<vector<int>>& grid) {
        /*
        Multi-source BFS from all 1s
        Time Complexity: O(N*M)
        Space Complexity: O(N*M)
        */
        int n = grid.size();
        int m = grid[0].size();
        vector<vector<int>> dist(n, vector<int>(m, -1));
        queue<pair<int, int>> q;
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == 1) {
                    dist[i][j] = 0;
                    q.push({i, j});
                }
            }
        }
        
        vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        while (!q.empty()) {
            auto [x, y] = q.front();
            q.pop();
            
            for (auto [dx, dy] : directions) {
                int nx = x + dx;
                int ny = y + dy;
                
                if (nx >= 0 && nx < n && ny >= 0 && ny < m && dist[nx][ny] == -1) {
                    dist[nx][ny] = dist[x][y] + 1;
                    q.push({nx, ny});
                }
            }
        }
        
        return dist;
    }
    
    vector<vector<int>> Distance_Nearest_One_Brute_Force(vector<vector<int>>& grid) {
        /*
        Brute force
        Time Complexity: O(N^2*M^2)
        Space Complexity: O(N*M)
        */
        int n = grid.size();
        int m = grid[0].size();
        vector<vector<int>> dist(n, vector<int>(m, INT_MAX));
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == 1) {
                    dist[i][j] = 0;
                } else {
                    for (int x = 0; x < n; x++) {
                        for (int y = 0; y < m; y++) {
                            if (grid[x][y] == 1) {
                                dist[i][j] = min(dist[i][j], abs(i - x) + abs(j - y));
                            }
                        }
                    }
                }
            }
        }
        
        return dist;
    }
};

void Test_Distance_Nearest_One() {
    Solution solution;
    
    vector<vector<int>> grid1 = {{0,0,0},{0,1,0},{0,0,0}};
    vector<vector<int>> result1 = solution.Distance_Nearest_One_BFS(grid1);
    cout << "Test 1 - BFS Result:" << endl;
    for (auto& row : result1) {
        for (int val : row) cout << val << " ";
        cout << endl;
    }
    
    vector<vector<int>> grid2 = {{0,0,0},{0,1,0},{1,0,1}};
    vector<vector<int>> result2 = solution.Distance_Nearest_One_BFS(grid2);
    cout << "Test 2 - BFS Result:" << endl;
    for (auto& row : result2) {
        for (int val : row) cout << val << " ";
        cout << endl;
    }
    
    vector<vector<int>> grid3 = {{1,0,1},{0,1,0},{1,0,1}};
    vector<vector<int>> result3 = solution.Distance_Nearest_One_BFS(grid3);
    cout << "Test 3 - BFS Result:" << endl;
    for (auto& row : result3) {
        for (int val : row) cout << val << " ";
        cout << endl;
    }
}

int main() {
    Test_Distance_Nearest_One();
    return 0;
}
