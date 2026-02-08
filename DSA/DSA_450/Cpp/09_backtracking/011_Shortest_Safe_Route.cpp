/*
Problem: Shortest Safe Route
URL: https://www.geeksforgeeks.org/find-shortest-safe-route-in-a-path-with-landmines/

Problem Statement:
Given a matrix with landmines (marked as 0), find the shortest safe route from any cell in the first column to any cell in the last column. Adjacent cells to landmines are also unsafe.

Sample Input/Output:
Input: 
Matrix = [[1, 1, 1, 1, 1],
          [1, 0, 1, 1, 1],
          [1, 1, 1, 1, 1],
          [1, 1, 1, 0, 1],
          [1, 0, 1, 1, 1]]
Output: 6
Explanation: Shortest path length is 6 from (0,0) to (4,4)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Shortest_Safe_Route_BFS(vector<vector<int>>& matrix) {
        /*
        BFS after marking unsafe cells
        Time Complexity: O(R*C)
        Space Complexity: O(R*C)
        */
        int R = matrix.size();
        int C = matrix[0].size();
        vector<vector<int>> safe(R, vector<int>(C, 1));
        
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                if (matrix[i][j] == 0) {
                    safe[i][j] = 0;
                    int dx[] = {-1, 1, 0, 0};
                    int dy[] = {0, 0, -1, 1};
                    for (int k = 0; k < 4; k++) {
                        int ni = i + dx[k];
                        int nj = j + dy[k];
                        if (ni >= 0 && ni < R && nj >= 0 && nj < C) {
                            safe[ni][nj] = 0;
                        }
                    }
                }
            }
        }
        
        queue<pair<int, int>> q;
        vector<vector<int>> dist(R, vector<int>(C, -1));
        
        for (int i = 0; i < R; i++) {
            if (safe[i][0] == 1) {
                q.push({i, 0});
                dist[i][0] = 1;
            }
        }
        
        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};
        
        while (!q.empty()) {
            auto [x, y] = q.front();
            q.pop();
            
            if (y == C - 1) {
                return dist[x][y];
            }
            
            for (int k = 0; k < 4; k++) {
                int nx = x + dx[k];
                int ny = y + dy[k];
                if (nx >= 0 && nx < R && ny >= 0 && ny < C && safe[nx][ny] == 1 && dist[nx][ny] == -1) {
                    dist[nx][ny] = dist[x][y] + 1;
                    q.push({nx, ny});
                }
            }
        }
        
        return -1;
    }
    
    int Shortest_Safe_Route_Backtracking(vector<vector<int>>& matrix) {
        /*
        Backtracking DFS approach
        Time Complexity: O(4^(R*C))
        Space Complexity: O(R*C)
        */
        int R = matrix.size();
        int C = matrix[0].size();
        vector<vector<int>> safe(R, vector<int>(C, 1));
        
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                if (matrix[i][j] == 0) {
                    safe[i][j] = 0;
                    int dx[] = {-1, 1, 0, 0};
                    int dy[] = {0, 0, -1, 1};
                    for (int k = 0; k < 4; k++) {
                        int ni = i + dx[k];
                        int nj = j + dy[k];
                        if (ni >= 0 && ni < R && nj >= 0 && nj < C) {
                            safe[ni][nj] = 0;
                        }
                    }
                }
            }
        }
        
        int min_path = INT_MAX;
        vector<vector<int>> visited(R, vector<int>(C, 0));
        
        function<void(int, int, int)> dfs = [&](int x, int y, int len) {
            if (y == C - 1) {
                min_path = min(min_path, len);
                return;
            }
            
            int dx[] = {-1, 1, 0, 0};
            int dy[] = {0, 0, -1, 1};
            
            for (int k = 0; k < 4; k++) {
                int nx = x + dx[k];
                int ny = y + dy[k];
                if (nx >= 0 && nx < R && ny >= 0 && ny < C && safe[nx][ny] == 1 && visited[nx][ny] == 0) {
                    visited[nx][ny] = 1;
                    dfs(nx, ny, len + 1);
                    visited[nx][ny] = 0;
                }
            }
        };
        
        for (int i = 0; i < R; i++) {
            if (safe[i][0] == 1) {
                visited[i][0] = 1;
                dfs(i, 0, 1);
                visited[i][0] = 0;
            }
        }
        
        return min_path == INT_MAX ? -1 : min_path;
    }
};

void Test_Shortest_Safe_Route() {
    Solution solution;
    vector<vector<int>> matrix = {
        {1, 1, 1, 1, 1},
        {1, 0, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 0, 1},
        {1, 0, 1, 1, 1}
    };
    cout << "BFS Approach: " << solution.Shortest_Safe_Route_BFS(matrix) << endl;
    cout << "Backtracking Approach: " << solution.Shortest_Safe_Route_Backtracking(matrix) << endl;
}

int main() {
    Test_Shortest_Safe_Route();
    return 0;
}
