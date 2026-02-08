/*
Problem: Longest Route in Matrix
URL: https://www.geeksforgeeks.org/longest-possible-route-in-a-matrix-with-hurdles/

Problem Statement:
Find the longest path in a matrix from source to destination with hurdles (0 = blocked). Can move in 4 directions.

Sample Input/Output:
Input: Matrix = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
       Source = (0, 0), Destination = (1, 7)
Output: 24
Explanation: Longest path length is 24
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Longest_Route_Matrix_DFS(vector<vector<int>>& matrix, pair<int, int> source, pair<int, int> destination) {
        /*
        DFS backtracking
        Time Complexity: O(4^(R*C))
        Space Complexity: O(R*C)
        */
        int R = matrix.size();
        int C = matrix[0].size();
        vector<vector<int>> visited(R, vector<int>(C, 0));
        int max_path = -1;
        
        function<void(int, int, int)> dfs = [&](int x, int y, int len) {
            if (x == destination.first && y == destination.second) {
                max_path = max(max_path, len);
                return;
            }
            
            int dx[] = {-1, 1, 0, 0};
            int dy[] = {0, 0, -1, 1};
            
            for (int k = 0; k < 4; k++) {
                int nx = x + dx[k];
                int ny = y + dy[k];
                if (nx >= 0 && nx < R && ny >= 0 && ny < C && matrix[nx][ny] == 1 && visited[nx][ny] == 0) {
                    visited[nx][ny] = 1;
                    dfs(nx, ny, len + 1);
                    visited[nx][ny] = 0;
                }
            }
        };
        
        if (matrix[source.first][source.second] == 1) {
            visited[source.first][source.second] = 1;
            dfs(source.first, source.second, 0);
        }
        
        return max_path;
    }
};

void Test_Longest_Route_Matrix() {
    Solution solution;
    vector<vector<int>> matrix = {
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 0, 1, 1, 0, 1, 1, 0, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    };
    pair<int, int> source = {0, 0};
    pair<int, int> destination = {1, 7};
    int result = solution.Longest_Route_Matrix_DFS(matrix, source, destination);
    cout << "Longest path length: " << result << endl;
}

int main() {
    Test_Longest_Route_Matrix();
    return 0;
}
