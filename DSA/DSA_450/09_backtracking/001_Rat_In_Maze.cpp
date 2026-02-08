/*
Problem: Rat In Maze
URL: https://practice.geeksforgeeks.org/problems/rat-in-a-maze-problem/1

Problem Statement:
Given NxN maze (0=blocked, 1=open), find all paths from (0,0) to (N-1,N-1). Can move D,L,R,U. Print paths in sorted order.

Sample Input/Output:
Input: N=4, maze[][] = {{1,0,0,0},{1,1,0,1},{1,1,0,0},{0,1,1,1}}
Output: DDRDRR DRDDRR
Explanation: Two paths exist from (0,0) to (3,3)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<string> Find_Path_DFS_Backtracking(vector<vector<int>> &maze, int n) {
        /*
        DFS backtracking with visited array
        Time Complexity: O(4^(n^2))
        Space Complexity: O(n^2)
        */
        vector<string> result;
        vector<vector<bool>> visited(n, vector<bool>(n, false));
        string path = "";
        
        if (maze[0][0] == 0) return result;
        
        function<void(int, int, string)> dfs = [&](int row, int col, string current_path) {
            if (row == n - 1 && col == n - 1) {
                result.push_back(current_path);
                return;
            }
            
            visited[row][col] = true;
            
            int directions[4][2] = {{1, 0}, {0, -1}, {0, 1}, {-1, 0}};
            char moves[4] = {'D', 'L', 'R', 'U'};
            
            for (int i = 0; i < 4; i++) {
                int new_row = row + directions[i][0];
                int new_col = col + directions[i][1];
                
                if (new_row >= 0 && new_row < n && new_col >= 0 && new_col < n &&
                    !visited[new_row][new_col] && maze[new_row][new_col] == 1) {
                    dfs(new_row, new_col, current_path + moves[i]);
                }
            }
            
            visited[row][col] = false;
        };
        
        dfs(0, 0, path);
        sort(result.begin(), result.end());
        return result;
    }
    
    int Count_Paths_DP(vector<vector<int>> &maze, int n) {
        /*
        DP count paths
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        if (maze[0][0] == 0 || maze[n-1][n-1] == 0) return 0;
        
        vector<vector<int>> dp(n, vector<int>(n, 0));
        dp[0][0] = 1;
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (maze[i][j] == 1) {
                    if (i > 0) dp[i][j] += dp[i-1][j];
                    if (j > 0) dp[i][j] += dp[i][j-1];
                }
            }
        }
        
        return dp[n-1][n-1];
    }
};

void Test_Rat_In_Maze() {
    Solution solution;
    
    vector<vector<int>> maze1 = {{1,0,0,0},{1,1,0,1},{1,1,0,0},{0,1,1,1}};
    vector<string> paths = solution.Find_Path_DFS_Backtracking(maze1, 4);
    cout << "Paths: ";
    for (const string &path : paths) {
        cout << path << " ";
    }
    cout << endl;
    
    int count = solution.Count_Paths_DP(maze1, 4);
    cout << "Total paths count: " << count << endl;
}

int main() {
    Test_Rat_In_Maze();
    return 0;
}
