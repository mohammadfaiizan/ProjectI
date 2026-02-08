/*
Problem: Rat in a Maze / Search in Maze
URL: https://practice.geeksforgeeks.org/problems/rat-in-a-maze-problem/1

Problem Statement:
Given a maze (N x N matrix with 0s and 1s), find all paths from (0,0) to (N-1,N-1). Can move in all 4 directions (D,L,R,U).

Sample Input/Output:
Input: 4x4 maze with blocked cells
Output: All valid paths as direction strings
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Search_Maze_Backtracking_Helper(int row, int col, int n, vector<vector<int>>& maze, vector<string>& result, string& path, vector<vector<bool>>& visited) {
        if (row == n - 1 && col == n - 1) {
            result.push_back(path);
            return;
        }
        
        int directions[4][2] = {{1, 0}, {0, -1}, {0, 1}, {-1, 0}};
        char dirChars[4] = {'D', 'L', 'R', 'U'};
        
        for (int i = 0; i < 4; i++) {
            int newRow = row + directions[i][0];
            int newCol = col + directions[i][1];
            
            if (newRow >= 0 && newRow < n && newCol >= 0 && newCol < n && 
                maze[newRow][newCol] == 1 && !visited[newRow][newCol]) {
                visited[newRow][newCol] = true;
                path.push_back(dirChars[i]);
                Search_Maze_Backtracking_Helper(newRow, newCol, n, maze, result, path, visited);
                path.pop_back();
                visited[newRow][newCol] = false;
            }
        }
    }

    vector<string> Search_Maze_Backtracking(int n, vector<vector<int>>& maze) {
        /*
        DFS Backtracking - All Paths
        Time Complexity: O(4^(n^2))
        Space Complexity: O(n^2)
        */
        vector<string> result;
        string path;
        vector<vector<bool>> visited(n, vector<bool>(n, false));
        
        if (maze[0][0] == 1) {
            visited[0][0] = true;
            Search_Maze_Backtracking_Helper(0, 0, n, maze, result, path, visited);
        }
        
        return result;
    }

    bool Search_Maze_Single_Path_Helper(int row, int col, int n, vector<vector<int>>& maze, string& path, vector<vector<bool>>& visited) {
        if (row == n - 1 && col == n - 1) {
            return true;
        }
        
        int directions[4][2] = {{1, 0}, {0, -1}, {0, 1}, {-1, 0}};
        char dirChars[4] = {'D', 'L', 'R', 'U'};
        
        for (int i = 0; i < 4; i++) {
            int newRow = row + directions[i][0];
            int newCol = col + directions[i][1];
            
            if (newRow >= 0 && newRow < n && newCol >= 0 && newCol < n && 
                maze[newRow][newCol] == 1 && !visited[newRow][newCol]) {
                visited[newRow][newCol] = true;
                path.push_back(dirChars[i]);
                
                if (Search_Maze_Single_Path_Helper(newRow, newCol, n, maze, path, visited)) {
                    return true;
                }
                
                path.pop_back();
                visited[newRow][newCol] = false;
            }
        }
        
        return false;
    }

    string Search_Maze_Single_Path(int n, vector<vector<int>>& maze) {
        /*
        Find Any One Path using DFS
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        */
        string path;
        vector<vector<bool>> visited(n, vector<bool>(n, false));
        
        if (maze[0][0] == 1) {
            visited[0][0] = true;
            if (Search_Maze_Single_Path_Helper(0, 0, n, maze, path, visited)) {
                return path;
            }
        }
        
        return "";
    }
};

void Test_Search_In_Maze() {
    Solution solution;
    
    cout << "Test: 4x4 Maze" << endl;
    int n = 4;
    vector<vector<int>> maze = {
        {1, 0, 0, 0},
        {1, 1, 0, 1},
        {1, 1, 0, 0},
        {0, 1, 1, 1}
    };
    
    vector<string> allPaths = solution.Search_Maze_Backtracking(n, maze);
    cout << "All paths found: " << allPaths.size() << endl;
    for (const string& path : allPaths) {
        cout << path << " ";
    }
    cout << endl;
    
    string singlePath = solution.Search_Maze_Single_Path(n, maze);
    cout << "\nSingle path: " << (singlePath.empty() ? "No path" : singlePath) << endl;
}

int main() {
    Test_Search_In_Maze();
    return 0;
}
