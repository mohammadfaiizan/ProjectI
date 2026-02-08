/*
Problem: Number of Islands
URL: https://leetcode.com/problems/number-of-islands/

Problem Statement:
Given a 2D grid of '1' (land) and '0' (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.

Sample Input/Output:
Input: grid = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
Output: 1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Islands_DFS(vector<vector<char>>& grid) {
        /*
        DFS flood fill
        Time Complexity: O(M*N)
        Space Complexity: O(M*N)
        */
        if (grid.empty() || grid[0].empty()) return 0;
        
        int m = grid.size();
        int n = grid[0].size();
        int count = 0;
        
        function<void(int, int)> dfs = [&](int i, int j) {
            if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] != '1') {
                return;
            }
            grid[i][j] = '0';
            dfs(i+1, j);
            dfs(i-1, j);
            dfs(i, j+1);
            dfs(i, j-1);
        };
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    dfs(i, j);
                }
            }
        }
        
        return count;
    }
    
    int Islands_BFS(vector<vector<char>>& grid) {
        /*
        BFS flood fill
        Time Complexity: O(M*N)
        Space Complexity: O(M*N)
        */
        if (grid.empty() || grid[0].empty()) return 0;
        
        int m = grid.size();
        int n = grid[0].size();
        int count = 0;
        int dx[] = {1, -1, 0, 0};
        int dy[] = {0, 0, 1, -1};
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    queue<pair<int, int>> q;
                    q.push({i, j});
                    grid[i][j] = '0';
                    
                    while (!q.empty()) {
                        auto [x, y] = q.front();
                        q.pop();
                        
                        for (int k = 0; k < 4; k++) {
                            int nx = x + dx[k];
                            int ny = y + dy[k];
                            if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == '1') {
                                grid[nx][ny] = '0';
                                q.push({nx, ny});
                            }
                        }
                    }
                }
            }
        }
        
        return count;
    }
    
    int Islands_Union_Find(vector<vector<char>>& grid) {
        /*
        Disjoint Set Union
        Time Complexity: O(M*N)
        Space Complexity: O(M*N)
        */
        if (grid.empty() || grid[0].empty()) return 0;
        
        int m = grid.size();
        int n = grid[0].size();
        vector<int> parent(m * n);
        vector<int> rank(m * n, 0);
        int count = 0;
        
        for (int i = 0; i < m * n; i++) {
            parent[i] = i;
        }
        
        function<int(int)> find = [&](int x) -> int {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        };
        
        function<void(int, int)> unite = [&](int x, int y) {
            x = find(x);
            y = find(y);
            if (x != y) {
                if (rank[x] < rank[y]) swap(x, y);
                parent[y] = x;
                if (rank[x] == rank[y]) rank[x]++;
                count--;
            }
        };
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    int idx = i * n + j;
                    if (i > 0 && grid[i-1][j] == '1') {
                        unite(idx, (i-1) * n + j);
                    }
                    if (j > 0 && grid[i][j-1] == '1') {
                        unite(idx, i * n + (j-1));
                    }
                }
            }
        }
        
        return count;
    }
};

void Test_Islands() {
    Solution solution;
    
    cout << "Test Case 1: Single island" << endl;
    vector<vector<char>> grid1 = {
        {'1','1','1','1','0'},
        {'1','1','0','1','0'},
        {'1','1','0','0','0'},
        {'0','0','0','0','0'}
    };
    vector<vector<char>> grid1_copy1 = grid1;
    vector<vector<char>> grid1_copy2 = grid1;
    cout << "DFS Result: " << solution.Islands_DFS(grid1_copy1) << endl;
    cout << "BFS Result: " << solution.Islands_BFS(grid1_copy2) << endl;
    cout << "Union-Find Result: " << solution.Islands_Union_Find(grid1) << endl;
    
    cout << "\nTest Case 2: Multiple islands" << endl;
    vector<vector<char>> grid2 = {
        {'1','1','0','0','0'},
        {'1','1','0','0','0'},
        {'0','0','1','0','0'},
        {'0','0','0','1','1'}
    };
    vector<vector<char>> grid2_copy1 = grid2;
    vector<vector<char>> grid2_copy2 = grid2;
    cout << "DFS Result: " << solution.Islands_DFS(grid2_copy1) << endl;
    cout << "BFS Result: " << solution.Islands_BFS(grid2_copy2) << endl;
    cout << "Union-Find Result: " << solution.Islands_Union_Find(grid2) << endl;
}

int main() {
    Test_Islands();
    return 0;
}
