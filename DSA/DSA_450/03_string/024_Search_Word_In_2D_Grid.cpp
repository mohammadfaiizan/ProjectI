/*
Problem: Search a Word in a 2D Grid (8 directions)
URL: https://practice.geeksforgeeks.org/problems/find-the-string-in-grid0111/1

Problem Statement:
Given a 2D grid of characters and a word, find all occurrences of the word in the grid.
The word can be matched in all 8 directions. Return the starting coordinates.

Sample Input/Output:
Input: grid = {{'a','b','c'},{'d','r','f'},{'g','h','i'}}, word = "abc"
Output: {{0,0}}
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> Search_Word_Eight_Dir(vector<vector<char>>& grid, string word) {
        /*
        Search in all 8 directions from each cell
        Time Complexity: O(R * C * 8 * L)
        Space Complexity: O(1) excluding result
        */
        int R = grid.size(), C = grid[0].size();
        int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};
        vector<vector<int>> ans;

        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                if (grid[i][j] != word[0]) continue;
                for (int d = 0; d < 8; d++) {
                    int k, rd = i + dx[d], cd = j + dy[d];
                    for (k = 1; k < (int)word.size(); k++) {
                        if (rd < 0 || rd >= R || cd < 0 || cd >= C) break;
                        if (grid[rd][cd] != word[k]) break;
                        rd += dx[d];
                        cd += dy[d];
                    }
                    if (k == (int)word.size()) {
                        ans.push_back({i, j});
                        break;
                    }
                }
            }
        }
        return ans;
    }

    vector<vector<int>> Search_Word_DFS(vector<vector<char>>& grid, string word) {
        /*
        DFS from each cell (4 directions with bending allowed)
        Time Complexity: O(R * C * 4^L)
        Space Complexity: O(L) recursion stack
        */
        int R = grid.size(), C = grid[0].size();
        vector<vector<int>> ans;
        vector<vector<bool>> visited(R, vector<bool>(C, false));

        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                if (DFS(grid, word, i, j, 0, visited)) {
                    ans.push_back({i, j});
                }
            }
        }
        return ans;
    }

private:
    bool DFS(vector<vector<char>>& grid, string& word, int r, int c, int idx,
             vector<vector<bool>>& visited) {
        if (idx == (int)word.size()) return true;
        int R = grid.size(), C = grid[0].size();
        if (r < 0 || r >= R || c < 0 || c >= C) return false;
        if (visited[r][c] || grid[r][c] != word[idx]) return false;

        visited[r][c] = true;
        int dr[] = {0, 0, 1, -1};
        int dc[] = {1, -1, 0, 0};
        for (int d = 0; d < 4; d++) {
            if (DFS(grid, word, r + dr[d], c + dc[d], idx + 1, visited))  {
                visited[r][c] = false;
                return true;
            }
        }
        visited[r][c] = false;
        return false;
    }
};

void Test_Search_Word_In_2D_Grid() {
    Solution sol;
    vector<vector<char>> grid = {
        {'a','b','c','d'},
        {'e','f','c','h'},
        {'i','j','b','a'},
        {'m','n','o','p'}
    };

    vector<string> words = {"abc", "abcba", "afj"};
    for (auto& word : words) {
        cout << "Word: " << word << endl;

        auto r1 = sol.Search_Word_Eight_Dir(grid, word);
        cout << "Eight Dir: ";
        for (auto& pos : r1) cout << "[" << pos[0] << "," << pos[1] << "] ";
        cout << endl;

        auto r2 = sol.Search_Word_DFS(grid, word);
        cout << "DFS: ";
        for (auto& pos : r2) cout << "[" << pos[0] << "," << pos[1] << "] ";
        cout << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Search_Word_In_2D_Grid();
    return 0;
}
