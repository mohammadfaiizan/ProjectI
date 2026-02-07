/*
Problem: Count of Number of Given String in a 2D Character Array
URL: https://www.geeksforgeeks.org/find-count-number-given-string-present-2d-character-array/

Problem Statement:
Given a 2D character array and a string, find the count of occurrences of the
string in the 2D array. The string can be searched in all 4 directions
(up, down, left, right) and can bend at any point.

Sample Input/Output:
Input: grid = {{"BBABBM"}, {"CBMBBA"}, {"IBABBG"}, {"GOZBBI"}, {"ABBBBC"}, {"MCIGAM"}}
       word = "MAGIC"
Output: 3
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Count_String_Backtrack(vector<string>& grid, string word) {
        /*
        Backtracking DFS - search from every cell
        Time Complexity: O(R * C * 4^L) where L = word length
        Space Complexity: O(L) recursion stack
        */
        int R = grid.size(), C = grid[0].size();
        int found = 0;
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                found += DFS(grid, word, i, j, 0, R, C);
            }
        }
        return found;
    }

    int Count_String_Eight_Dir(vector<vector<char>>& grid, string word) {
        /*
        Search in all 8 directions (straight lines only, no bending)
        Time Complexity: O(R * C * 8 * L)
        Space Complexity: O(1)
        */
        int R = grid.size(), C = grid[0].size();
        int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};
        int count = 0;

        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                for (int d = 0; d < 8; d++) {
                    int k;
                    int rd = i, cd = j;
                    for (k = 0; k < (int)word.size(); k++) {
                        if (rd < 0 || rd >= R || cd < 0 || cd >= C) break;
                        if (grid[rd][cd] != word[k]) break;
                        rd += dx[d];
                        cd += dy[d];
                    }
                    if (k == (int)word.size()) count++;
                }
            }
        }
        return count;
    }

private:
    int DFS(vector<string>& grid, string& word, int r, int c, int idx, int R, int C) {
        if (idx == (int)word.size()) return 1;
        if (r < 0 || r >= R || c < 0 || c >= C) return 0;
        if (grid[r][c] != word[idx]) return 0;

        char temp = grid[r][c];
        grid[r][c] = '#';
        int found = 0;
        int dr[] = {0, 0, 1, -1};
        int dc[] = {1, -1, 0, 0};
        for (int d = 0; d < 4; d++) {
            found += DFS(grid, word, r + dr[d], c + dc[d], idx + 1, R, C);
        }
        grid[r][c] = temp;
        return found;
    }
};

void Test_Count_String_In_2D_Grid() {
    Solution sol;

    vector<string> grid1 = {"BBABBM", "CBMBBA", "IBABBG", "GOZBBI", "ABBBBC", "MCIGAM"};
    string word1 = "MAGIC";
    cout << "Grid 1, Word: " << word1 << endl;
    cout << "Backtrack: " << sol.Count_String_Backtrack(grid1, word1) << endl;

    vector<vector<char>> grid2 = {
        {'A','B','C'},
        {'D','E','F'},
        {'G','H','I'}
    };
    string word2 = "ABC";
    cout << "Grid 2, Word: " << word2 << endl;
    cout << "Eight Dir: " << sol.Count_String_Eight_Dir(grid2, word2) << endl;

    cout << string(50, '-') << endl;
}

int main() {
    Test_Count_String_In_2D_Grid();
    return 0;
}
