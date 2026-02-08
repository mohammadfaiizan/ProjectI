/*
Problem: Sudoku Solver
URL: https://practice.geeksforgeeks.org/problems/solve-the-sudoku-1587115621/1

Problem Statement:
Solve a 9x9 Sudoku puzzle using backtracking. Fill empty cells (0) with digits 1-9.

Sample Input/Output:
Input: grid[][] = {{3,0,6,5,0,8,4,0,0},{5,2,0,0,0,0,0,0,0},{0,8,7,0,0,0,0,3,1},{0,0,3,0,1,0,0,8,0},{9,0,0,8,6,3,0,0,5},{0,5,0,0,9,0,6,0,0},{1,3,0,0,0,0,2,5,0},{0,0,0,0,0,0,0,7,4},{0,0,5,2,0,6,3,0,0}}
Output: Solved grid
Explanation: Fill all zeros with valid digits 1-9
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Solve_Sudoku_Backtracking(vector<vector<int>> &grid) {
        /*
        Backtracking with row/col/box validation
        Time Complexity: O(9^(n*n))
        Space Complexity: O(n*n)
        */
        function<bool(int, int, int)> Is_Safe = [&](int row, int col, int num) {
            for (int i = 0; i < 9; i++) {
                if (grid[row][i] == num || grid[i][col] == num) {
                    return false;
                }
            }
            
            int box_row = (row / 3) * 3;
            int box_col = (col / 3) * 3;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (grid[box_row + i][box_col + j] == num) {
                        return false;
                    }
                }
            }
            
            return true;
        };
        
        function<bool()> solve = [&]() {
            for (int row = 0; row < 9; row++) {
                for (int col = 0; col < 9; col++) {
                    if (grid[row][col] == 0) {
                        for (int num = 1; num <= 9; num++) {
                            if (Is_Safe(row, col, num)) {
                                grid[row][col] = num;
                                if (solve()) {
                                    return true;
                                }
                                grid[row][col] = 0;
                            }
                        }
                        return false;
                    }
                }
            }
            return true;
        };
        
        return solve();
    }
};

void Test_Sudoku_Solver() {
    Solution solution;
    
    vector<vector<int>> grid = {
        {3,0,6,5,0,8,4,0,0},
        {5,2,0,0,0,0,0,0,0},
        {0,8,7,0,0,0,0,3,1},
        {0,0,3,0,1,0,0,8,0},
        {9,0,0,8,6,3,0,0,5},
        {0,5,0,0,9,0,6,0,0},
        {1,3,0,0,0,0,2,5,0},
        {0,0,0,0,0,0,0,7,4},
        {0,0,5,2,0,6,3,0,0}
    };
    
    bool solved = solution.Solve_Sudoku_Backtracking(grid);
    
    if (solved) {
        cout << "Sudoku solved:" << endl;
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                cout << grid[i][j] << " ";
            }
            cout << endl;
        }
    } else {
        cout << "No solution exists" << endl;
    }
}

int main() {
    Test_Sudoku_Solver();
    return 0;
}
