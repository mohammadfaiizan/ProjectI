/*
Problem: Knights Tour
URL: https://www.geeksforgeeks.org/the-knights-tour-problem-backtracking-1/

Problem Statement:
Find a knight's tour on NxN chessboard - visit every square exactly once.

Sample Input/Output:
Input: N=8, start=(0,0)
Output: 8x8 matrix with move numbers
Explanation: Knight visits all 64 squares exactly once
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> Knights_Tour_Backtracking(int n, int start_row, int start_col) {
        /*
        Backtracking with 8 moves
        Time Complexity: O(8^(N^2))
        Space Complexity: O(N^2)
        */
        vector<vector<int>> board(n, vector<int>(n, -1));
        int move_count = 0;
        
        int moves[8][2] = {{2,1},{1,2},{-1,2},{-2,1},{-2,-1},{-1,-2},{1,-2},{2,-1}};
        
        function<bool(int, int, int)> Is_Safe = [&](int row, int col, int move_num) {
            return row >= 0 && row < n && col >= 0 && col < n && board[row][col] == -1;
        };
        
        function<bool(int, int, int)> backtrack = [&](int row, int col, int move_num) {
            board[row][col] = move_num;
            
            if (move_num == n * n - 1) {
                return true;
            }
            
            for (int i = 0; i < 8; i++) {
                int new_row = row + moves[i][0];
                int new_col = col + moves[i][1];
                
                if (Is_Safe(new_row, new_col, move_num + 1)) {
                    if (backtrack(new_row, new_col, move_num + 1)) {
                        return true;
                    }
                }
            }
            
            board[row][col] = -1;
            return false;
        };
        
        if (backtrack(start_row, start_col, 0)) {
            return board;
        }
        
        return vector<vector<int>>();
    }
};

void Test_Knights_Tour() {
    Solution solution;
    
    int n = 8;
    vector<vector<int>> tour = solution.Knights_Tour_Backtracking(n, 0, 0);
    
    if (!tour.empty()) {
        cout << "Knight's Tour found:" << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << setw(3) << tour[i][j] << " ";
            }
            cout << endl;
        }
    } else {
        cout << "No solution found" << endl;
    }
}

int main() {
    Test_Knights_Tour();
    return 0;
}
