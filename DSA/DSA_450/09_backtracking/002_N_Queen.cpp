/*
Problem: N Queen
URL: https://www.geeksforgeeks.org/printing-solutions-n-queen-problem/

Problem Statement:
Place N queens on NxN board so no two attack each other. Print all solutions.

Sample Input/Output:
Input: N=4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
Explanation: Two distinct solutions exist for 4-queen problem
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<string>> Solve_N_Queen_Backtracking(int n) {
        /*
        Backtracking with isSafe check
        Time Complexity: O(N!)
        Space Complexity: O(N^2)
        */
        vector<vector<string>> result;
        vector<string> board(n, string(n, '.'));
        
        function<bool(int, int, vector<string>&)> Is_Safe = [&](int row, int col, vector<string> &board) {
            for (int i = 0; i < row; i++) {
                if (board[i][col] == 'Q') return false;
            }
            
            for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
                if (board[i][j] == 'Q') return false;
            }
            
            for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
                if (board[i][j] == 'Q') return false;
            }
            
            return true;
        };
        
        function<void(int)> backtrack = [&](int row) {
            if (row == n) {
                result.push_back(board);
                return;
            }
            
            for (int col = 0; col < n; col++) {
                if (Is_Safe(row, col, board)) {
                    board[row][col] = 'Q';
                    backtrack(row + 1);
                    board[row][col] = '.';
                }
            }
        };
        
        backtrack(0);
        return result;
    }
    
    vector<vector<string>> Solve_N_Queen_Optimized(int n) {
        /*
        Optimized with row/diagonal arrays
        Time Complexity: O(N!)
        Space Complexity: O(N)
        */
        vector<vector<string>> result;
        vector<string> board(n, string(n, '.'));
        vector<bool> col_used(n, false);
        vector<bool> diag1(2 * n - 1, false);
        vector<bool> diag2(2 * n - 1, false);
        
        function<void(int)> backtrack = [&](int row) {
            if (row == n) {
                result.push_back(board);
                return;
            }
            
            for (int col = 0; col < n; col++) {
                int d1 = row + col;
                int d2 = row - col + n - 1;
                
                if (!col_used[col] && !diag1[d1] && !diag2[d2]) {
                    board[row][col] = 'Q';
                    col_used[col] = diag1[d1] = diag2[d2] = true;
                    backtrack(row + 1);
                    col_used[col] = diag1[d1] = diag2[d2] = false;
                    board[row][col] = '.';
                }
            }
        };
        
        backtrack(0);
        return result;
    }
};

void Test_N_Queen() {
    Solution solution;
    
    int n = 4;
    vector<vector<string>> solutions = solution.Solve_N_Queen_Backtracking(n);
    cout << "Number of solutions: " << solutions.size() << endl;
    
    for (int i = 0; i < solutions.size(); i++) {
        cout << "Solution " << i + 1 << ":" << endl;
        for (const string &row : solutions[i]) {
            cout << row << endl;
        }
        cout << endl;
    }
}

int main() {
    Test_N_Queen();
    return 0;
}
