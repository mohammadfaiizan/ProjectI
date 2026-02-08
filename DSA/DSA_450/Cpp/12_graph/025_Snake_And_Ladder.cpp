/*
Problem: Snakes and Ladders
URL: https://leetcode.com/problems/snakes-and-ladders/

Problem Statement:
Given a snakes and ladders board, find the minimum number of dice throws required to reach the end of the board. The board is represented as a 2D array where -1 means no snake or ladder, and a positive number means a snake or ladder that takes you to that cell.

Sample Input/Output:
Input: board = [[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,35,-1,-1,13,-1],[-1,-1,-1,-1,-1,-1],[-1,15,-1,-1,-1,-1]]
Output: 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Snake_Ladder_BFS(vector<vector<int>>& board) {
        /*
        BFS from cell 1, handle snakes/ladders as edges
        Time Complexity: O(N)
        Space Complexity: O(N)
        */
        int n = board.size();
        int total = n * n;
        vector<int> dist(total + 1, -1);
        queue<int> q;
        
        dist[1] = 0;
        q.push(1);
        
        while (!q.empty()) {
            int curr = q.front();
            q.pop();
            
            if (curr == total) {
                return dist[curr];
            }
            
            for (int dice = 1; dice <= 6 && curr + dice <= total; dice++) {
                int next = curr + dice;
                int row = n - 1 - (next - 1) / n;
                int col = (next - 1) % n;
                if ((n - 1 - row) % 2 == 1) {
                    col = n - 1 - col;
                }
                
                if (board[row][col] != -1) {
                    next = board[row][col];
                }
                
                if (dist[next] == -1) {
                    dist[next] = dist[curr] + 1;
                    q.push(next);
                }
            }
        }
        
        return -1;
    }
};

void Test_Snake_Ladder_BFS() {
    Solution solution;
    
    vector<vector<int>> board1 = {
        {-1,-1,-1,-1,-1,-1},
        {-1,-1,-1,-1,-1,-1},
        {-1,-1,-1,-1,-1,-1},
        {-1,35,-1,-1,13,-1},
        {-1,-1,-1,-1,-1,-1},
        {-1,15,-1,-1,-1,-1}
    };
    cout << "Test 1: " << solution.Snake_Ladder_BFS(board1) << endl;
    
    vector<vector<int>> board2 = {
        {-1,-1},
        {-1,3}
    };
    cout << "Test 2: " << solution.Snake_Ladder_BFS(board2) << endl;
    
    vector<vector<int>> board3 = {
        {-1,1,2,-1},
        {2,13,15,-1},
        {-1,10,-1,-1},
        {-1,6,2,8}
    };
    cout << "Test 3: " << solution.Snake_Ladder_BFS(board3) << endl;
}

int main() {
    Test_Snake_Ladder_BFS();
    return 0;
}
