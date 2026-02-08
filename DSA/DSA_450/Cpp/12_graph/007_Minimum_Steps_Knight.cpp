/*
Problem: Minimum Steps by a Knight
URL: https://practice.geeksforgeeks.org/problems/steps-by-knight5927/1

Problem Statement:
Find minimum steps for a knight to reach from source to target on an N x N chessboard.

Sample Input/Output:
Input: N=6, source=(4,5), target=(1,1)
Output: Minimum steps: 3
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Steps_Knight_BFS(int N, pair<int, int> source, pair<int, int> target) {
        /*
        BFS Exploring All 8 Knight Moves
        Time Complexity: O(N^2)
        Space Complexity: O(N^2)
        */
        if (source.first == target.first && source.second == target.second) {
            return 0;
        }
        
        vector<vector<bool>> visited(N, vector<bool>(N, false));
        queue<pair<pair<int, int>, int>> q;
        
        int dx[] = {-2, -1, 1, 2, 2, 1, -1, -2};
        int dy[] = {1, 2, 2, 1, -1, -2, -2, -1};
        
        visited[source.first][source.second] = true;
        q.push({{source.first, source.second}, 0});
        
        while (!q.empty()) {
            auto current = q.front();
            q.pop();
            
            int x = current.first.first;
            int y = current.first.second;
            int steps = current.second;
            
            for (int i = 0; i < 8; i++) {
                int newX = x + dx[i];
                int newY = y + dy[i];
                
                if (newX >= 0 && newX < N && newY >= 0 && newY < N && !visited[newX][newY]) {
                    if (newX == target.first && newY == target.second) {
                        return steps + 1;
                    }
                    
                    visited[newX][newY] = true;
                    q.push({{newX, newY}, steps + 1});
                }
            }
        }
        
        return -1;
    }
};

void Test_Minimum_Steps_Knight() {
    Solution solution;
    
    cout << "Test 1: N=6, source=(4,5), target=(1,1)" << endl;
    int steps1 = solution.Min_Steps_Knight_BFS(6, {4, 5}, {1, 1});
    cout << "Minimum steps: " << steps1 << endl;
    
    cout << "\nTest 2: N=8, source=(0,0), target=(7,7)" << endl;
    int steps2 = solution.Min_Steps_Knight_BFS(8, {0, 0}, {7, 7});
    cout << "Minimum steps: " << steps2 << endl;
    
    cout << "\nTest 3: N=5, source=(0,0), target=(4,4)" << endl;
    int steps3 = solution.Min_Steps_Knight_BFS(5, {0, 0}, {4, 4});
    cout << "Minimum steps: " << steps3 << endl;
}

int main() {
    Test_Minimum_Steps_Knight();
    return 0;
}
