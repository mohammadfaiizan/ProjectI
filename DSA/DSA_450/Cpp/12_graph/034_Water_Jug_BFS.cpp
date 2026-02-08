/*
Problem: Water Jug Problem
URL: https://www.geeksforgeeks.org/water-jug-problem-using-bfs/

Problem Statement:
Given two jugs of capacities a and b, measure exactly d liters using BFS. You can fill, empty, or pour water between jugs.

Sample Input/Output:
Input: jug1=4, jug2=3, target=2
Output: true (can measure 2 liters)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Water_Jug_BFS(int a, int b, int d) {
        /*
        BFS with state (x,y) and all 6 operations
        Time Complexity: O(a*b)
        Space Complexity: O(a*b)
        */
        if (d > max(a, b) || d < 0) {
            return false;
        }
        
        if (d == 0) {
            return true;
        }
        
        set<pair<int, int>> visited;
        queue<pair<int, int>> q;
        q.push({0, 0});
        visited.insert({0, 0});
        
        while (!q.empty()) {
            auto [x, y] = q.front();
            q.pop();
            
            if (x == d || y == d || x + y == d) {
                return true;
            }
            
            vector<pair<int, int>> nextStates = {
                {a, y},
                {x, b},
                {0, y},
                {x, 0},
                {x - min(x, b - y), y + min(x, b - y)},
                {x + min(y, a - x), y - min(y, a - x)}
            };
            
            for (auto [nx, ny] : nextStates) {
                if (visited.find({nx, ny}) == visited.end()) {
                    visited.insert({nx, ny});
                    q.push({nx, ny});
                }
            }
        }
        
        return false;
    }
};

void Test_Water_Jug_BFS() {
    Solution solution;
    
    cout << "Test 1 (4,3,2): " << (solution.Water_Jug_BFS(4, 3, 2) ? "true" : "false") << endl;
    cout << "Test 2 (5,3,4): " << (solution.Water_Jug_BFS(5, 3, 4) ? "true" : "false") << endl;
    cout << "Test 3 (3,5,4): " << (solution.Water_Jug_BFS(3, 5, 4) ? "true" : "false") << endl;
    cout << "Test 4 (8,5,3): " << (solution.Water_Jug_BFS(8, 5, 3) ? "true" : "false") << endl;
    cout << "Test 5 (2,6,5): " << (solution.Water_Jug_BFS(2, 6, 5) ? "true" : "false") << endl;
}

int main() {
    Test_Water_Jug_BFS();
    return 0;
}
