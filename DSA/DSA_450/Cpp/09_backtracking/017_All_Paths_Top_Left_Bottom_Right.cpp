/*
Problem: All Paths Top Left to Bottom Right
URL: https://www.geeksforgeeks.org/print-all-possible-paths-from-top-left-to-bottom-right-of-a-mxn-matrix/

Problem Statement:
Print all possible paths from top-left to bottom-right in MxN matrix. Can only move right or down.

Sample Input/Output:
Input: Matrix 2x2
Output: 
DR
RD
Explanation: D=Down, R=Right
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<string> All_Paths_Backtracking(int m, int n) {
        /*
        Backtracking
        Time Complexity: O(2^(m+n))
        Space Complexity: O(m+n)
        */
        vector<string> result;
        string path;
        
        function<void(int, int)> backtrack = [&](int x, int y) {
            if (x == m - 1 && y == n - 1) {
                result.push_back(path);
                return;
            }
            
            if (x < m - 1) {
                path.push_back('D');
                backtrack(x + 1, y);
                path.pop_back();
            }
            
            if (y < n - 1) {
                path.push_back('R');
                backtrack(x, y + 1);
                path.pop_back();
            }
        };
        
        backtrack(0, 0);
        return result;
    }
    
    vector<string> All_Paths_Iterative(int m, int n) {
        /*
        Iterative using queue
        Time Complexity: O(2^(m+n))
        Space Complexity: O(2^(m+n))
        */
        vector<string> result;
        queue<pair<pair<int, int>, string>> q;
        q.push({{0, 0}, ""});
        
        while (!q.empty()) {
            auto [pos, path] = q.front();
            auto [x, y] = pos;
            q.pop();
            
            if (x == m - 1 && y == n - 1) {
                result.push_back(path);
                continue;
            }
            
            if (x < m - 1) {
                q.push({{x + 1, y}, path + "D"});
            }
            
            if (y < n - 1) {
                q.push({{x, y + 1}, path + "R"});
            }
        }
        
        return result;
    }
};

void Test_All_Paths_Top_Left_Bottom_Right() {
    Solution solution;
    int m = 2, n = 2;
    vector<string> result1 = solution.All_Paths_Backtracking(m, n);
    vector<string> result2 = solution.All_Paths_Iterative(m, n);
    
    cout << "Backtracking Approach:" << endl;
    for (string& path : result1) {
        cout << path << endl;
    }
    
    cout << "Iterative Approach:" << endl;
    for (string& path : result2) {
        cout << path << endl;
    }
}

int main() {
    Test_All_Paths_Top_Left_Bottom_Right();
    return 0;
}
