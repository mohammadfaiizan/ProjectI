/*
Problem: M-Coloring Problem
URL: https://practice.geeksforgeeks.org/problems/m-coloring-problem-1587115620/1

Problem Statement:
Check if the graph can be colored with at most M colors such that no two adjacent vertices have the same color. If yes, print the coloring.

Sample Input/Output:
Input: V=4, edges = [[0,1],[1,2],[2,3],[0,3]], m=3
Output: true, coloring = [0,1,0,1]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool M_Coloring_Backtracking(int V, vector<vector<int>>& edges, int m, vector<int>& coloring) {
        /*
        Try each color, backtrack if conflict
        Time Complexity: O(M^V)
        Space Complexity: O(V+E)
        */
        vector<vector<int>> adj(V);
        for (auto& e : edges) {
            adj[e[0]].push_back(e[1]);
            adj[e[1]].push_back(e[0]);
        }
        
        coloring.assign(V, -1);
        
        function<bool(int)> canColor = [&](int u) {
            if (u == V) {
                return true;
            }
            
            for (int c = 0; c < m; c++) {
                bool valid = true;
                for (int v : adj[u]) {
                    if (coloring[v] == c) {
                        valid = false;
                        break;
                    }
                }
                
                if (valid) {
                    coloring[u] = c;
                    if (canColor(u + 1)) {
                        return true;
                    }
                    coloring[u] = -1;
                }
            }
            
            return false;
        };
        
        return canColor(0);
    }
    
    bool M_Coloring_Greedy(int V, vector<vector<int>>& edges, int m, vector<int>& coloring) {
        /*
        Greedy assignment
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        vector<vector<int>> adj(V);
        for (auto& e : edges) {
            adj[e[0]].push_back(e[1]);
            adj[e[1]].push_back(e[0]);
        }
        
        coloring.assign(V, -1);
        vector<bool> used(m, false);
        
        for (int u = 0; u < V; u++) {
            fill(used.begin(), used.end(), false);
            
            for (int v : adj[u]) {
                if (coloring[v] != -1) {
                    used[coloring[v]] = true;
                }
            }
            
            int c;
            for (c = 0; c < m; c++) {
                if (!used[c]) {
                    break;
                }
            }
            
            if (c == m) {
                return false;
            }
            
            coloring[u] = c;
        }
        
        return true;
    }
};

void Test_M_Coloring_Backtracking() {
    Solution solution;
    
    int V1 = 4;
    vector<vector<int>> edges1 = {{0,1},{1,2},{2,3},{0,3}};
    vector<int> coloring1;
    bool result1 = solution.M_Coloring_Backtracking(V1, edges1, 3, coloring1);
    cout << "Test 1 Backtracking: " << (result1 ? "true" : "false");
    if (result1) {
        cout << ", Coloring: ";
        for (int c : coloring1) cout << c << " ";
    }
    cout << endl;
    
    int V2 = 3;
    vector<vector<int>> edges2 = {{0,1},{1,2},{0,2}};
    vector<int> coloring2;
    bool result2 = solution.M_Coloring_Backtracking(V2, edges2, 2, coloring2);
    cout << "Test 2 Backtracking: " << (result2 ? "true" : "false");
    if (result2) {
        cout << ", Coloring: ";
        for (int c : coloring2) cout << c << " ";
    }
    cout << endl;
    
    int V3 = 4;
    vector<vector<int>> edges3 = {{0,1},{1,2},{2,3},{3,0}};
    vector<int> coloring3;
    bool result3 = solution.M_Coloring_Greedy(V3, edges3, 2, coloring3);
    cout << "Test 3 Greedy: " << (result3 ? "true" : "false");
    if (result3) {
        cout << ", Coloring: ";
        for (int c : coloring3) cout << c << " ";
    }
    cout << endl;
}

int main() {
    Test_M_Coloring_Backtracking();
    return 0;
}
