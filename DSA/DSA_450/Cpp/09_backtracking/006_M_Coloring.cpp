/*
Problem: M Coloring
URL: https://practice.geeksforgeeks.org/problems/m-coloring-problem-1587115620/1

Problem Statement:
Given undirected graph and M colors, determine if the graph can be colored with at most M colors such that no two adjacent vertices have same color.

Sample Input/Output:
Input: N=4, M=3, E=5, edges={{0,1},{1,2},{2,3},{3,0},{0,2}}
Output: true
Explanation: Graph can be colored with 3 colors
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Graph_Coloring_Backtracking(int n, int m, vector<vector<int>> &edges) {
        /*
        Backtracking
        Time Complexity: O(m^V)
        Space Complexity: O(V)
        */
        vector<vector<int>> graph(n);
        for (const auto &edge : edges) {
            graph[edge[0]].push_back(edge[1]);
            graph[edge[1]].push_back(edge[0]);
        }
        
        vector<int> color(n, 0);
        
        function<bool(int, int)> Is_Safe = [&](int vertex, int c) {
            for (int neighbor : graph[vertex]) {
                if (color[neighbor] == c) {
                    return false;
                }
            }
            return true;
        };
        
        function<bool(int)> backtrack = [&](int vertex) {
            if (vertex == n) {
                return true;
            }
            
            for (int c = 1; c <= m; c++) {
                if (Is_Safe(vertex, c)) {
                    color[vertex] = c;
                    if (backtrack(vertex + 1)) {
                        return true;
                    }
                    color[vertex] = 0;
                }
            }
            
            return false;
        };
        
        return backtrack(0);
    }
};

void Test_M_Coloring() {
    Solution solution;
    
    int n = 4, m = 3;
    vector<vector<int>> edges = {{0,1},{1,2},{2,3},{3,0},{0,2}};
    
    bool can_color = solution.Graph_Coloring_Backtracking(n, m, edges);
    cout << "Can color graph with " << m << " colors: " << (can_color ? "Yes" : "No") << endl;
}

int main() {
    Test_M_Coloring();
    return 0;
}
