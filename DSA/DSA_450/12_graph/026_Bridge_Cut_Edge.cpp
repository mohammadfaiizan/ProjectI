/*
Problem: Find Bridges in a Graph (Cut Edges)
URL: https://www.geeksforgeeks.org/bridge-in-a-graph/

Problem Statement:
Find all bridges (edges whose removal disconnects the graph) in an undirected graph using Tarjan's algorithm.

Sample Input/Output:
Input: V=4, edges = [[0,1],[1,2],[2,3]]
Output: [[0,1],[1,2],[2,3]]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> Bridges_Tarjan(int V, vector<vector<int>>& edges) {
        /*
        DFS with disc[] and low[]
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        vector<vector<int>> adj(V);
        for (auto& e : edges) {
            adj[e[0]].push_back(e[1]);
            adj[e[1]].push_back(e[0]);
        }
        
        vector<int> disc(V, -1);
        vector<int> low(V, -1);
        vector<int> parent(V, -1);
        vector<vector<int>> bridges;
        int time = 0;
        
        function<void(int)> dfs = [&](int u) {
            disc[u] = low[u] = time++;
            
            for (int v : adj[u]) {
                if (disc[v] == -1) {
                    parent[v] = u;
                    dfs(v);
                    low[u] = min(low[u], low[v]);
                    
                    if (low[v] > disc[u]) {
                        bridges.push_back({u, v});
                    }
                } else if (v != parent[u]) {
                    low[u] = min(low[u], disc[v]);
                }
            }
        };
        
        for (int i = 0; i < V; i++) {
            if (disc[i] == -1) {
                dfs(i);
            }
        }
        
        return bridges;
    }
};

void Test_Bridges_Tarjan() {
    Solution solution;
    
    int V1 = 4;
    vector<vector<int>> edges1 = {{0,1},{1,2},{2,3}};
    vector<vector<int>> result1 = solution.Bridges_Tarjan(V1, edges1);
    cout << "Test 1 Bridges: ";
    for (auto& b : result1) {
        cout << "[" << b[0] << "," << b[1] << "] ";
    }
    cout << endl;
    
    int V2 = 5;
    vector<vector<int>> edges2 = {{0,1},{1,2},{2,0},{1,3},{3,4}};
    vector<vector<int>> result2 = solution.Bridges_Tarjan(V2, edges2);
    cout << "Test 2 Bridges: ";
    for (auto& b : result2) {
        cout << "[" << b[0] << "," << b[1] << "] ";
    }
    cout << endl;
    
    int V3 = 3;
    vector<vector<int>> edges3 = {{0,1},{1,2},{2,0}};
    vector<vector<int>> result3 = solution.Bridges_Tarjan(V3, edges3);
    cout << "Test 3 Bridges: ";
    for (auto& b : result3) {
        cout << "[" << b[0] << "," << b[1] << "] ";
    }
    cout << endl;
}

int main() {
    Test_Bridges_Tarjan();
    return 0;
}
