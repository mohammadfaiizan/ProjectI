/*
Problem: Check if Graph is Bipartite
URL: https://leetcode.com/problems/is-graph-bipartite/

Problem Statement:
Check if a graph can be 2-colored (bipartite). A graph is bipartite if we can split its set of nodes into two independent subsets A and B such that every edge in the graph has one node in A and another node in B.

Sample Input/Output:
Input: graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
Output: false
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Bipartite_BFS(vector<vector<int>>& graph) {
        /*
        BFS coloring
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        int n = graph.size();
        vector<int> color(n, -1);
        
        for (int i = 0; i < n; i++) {
            if (color[i] == -1) {
                queue<int> q;
                q.push(i);
                color[i] = 0;
                
                while (!q.empty()) {
                    int u = q.front();
                    q.pop();
                    
                    for (int v : graph[u]) {
                        if (color[v] == -1) {
                            color[v] = 1 - color[u];
                            q.push(v);
                        } else if (color[v] == color[u]) {
                            return false;
                        }
                    }
                }
            }
        }
        
        return true;
    }
    
    bool Bipartite_DFS(vector<vector<int>>& graph) {
        /*
        DFS coloring
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        int n = graph.size();
        vector<int> color(n, -1);
        
        function<bool(int, int)> dfs = [&](int u, int c) {
            color[u] = c;
            for (int v : graph[u]) {
                if (color[v] == -1) {
                    if (!dfs(v, 1 - c)) {
                        return false;
                    }
                } else if (color[v] == c) {
                    return false;
                }
            }
            return true;
        };
        
        for (int i = 0; i < n; i++) {
            if (color[i] == -1) {
                if (!dfs(i, 0)) {
                    return false;
                }
            }
        }
        
        return true;
    }
};

void Test_Bipartite_BFS() {
    Solution solution;
    
    vector<vector<int>> graph1 = {{1,2,3},{0,2},{0,1,3},{0,2}};
    cout << "Test 1 BFS: " << (solution.Bipartite_BFS(graph1) ? "true" : "false") << endl;
    
    vector<vector<int>> graph2 = {{1,3},{0,2},{1,3},{0,2}};
    cout << "Test 2 BFS: " << (solution.Bipartite_BFS(graph2) ? "true" : "false") << endl;
    
    vector<vector<int>> graph3 = {{1},{0,2},{1}};
    cout << "Test 3 DFS: " << (solution.Bipartite_DFS(graph3) ? "true" : "false") << endl;
    
    vector<vector<int>> graph4 = {{1,2},{0,2},{0,1}};
    cout << "Test 4 DFS: " << (solution.Bipartite_DFS(graph4) ? "true" : "false") << endl;
}

int main() {
    Test_Bipartite_BFS();
    return 0;
}
