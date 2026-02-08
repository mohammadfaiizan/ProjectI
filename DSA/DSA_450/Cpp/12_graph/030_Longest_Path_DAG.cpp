/*
Problem: Longest Path in a Directed Acyclic Graph
URL: https://www.geeksforgeeks.org/find-longest-path-directed-acyclic-graph/

Problem Statement:
Find longest path from a given source vertex in a Directed Acyclic Graph (DAG). The graph has weighted edges.

Sample Input/Output:
Input: V=6, edges = [[0,1,5],[0,2,3],[1,3,6],[1,2,2],[2,4,4],[2,5,2],[2,3,7],[3,5,1],[3,4,-1],[4,5,-2]], src=1
Output: [0, 5, 3, 11, 7, 9]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Longest_Path_Topological(int V, vector<vector<int>>& edges, int src) {
        /*
        Topological sort + DP relaxation with negated weights
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        vector<vector<pair<int, int>>> adj(V);
        for (auto& e : edges) {
            adj[e[0]].push_back({e[1], e[2]});
        }
        
        vector<int> indegree(V, 0);
        for (auto& e : edges) {
            indegree[e[1]]++;
        }
        
        queue<int> q;
        for (int i = 0; i < V; i++) {
            if (indegree[i] == 0) {
                q.push(i);
            }
        }
        
        vector<int> topo;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            topo.push_back(u);
            
            for (auto& [v, w] : adj[u]) {
                indegree[v]--;
                if (indegree[v] == 0) {
                    q.push(v);
                }
            }
        }
        
        vector<int> dist(V, INT_MIN);
        dist[src] = 0;
        
        for (int u : topo) {
            if (dist[u] != INT_MIN) {
                for (auto& [v, w] : adj[u]) {
                    if (dist[u] + w > dist[v]) {
                        dist[v] = dist[u] + w;
                    }
                }
            }
        }
        
        return dist;
    }
};

void Test_Longest_Path_Topological() {
    Solution solution;
    
    int V1 = 6;
    vector<vector<int>> edges1 = {{0,1,5},{0,2,3},{1,3,6},{1,2,2},{2,4,4},{2,5,2},{2,3,7},{3,5,1},{3,4,-1},{4,5,-2}};
    vector<int> result1 = solution.Longest_Path_Topological(V1, edges1, 1);
    cout << "Test 1 Longest paths from src=1: ";
    for (int d : result1) {
        if (d == INT_MIN) cout << "-INF ";
        else cout << d << " ";
    }
    cout << endl;
    
    int V2 = 4;
    vector<vector<int>> edges2 = {{0,1,1},{0,2,4},{1,2,2},{1,3,5},{2,3,1}};
    vector<int> result2 = solution.Longest_Path_Topological(V2, edges2, 0);
    cout << "Test 2 Longest paths from src=0: ";
    for (int d : result2) {
        if (d == INT_MIN) cout << "-INF ";
        else cout << d << " ";
    }
    cout << endl;
    
    int V3 = 3;
    vector<vector<int>> edges3 = {{0,1,10},{1,2,20}};
    vector<int> result3 = solution.Longest_Path_Topological(V3, edges3, 0);
    cout << "Test 3 Longest paths from src=0: ";
    for (int d : result3) {
        if (d == INT_MIN) cout << "-INF ";
        else cout << d << " ";
    }
    cout << endl;
}

int main() {
    Test_Longest_Path_Topological();
    return 0;
}
