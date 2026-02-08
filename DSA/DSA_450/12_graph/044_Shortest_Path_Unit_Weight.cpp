/*
Problem: Shortest Path in Undirected Graph with Unit Weights
URL: https://www.geeksforgeeks.org/shortest-path-unweighted-graph/

Problem Statement:
Find shortest distance from source to all vertices in an unweighted graph.

Sample Input/Output:
Input: Unweighted graph with source vertex
Output: Shortest distances from source to all vertices
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Shortest_Path_BFS(int V, vector<pair<int, int>>& edges, int src) {
        /*
        Simple BFS
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        vector<vector<int>> adj(V);
        for (auto& edge : edges) {
            int u = edge.first;
            int v = edge.second;
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        
        vector<int> dist(V, -1);
        queue<int> q;
        
        dist[src] = 0;
        q.push(src);
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            
            for (int v : adj[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }
        
        return dist;
    }
};

void Test_Shortest_Path_Unit_Weight() {
    Solution solution;
    
    cout << "Test Case 1:" << endl;
    int V1 = 6;
    vector<pair<int, int>> edges1 = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}};
    int src1 = 0;
    vector<int> dist1 = solution.Shortest_Path_BFS(V1, edges1, src1);
    cout << "Source: " << src1 << endl;
    cout << "Distances: ";
    for (int i = 0; i < V1; i++) {
        cout << "[" << i << ":" << dist1[i] << "] ";
    }
    cout << endl;
    cout << endl;
    
    cout << "Test Case 2:" << endl;
    int V2 = 5;
    vector<pair<int, int>> edges2 = {{0, 1}, {0, 4}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {3, 4}};
    int src2 = 0;
    vector<int> dist2 = solution.Shortest_Path_BFS(V2, edges2, src2);
    cout << "Source: " << src2 << endl;
    cout << "Distances: ";
    for (int i = 0; i < V2; i++) {
        cout << "[" << i << ":" << dist2[i] << "] ";
    }
    cout << endl;
    cout << endl;
    
    cout << "Test Case 3:" << endl;
    int V3 = 4;
    vector<pair<int, int>> edges3 = {{0, 1}, {1, 2}, {2, 3}};
    int src3 = 0;
    vector<int> dist3 = solution.Shortest_Path_BFS(V3, edges3, src3);
    cout << "Source: " << src3 << endl;
    cout << "Distances: ";
    for (int i = 0; i < V3; i++) {
        cout << "[" << i << ":" << dist3[i] << "] ";
    }
    cout << endl;
}

int main() {
    Test_Shortest_Path_Unit_Weight();
    return 0;
}
