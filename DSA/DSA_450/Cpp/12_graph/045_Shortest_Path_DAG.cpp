/*
Problem: Shortest Path in a DAG with Weights
URL: https://www.geeksforgeeks.org/shortest-path-for-directed-acyclic-graph/

Problem Statement:
Find shortest paths from source in a weighted DAG.

Sample Input/Output:
Input: Weighted DAG with source vertex
Output: Shortest distances from source to all vertices
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Topological_Sort(int u, vector<vector<pair<int, int>>>& adj, vector<bool>& visited, stack<int>& st) {
        visited[u] = true;
        for (auto& neighbor : adj[u]) {
            int v = neighbor.first;
            if (!visited[v]) {
                Topological_Sort(v, adj, visited, st);
            }
        }
        st.push(u);
    }

    vector<int> Shortest_Path_DAG_Topological(int V, vector<pair<pair<int, int>, int>>& weightedEdges, int src) {
        /*
        Topological sort + relax in order
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        vector<vector<pair<int, int>>> adj(V);
        for (auto& edge : weightedEdges) {
            int u = edge.first.first;
            int v = edge.first.second;
            int w = edge.second;
            adj[u].push_back({v, w});
        }
        
        vector<bool> visited(V, false);
        stack<int> st;
        
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                Topological_Sort(i, adj, visited, st);
            }
        }
        
        vector<int> dist(V, INT_MAX);
        dist[src] = 0;
        
        while (!st.empty()) {
            int u = st.top();
            st.pop();
            
            if (dist[u] != INT_MAX) {
                for (auto& neighbor : adj[u]) {
                    int v = neighbor.first;
                    int w = neighbor.second;
                    if (dist[v] > dist[u] + w) {
                        dist[v] = dist[u] + w;
                    }
                }
            }
        }
        
        return dist;
    }
};

void Test_Shortest_Path_DAG() {
    Solution solution;
    
    cout << "Test Case 1:" << endl;
    int V1 = 6;
    vector<pair<pair<int, int>, int>> edges1 = {
        {{0, 1}, 5}, {{0, 2}, 3}, {{1, 3}, 6}, {{1, 2}, 2},
        {{2, 4}, 4}, {{2, 5}, 2}, {{2, 3}, 7}, {{3, 4}, -1},
        {{4, 5}, -2}
    };
    int src1 = 1;
    vector<int> dist1 = solution.Shortest_Path_DAG_Topological(V1, edges1, src1);
    cout << "Source: " << src1 << endl;
    cout << "Distances: ";
    for (int i = 0; i < V1; i++) {
        if (dist1[i] == INT_MAX) {
            cout << "[" << i << ":INF] ";
        } else {
            cout << "[" << i << ":" << dist1[i] << "] ";
        }
    }
    cout << endl;
    cout << endl;
    
    cout << "Test Case 2:" << endl;
    int V2 = 4;
    vector<pair<pair<int, int>, int>> edges2 = {
        {{0, 1}, 1}, {{0, 2}, 4}, {{1, 2}, 2}, {{1, 3}, 5}, {{2, 3}, 1}
    };
    int src2 = 0;
    vector<int> dist2 = solution.Shortest_Path_DAG_Topological(V2, edges2, src2);
    cout << "Source: " << src2 << endl;
    cout << "Distances: ";
    for (int i = 0; i < V2; i++) {
        if (dist2[i] == INT_MAX) {
            cout << "[" << i << ":INF] ";
        } else {
            cout << "[" << i << ":" << dist2[i] << "] ";
        }
    }
    cout << endl;
    cout << endl;
    
    cout << "Test Case 3:" << endl;
    int V3 = 5;
    vector<pair<pair<int, int>, int>> edges3 = {
        {{0, 1}, 2}, {{0, 2}, 3}, {{1, 3}, 1}, {{2, 3}, 4}, {{3, 4}, 2}
    };
    int src3 = 0;
    vector<int> dist3 = solution.Shortest_Path_DAG_Topological(V3, edges3, src3);
    cout << "Source: " << src3 << endl;
    cout << "Distances: ";
    for (int i = 0; i < V3; i++) {
        if (dist3[i] == INT_MAX) {
            cout << "[" << i << ":INF] ";
        } else {
            cout << "[" << i << ":" << dist3[i] << "] ";
        }
    }
    cout << endl;
}

int main() {
    Test_Shortest_Path_DAG();
    return 0;
}
