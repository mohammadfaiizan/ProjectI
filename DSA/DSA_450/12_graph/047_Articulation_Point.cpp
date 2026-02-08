/*
Problem: Articulation Points (Cut Vertices)
URL: https://www.geeksforgeeks.org/articulation-points-or-cut-vertices-in-a-graph/

Problem Statement:
Find all articulation points (vertices whose removal disconnects the graph).

Sample Input/Output:
Input: Graph with edges
Output: List of articulation points
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void DFS_Articulation(int u, int parent, vector<vector<int>>& adj, vector<int>& disc, vector<int>& low, vector<bool>& visited, vector<bool>& isArticulation, int& time) {
        visited[u] = true;
        disc[u] = low[u] = ++time;
        int children = 0;
        
        for (int v : adj[u]) {
            if (!visited[v]) {
                children++;
                DFS_Articulation(v, u, adj, disc, low, visited, isArticulation, time);
                
                low[u] = min(low[u], low[v]);
                
                if (parent == -1 && children > 1) {
                    isArticulation[u] = true;
                }
                
                if (parent != -1 && low[v] >= disc[u]) {
                    isArticulation[u] = true;
                }
            } else if (v != parent) {
                low[u] = min(low[u], disc[v]);
            }
        }
    }

    vector<int> Articulation_Points_Tarjan(int V, vector<pair<int, int>>& edges) {
        /*
        DFS with disc[] and low[]
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
        
        vector<int> disc(V, -1);
        vector<int> low(V, -1);
        vector<bool> visited(V, false);
        vector<bool> isArticulation(V, false);
        int time = 0;
        
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                DFS_Articulation(i, -1, adj, disc, low, visited, isArticulation, time);
            }
        }
        
        vector<int> articulationPoints;
        for (int i = 0; i < V; i++) {
            if (isArticulation[i]) {
                articulationPoints.push_back(i);
            }
        }
        
        return articulationPoints;
    }
};

void Test_Articulation_Points() {
    Solution solution;
    
    cout << "Test Case 1:" << endl;
    int V1 = 5;
    vector<pair<int, int>> edges1 = {{0, 1}, {1, 2}, {2, 0}, {1, 3}, {3, 4}};
    vector<int> result1 = solution.Articulation_Points_Tarjan(V1, edges1);
    cout << "Articulation Points: ";
    for (int v : result1) {
        cout << v << " ";
    }
    cout << endl;
    cout << endl;
    
    cout << "Test Case 2:" << endl;
    int V2 = 7;
    vector<pair<int, int>> edges2 = {{0, 1}, {1, 2}, {2, 0}, {1, 3}, {1, 4}, {1, 6}, {3, 5}, {4, 5}};
    vector<int> result2 = solution.Articulation_Points_Tarjan(V2, edges2);
    cout << "Articulation Points: ";
    for (int v : result2) {
        cout << v << " ";
    }
    cout << endl;
    cout << endl;
    
    cout << "Test Case 3:" << endl;
    int V3 = 4;
    vector<pair<int, int>> edges3 = {{0, 1}, {1, 2}, {2, 3}};
    vector<int> result3 = solution.Articulation_Points_Tarjan(V3, edges3);
    cout << "Articulation Points: ";
    for (int v : result3) {
        cout << v << " ";
    }
    cout << endl;
    cout << endl;
    
    cout << "Test Case 4: No articulation points" << endl;
    int V4 = 4;
    vector<pair<int, int>> edges4 = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
    vector<int> result4 = solution.Articulation_Points_Tarjan(V4, edges4);
    cout << "Articulation Points: ";
    if (result4.empty()) {
        cout << "None";
    } else {
        for (int v : result4) {
            cout << v << " ";
        }
    }
    cout << endl;
}

int main() {
    Test_Articulation_Points();
    return 0;
}
