/*
Problem: Detect Negative Weight Cycle in a Graph
URL: https://www.geeksforgeeks.org/detect-negative-cycle-graph-bellman-ford/

Problem Statement:
Detect if a graph contains a negative weight cycle. A negative weight cycle is a cycle whose edges sum to a negative value.

Sample Input/Output:
Input: V=4, edges = [[0,1,1],[1,2,-1],[2,3,-1],[3,0,-1]]
Output: true (negative cycle exists)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Negative_Cycle_Bellman_Ford(int V, vector<vector<int>>& edges, int src) {
        /*
        Run Bellman-Ford, check if Vth relaxation reduces any distance
        Time Complexity: O(V*E)
        Space Complexity: O(V)
        */
        vector<int> dist(V, INT_MAX);
        dist[src] = 0;
        
        for (int i = 0; i < V - 1; i++) {
            for (auto& e : edges) {
                int u = e[0], v = e[1], w = e[2];
                if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                }
            }
        }
        
        for (auto& e : edges) {
            int u = e[0], v = e[1], w = e[2];
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                return true;
            }
        }
        
        return false;
    }
    
    bool Negative_Cycle_Floyd_Warshall(int V, vector<vector<int>>& graph) {
        /*
        Check diagonal of all-pairs matrix for negative
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        */
        vector<vector<int>> dist(V, vector<int>(V));
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                dist[i][j] = graph[i][j];
            }
        }
        
        for (int k = 0; k < V; k++) {
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }
        
        for (int i = 0; i < V; i++) {
            if (dist[i][i] < 0) {
                return true;
            }
        }
        
        return false;
    }
};

void Test_Negative_Cycle_Bellman_Ford() {
    Solution solution;
    
    int V1 = 4;
    vector<vector<int>> edges1 = {{0,1,1},{1,2,-1},{2,3,-1},{3,0,-1}};
    cout << "Test 1 Bellman-Ford: " << (solution.Negative_Cycle_Bellman_Ford(V1, edges1, 0) ? "true" : "false") << endl;
    
    int V2 = 3;
    vector<vector<int>> edges2 = {{0,1,1},{1,2,2},{2,0,3}};
    cout << "Test 2 Bellman-Ford: " << (solution.Negative_Cycle_Bellman_Ford(V2, edges2, 0) ? "true" : "false") << endl;
    
    int V3 = 4;
    vector<vector<int>> graph3(V3, vector<int>(V3, INT_MAX));
    graph3[0][1] = 1;
    graph3[1][2] = -1;
    graph3[2][3] = -1;
    graph3[3][0] = -1;
    for (int i = 0; i < V3; i++) graph3[i][i] = 0;
    cout << "Test 3 Floyd-Warshall: " << (solution.Negative_Cycle_Floyd_Warshall(V3, graph3) ? "true" : "false") << endl;
}

int main() {
    Test_Negative_Cycle_Bellman_Ford();
    return 0;
}
