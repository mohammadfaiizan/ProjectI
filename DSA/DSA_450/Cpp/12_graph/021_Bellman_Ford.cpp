/*
Problem: Bellman-Ford Algorithm
URL: https://practice.geeksforgeeks.org/problems/distance-from-the-source-bellman-ford-algorithm/1

Problem Statement:
Find single-source shortest paths from a source vertex to all other vertices in a weighted directed graph. The algorithm can handle negative weight edges and detect negative cycles.

Sample Input/Output:
Input: Graph with edges (0,1,5), (0,2,3), (1,2,2), (1,3,6), (2,3,7), source=0
Output: Distances: [0, 5, 3, 9]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Bellman_Ford_Standard(int V, vector<vector<int>>& edges, int src) {
        /*
        Relax all edges V-1 times
        Time Complexity: O(V*E)
        Space Complexity: O(V)
        */
        vector<int> dist(V, INT_MAX);
        dist[src] = 0;
        
        for (int i = 0; i < V - 1; i++) {
            for (auto& edge : edges) {
                int u = edge[0];
                int v = edge[1];
                int w = edge[2];
                
                if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                }
            }
        }
        
        for (auto& edge : edges) {
            int u = edge[0];
            int v = edge[1];
            int w = edge[2];
            
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                return vector<int>(1, -1);
            }
        }
        
        return dist;
    }
    
    bool Has_Negative_Cycle(int V, vector<vector<int>>& edges) {
        /*
        Check for negative cycle
        Time Complexity: O(V*E)
        Space Complexity: O(V)
        */
        vector<int> dist(V, 0);
        
        for (int i = 0; i < V - 1; i++) {
            for (auto& edge : edges) {
                int u = edge[0];
                int v = edge[1];
                int w = edge[2];
                
                if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                }
            }
        }
        
        for (auto& edge : edges) {
            int u = edge[0];
            int v = edge[1];
            int w = edge[2];
            
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                return true;
            }
        }
        
        return false;
    }
};

void Test_Bellman_Ford() {
    Solution solution;
    
    cout << "Test Case 1: Graph with negative edges (no cycle)" << endl;
    int V1 = 4;
    vector<vector<int>> edges1 = {
        {0, 1, 5},
        {0, 2, 3},
        {1, 2, -2},
        {1, 3, 6},
        {2, 3, 7}
    };
    vector<int> dist1 = solution.Bellman_Ford_Standard(V1, edges1, 0);
    if (dist1[0] == -1) {
        cout << "Negative cycle detected!" << endl;
    } else {
        cout << "Distances from source 0: ";
        for (int d : dist1) cout << d << " ";
        cout << endl;
    }
    
    cout << "\nTest Case 2: Graph with negative cycle" << endl;
    int V2 = 3;
    vector<vector<int>> edges2 = {
        {0, 1, 1},
        {1, 2, -3},
        {2, 0, 2}
    };
    bool hasCycle = solution.Has_Negative_Cycle(V2, edges2);
    cout << "Has negative cycle: " << (hasCycle ? "Yes" : "No") << endl;
    
    cout << "\nTest Case 3: Simple path graph" << endl;
    int V3 = 5;
    vector<vector<int>> edges3 = {
        {0, 1, 1},
        {1, 2, 2},
        {2, 3, 3},
        {3, 4, 4}
    };
    vector<int> dist3 = solution.Bellman_Ford_Standard(V3, edges3, 0);
    cout << "Distances from source 0: ";
    for (int d : dist3) cout << d << " ";
    cout << endl;
}

int main() {
    Test_Bellman_Ford();
    return 0;
}
