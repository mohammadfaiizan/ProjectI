/*
Problem: Chinese Postman Problem (Route Inspection)
URL: https://www.geeksforgeeks.org/chinese-postman-route-inspection-algorithm/

Problem Statement:
Find the shortest closed path that visits every edge at least once in a weighted undirected graph.

Sample Input/Output:
Input: Weighted undirected graph
Output: Minimum cost to traverse all edges
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Is_Eulerian(int V, vector<vector<pair<int, int>>>& adj) {
        for (int i = 0; i < V; i++) {
            if (adj[i].size() % 2 != 0) {
                return false;
            }
        }
        return true;
    }

    vector<int> Get_Odd_Degree_Vertices(int V, vector<vector<pair<int, int>>>& adj) {
        vector<int> oddVertices;
        for (int i = 0; i < V; i++) {
            if (adj[i].size() % 2 != 0) {
                oddVertices.push_back(i);
            }
        }
        return oddVertices;
    }

    vector<vector<int>> Floyd_Warshall(int V, vector<vector<pair<int, int>>>& adj) {
        vector<vector<int>> dist(V, vector<int>(V, INT_MAX));
        
        for (int i = 0; i < V; i++) {
            dist[i][i] = 0;
            for (auto& neighbor : adj[i]) {
                int v = neighbor.first;
                int w = neighbor.second;
                dist[i][v] = min(dist[i][v], w);
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
        
        return dist;
    }

    int Min_Weight_Perfect_Matching(vector<int>& oddVertices, vector<vector<int>>& dist) {
        int n = oddVertices.size();
        if (n == 0) return 0;
        
        vector<int> dp(1 << n, INT_MAX);
        dp[0] = 0;
        
        for (int mask = 0; mask < (1 << n); mask++) {
            if (dp[mask] == INT_MAX) continue;
            
            int first = -1;
            for (int i = 0; i < n; i++) {
                if (!(mask & (1 << i))) {
                    first = i;
                    break;
                }
            }
            
            if (first == -1) continue;
            
            for (int j = first + 1; j < n; j++) {
                if (mask & (1 << j)) continue;
                
                int newMask = mask | (1 << first) | (1 << j);
                int u = oddVertices[first];
                int v = oddVertices[j];
                int cost = dist[u][v];
                
                if (cost != INT_MAX) {
                    dp[newMask] = min(dp[newMask], dp[mask] + cost);
                }
            }
        }
        
        return dp[(1 << n) - 1];
    }

    int Chinese_Postman_Solve(int V, vector<pair<pair<int, int>, int>>& weightedEdges) {
        /*
        Check Eulerian, if not find odd-degree vertices and add shortest paths between pairs
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        */
        vector<vector<pair<int, int>>> adj(V);
        int totalWeight = 0;
        
        for (auto& edge : weightedEdges) {
            int u = edge.first.first;
            int v = edge.first.second;
            int w = edge.second;
            adj[u].push_back({v, w});
            adj[v].push_back({u, w});
            totalWeight += w;
        }
        
        if (Is_Eulerian(V, adj)) {
            return totalWeight;
        }
        
        vector<int> oddVertices = Get_Odd_Degree_Vertices(V, adj);
        vector<vector<int>> dist = Floyd_Warshall(V, adj);
        
        int matchingCost = Min_Weight_Perfect_Matching(oddVertices, dist);
        
        return totalWeight + matchingCost;
    }
};

void Test_Chinese_Postman() {
    Solution solution;
    
    cout << "Test Case 1: Eulerian Graph" << endl;
    int V1 = 4;
    vector<pair<pair<int, int>, int>> edges1 = {
        {{0, 1}, 1}, {{1, 2}, 2}, {{2, 3}, 3}, {{3, 0}, 4}, {{0, 2}, 5}, {{1, 3}, 6}
    };
    int result1 = solution.Chinese_Postman_Solve(V1, edges1);
    cout << "Minimum Cost: " << result1 << endl;
    cout << endl;
    
    cout << "Test Case 2: Non-Eulerian Graph" << endl;
    int V2 = 4;
    vector<pair<pair<int, int>, int>> edges2 = {
        {{0, 1}, 1}, {{1, 2}, 2}, {{2, 3}, 3}, {{3, 0}, 4}
    };
    int result2 = solution.Chinese_Postman_Solve(V2, edges2);
    cout << "Minimum Cost: " << result2 << endl;
    cout << endl;
    
    cout << "Test Case 3: Complex Graph" << endl;
    int V3 = 5;
    vector<pair<pair<int, int>, int>> edges3 = {
        {{0, 1}, 2}, {{1, 2}, 3}, {{2, 3}, 1}, {{3, 4}, 4}, {{4, 0}, 5}
    };
    int result3 = solution.Chinese_Postman_Solve(V3, edges3);
    cout << "Minimum Cost: " << result3 << endl;
}

int main() {
    Test_Chinese_Postman();
    return 0;
}
