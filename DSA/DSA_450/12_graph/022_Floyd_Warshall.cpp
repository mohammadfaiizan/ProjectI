/*
Problem: Floyd-Warshall Algorithm
URL: https://practice.geeksforgeeks.org/problems/implementing-floyd-warshall2042/1

Problem Statement:
Find all-pairs shortest paths in a weighted directed graph. The algorithm uses dynamic programming with intermediate vertices to compute shortest distances between all pairs of vertices.

Sample Input/Output:
Input: Graph with 4 vertices, adjacency matrix with weights
Output: Shortest distance matrix for all pairs
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> Floyd_Warshall_DP(int V, vector<vector<int>>& graph) {
        /*
        3 nested loops with intermediate vertex
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        */
        vector<vector<int>> dist = graph;
        
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][j] == -1) {
                    dist[i][j] = INT_MAX;
                }
                if (i == j) {
                    dist[i][j] = 0;
                }
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
            for (int j = 0; j < V; j++) {
                if (dist[i][j] == INT_MAX) {
                    dist[i][j] = -1;
                }
            }
        }
        
        return dist;
    }
};

void Test_Floyd_Warshall() {
    Solution solution;
    
    cout << "Test Case 1: Weighted graph with 4 vertices" << endl;
    int V1 = 4;
    vector<vector<int>> graph1 = {
        {0, 5, -1, 10},
        {-1, 0, 3, -1},
        {-1, -1, 0, 1},
        {-1, -1, -1, 0}
    };
    vector<vector<int>> result1 = solution.Floyd_Warshall_DP(V1, graph1);
    cout << "Shortest distance matrix:" << endl;
    for (int i = 0; i < V1; i++) {
        for (int j = 0; j < V1; j++) {
            cout << result1[i][j] << " ";
        }
        cout << endl;
    }
    
    cout << "\nTest Case 2: Complete graph" << endl;
    int V2 = 3;
    vector<vector<int>> graph2 = {
        {0, 1, 4},
        {1, 0, 2},
        {4, 2, 0}
    };
    vector<vector<int>> result2 = solution.Floyd_Warshall_DP(V2, graph2);
    cout << "Shortest distance matrix:" << endl;
    for (int i = 0; i < V2; i++) {
        for (int j = 0; j < V2; j++) {
            cout << result2[i][j] << " ";
        }
        cout << endl;
    }
    
    cout << "\nTest Case 3: Graph with no direct paths" << endl;
    int V3 = 4;
    vector<vector<int>> graph3 = {
        {0, -1, -1, -1},
        {-1, 0, 2, -1},
        {-1, -1, 0, 3},
        {-1, -1, -1, 0}
    };
    vector<vector<int>> result3 = solution.Floyd_Warshall_DP(V3, graph3);
    cout << "Shortest distance matrix:" << endl;
    for (int i = 0; i < V3; i++) {
        for (int j = 0; j < V3; j++) {
            cout << result3[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    Test_Floyd_Warshall();
    return 0;
}
