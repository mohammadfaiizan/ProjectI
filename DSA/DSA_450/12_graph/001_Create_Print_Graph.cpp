/*
Problem: Create and Print a Graph
URL: https://www.geeksforgeeks.org/graph-and-its-representations/

Problem Statement:
Create a graph using adjacency matrix and adjacency list representations and print them.

Sample Input/Output:
Input: 5 vertices, edges: (0,1), (0,4), (1,2), (1,3), (1,4), (2,3), (3,4)
Output: Adjacency Matrix and Adjacency List representations
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> Create_Graph_Adjacency_Matrix(int V, vector<pair<int, int>>& edges) {
        /*
        Adjacency Matrix Representation
        Time Complexity: O(V^2)
        Space Complexity: O(V^2)
        */
        vector<vector<int>> adjMatrix(V, vector<int>(V, 0));
        for (auto& edge : edges) {
            int u = edge.first;
            int v = edge.second;
            adjMatrix[u][v] = 1;
            adjMatrix[v][u] = 1;
        }
        return adjMatrix;
    }

    vector<vector<int>> Create_Graph_Adjacency_List(int V, vector<pair<int, int>>& edges) {
        /*
        Adjacency List Representation
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        vector<vector<int>> adjList(V);
        for (auto& edge : edges) {
            int u = edge.first;
            int v = edge.second;
            adjList[u].push_back(v);
            adjList[v].push_back(u);
        }
        return adjList;
    }
};

void Test_Create_Print_Graph() {
    Solution solution;
    int V = 5;
    vector<pair<int, int>> edges = {{0,1}, {0,4}, {1,2}, {1,3}, {1,4}, {2,3}, {3,4}};
    
    cout << "Adjacency Matrix:" << endl;
    vector<vector<int>> adjMatrix = solution.Create_Graph_Adjacency_Matrix(V, edges);
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            cout << adjMatrix[i][j] << " ";
        }
        cout << endl;
    }
    
    cout << "\nAdjacency List:" << endl;
    vector<vector<int>> adjList = solution.Create_Graph_Adjacency_List(V, edges);
    for (int i = 0; i < V; i++) {
        cout << i << ": ";
        for (int neighbor : adjList[i]) {
            cout << neighbor << " ";
        }
        cout << endl;
    }
}

int main() {
    Test_Create_Print_Graph();
    return 0;
}
