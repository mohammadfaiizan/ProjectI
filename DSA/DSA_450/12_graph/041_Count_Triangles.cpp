/*
Problem: Number of Triangles in a Graph
URL: https://www.geeksforgeeks.org/number-of-triangles-in-directed-and-undirected-graphs/

Problem Statement:
Count the number of triangles in directed and undirected graphs.

Sample Input/Output:
Input: Graph with edges
Output: Number of triangles
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Count_Triangles_Brute(int V, vector<pair<int, int>>& edges, bool directed) {
        /*
        Check all triplets (i,j,k)
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        */
        vector<vector<bool>> adj(V, vector<bool>(V, false));
        
        for (auto& edge : edges) {
            int u = edge.first;
            int v = edge.second;
            adj[u][v] = true;
            if (!directed) {
                adj[v][u] = true;
            }
        }
        
        int count = 0;
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (i == j || !adj[i][j]) continue;
                for (int k = 0; k < V; k++) {
                    if (i == k || j == k) continue;
                    if (adj[j][k] && adj[k][i]) {
                        count++;
                    }
                }
            }
        }
        
        if (directed) {
            return count / 3;
        } else {
            return count / 6;
        }
    }

    int Count_Triangles_Matrix(int V, vector<pair<int, int>>& edges, bool directed) {
        /*
        Matrix multiplication trace method
        Time Complexity: O(V^3)
        Space Complexity: O(V^2)
        */
        vector<vector<int>> adj(V, vector<int>(V, 0));
        
        for (auto& edge : edges) {
            int u = edge.first;
            int v = edge.second;
            adj[u][v] = 1;
            if (!directed) {
                adj[v][u] = 1;
            }
        }
        
        vector<vector<int>> adj2(V, vector<int>(V, 0));
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                for (int k = 0; k < V; k++) {
                    adj2[i][j] += adj[i][k] * adj[k][j];
                }
            }
        }
        
        vector<vector<int>> adj3(V, vector<int>(V, 0));
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                for (int k = 0; k < V; k++) {
                    adj3[i][j] += adj2[i][k] * adj[k][j];
                }
            }
        }
        
        int trace = 0;
        for (int i = 0; i < V; i++) {
            trace += adj3[i][i];
        }
        
        if (directed) {
            return trace / 3;
        } else {
            return trace / 6;
        }
    }
};

void Test_Count_Triangles() {
    Solution solution;
    
    cout << "Test Case 1: Undirected Graph" << endl;
    int V1 = 4;
    vector<pair<int, int>> edges1 = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};
    int result1_brute = solution.Count_Triangles_Brute(V1, edges1, false);
    int result1_matrix = solution.Count_Triangles_Matrix(V1, edges1, false);
    cout << "Brute Force: " << result1_brute << " triangles" << endl;
    cout << "Matrix Method: " << result1_matrix << " triangles" << endl;
    cout << endl;
    
    cout << "Test Case 2: Directed Graph" << endl;
    int V2 = 4;
    vector<pair<int, int>> edges2 = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {3, 1}};
    int result2_brute = solution.Count_Triangles_Brute(V2, edges2, true);
    int result2_matrix = solution.Count_Triangles_Matrix(V2, edges2, true);
    cout << "Brute Force: " << result2_brute << " triangles" << endl;
    cout << "Matrix Method: " << result2_matrix << " triangles" << endl;
    cout << endl;
    
    cout << "Test Case 3: Complete Graph K4" << endl;
    int V3 = 4;
    vector<pair<int, int>> edges3 = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    int result3_brute = solution.Count_Triangles_Brute(V3, edges3, false);
    int result3_matrix = solution.Count_Triangles_Matrix(V3, edges3, false);
    cout << "Brute Force: " << result3_brute << " triangles" << endl;
    cout << "Matrix Method: " << result3_matrix << " triangles" << endl;
}

int main() {
    Test_Count_Triangles();
    return 0;
}
