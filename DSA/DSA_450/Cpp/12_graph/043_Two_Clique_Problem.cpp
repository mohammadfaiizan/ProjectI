/*
Problem: Two Clique Problem
URL: https://www.geeksforgeeks.org/two-clique-problem-check-graph-can-divided-two-cliques/

Problem Statement:
Check if vertices of a graph can be divided into two cliques.

Sample Input/Output:
Input: Graph with edges
Output: True if can be divided, False otherwise
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Is_Bipartite(int V, vector<vector<int>>& adj) {
        vector<int> color(V, -1);
        queue<int> q;
        
        for (int start = 0; start < V; start++) {
            if (color[start] != -1) continue;
            
            color[start] = 0;
            q.push(start);
            
            while (!q.empty()) {
                int u = q.front();
                q.pop();
                
                for (int v : adj[u]) {
                    if (color[v] == -1) {
                        color[v] = 1 - color[u];
                        q.push(v);
                    } else if (color[v] == color[u]) {
                        return false;
                    }
                }
            }
        }
        
        return true;
    }

    bool Two_Clique_Complement_Bipartite(int V, vector<pair<int, int>>& edges) {
        /*
        Complement + BFS bipartite check
        Time Complexity: O(V^2)
        Space Complexity: O(V^2)
        */
        vector<vector<bool>> original(V, vector<bool>(V, false));
        for (auto& edge : edges) {
            int u = edge.first;
            int v = edge.second;
            original[u][v] = true;
            original[v][u] = true;
        }
        
        vector<vector<int>> complement(V);
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (i != j && !original[i][j]) {
                    complement[i].push_back(j);
                }
            }
        }
        
        return Is_Bipartite(V, complement);
    }
};

void Test_Two_Clique() {
    Solution solution;
    
    cout << "Test Case 1: Can be divided into two cliques" << endl;
    int V1 = 4;
    vector<pair<int, int>> edges1 = {{0, 1}, {0, 2}, {1, 2}, {3, 0}, {3, 1}, {3, 2}};
    bool result1 = solution.Two_Clique_Complement_Bipartite(V1, edges1);
    cout << "Result: " << (result1 ? "True" : "False") << endl;
    cout << endl;
    
    cout << "Test Case 2: Cannot be divided" << endl;
    int V2 = 5;
    vector<pair<int, int>> edges2 = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 2}};
    bool result2 = solution.Two_Clique_Complement_Bipartite(V2, edges2);
    cout << "Result: " << (result2 ? "True" : "False") << endl;
    cout << endl;
    
    cout << "Test Case 3: Complete bipartite graph" << endl;
    int V3 = 4;
    vector<pair<int, int>> edges3 = {{0, 2}, {0, 3}, {1, 2}, {1, 3}};
    bool result3 = solution.Two_Clique_Complement_Bipartite(V3, edges3);
    cout << "Result: " << (result3 ? "True" : "False") << endl;
    cout << endl;
    
    cout << "Test Case 4: Two separate cliques" << endl;
    int V4 = 6;
    vector<pair<int, int>> edges4 = {{0, 1}, {0, 2}, {1, 2}, {3, 4}, {3, 5}, {4, 5}};
    bool result4 = solution.Two_Clique_Complement_Bipartite(V4, edges4);
    cout << "Result: " << (result4 ? "True" : "False") << endl;
}

int main() {
    Test_Two_Clique();
    return 0;
}
