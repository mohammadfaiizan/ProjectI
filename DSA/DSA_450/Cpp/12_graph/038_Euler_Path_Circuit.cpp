/*
Problem: Euler Path and Circuit (Seven Bridges of Konigsberg)
URL: https://www.geeksforgeeks.org/eulerian-path-and-circuit/

Problem Statement:
Determine if a graph has an Eulerian Circuit (all vertices have even degree), Eulerian Path (exactly 2 vertices have odd degree), or neither.

Sample Input/Output:
Input: Graph with edges
Output: Eulerian Circuit, Eulerian Path, or Neither
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void DFS(int u, vector<vector<int>>& adj, vector<bool>& visited) {
        visited[u] = true;
        for (int v : adj[u]) {
            if (!visited[v]) {
                DFS(v, adj, visited);
            }
        }
    }

    bool Is_Connected(int V, vector<vector<int>>& adj) {
        vector<bool> visited(V, false);
        int start = -1;
        for (int i = 0; i < V; i++) {
            if (adj[i].size() > 0) {
                start = i;
                break;
            }
        }
        if (start == -1) return true;
        
        DFS(start, adj, visited);
        
        for (int i = 0; i < V; i++) {
            if (adj[i].size() > 0 && !visited[i]) {
                return false;
            }
        }
        return true;
    }

    string Euler_Check(int V, vector<pair<int, int>>& edges) {
        /*
        Check connectivity + count odd-degree vertices
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        vector<vector<int>> adj(V);
        vector<int> degree(V, 0);
        
        for (auto& edge : edges) {
            int u = edge.first;
            int v = edge.second;
            adj[u].push_back(v);
            adj[v].push_back(u);
            degree[u]++;
            degree[v]++;
        }
        
        if (!Is_Connected(V, adj)) {
            return "Neither";
        }
        
        int oddCount = 0;
        for (int i = 0; i < V; i++) {
            if (degree[i] % 2 != 0) {
                oddCount++;
            }
        }
        
        if (oddCount == 0) {
            return "Eulerian Circuit";
        } else if (oddCount == 2) {
            return "Eulerian Path";
        } else {
            return "Neither";
        }
    }
};

void Test_Euler_Check() {
    Solution solution;
    
    cout << "Test Case 1: Eulerian Circuit" << endl;
    int V1 = 3;
    vector<pair<int, int>> edges1 = {{0, 1}, {1, 2}, {2, 0}};
    string result1 = solution.Euler_Check(V1, edges1);
    cout << "Result: " << result1 << endl;
    cout << endl;
    
    cout << "Test Case 2: Eulerian Path" << endl;
    int V2 = 4;
    vector<pair<int, int>> edges2 = {{0, 1}, {1, 2}, {2, 3}};
    string result2 = solution.Euler_Check(V2, edges2);
    cout << "Result: " << result2 << endl;
    cout << endl;
    
    cout << "Test Case 3: Neither" << endl;
    int V3 = 4;
    vector<pair<int, int>> edges3 = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}};
    string result3 = solution.Euler_Check(V3, edges3);
    cout << "Result: " << result3 << endl;
    cout << endl;
    
    cout << "Test Case 4: Eulerian Circuit (Complex)" << endl;
    int V4 = 5;
    vector<pair<int, int>> edges4 = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 2}, {1, 3}};
    string result4 = solution.Euler_Check(V4, edges4);
    cout << "Result: " << result4 << endl;
}

int main() {
    Test_Euler_Check();
    return 0;
}
