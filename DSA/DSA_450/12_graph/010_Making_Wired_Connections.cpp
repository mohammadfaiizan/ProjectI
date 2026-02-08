/*
Problem: Number of Operations to Make Network Connected
URL: https://leetcode.com/problems/number-of-operations-to-make-network-connected/

Problem Statement:
Given n computers and connections, find minimum connections to move to connect all. If not enough cables, return -1.

Sample Input/Output:
Input: n=4, connections=[[0,1],[0,2],[1,2]]
Output: 1
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Wired_Connections_DFS_Helper(int node, vector<vector<int>>& adj, vector<bool>& visited) {
        visited[node] = true;
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                Wired_Connections_DFS_Helper(neighbor, adj, visited);
            }
        }
    }

    int Wired_Connections_DFS(int n, vector<vector<int>>& connections) {
        /*
        Count Components - Need components-1 extra cables
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        if (connections.size() < n - 1) {
            return -1;
        }
        
        vector<vector<int>> adj(n);
        for (auto& conn : connections) {
            adj[conn[0]].push_back(conn[1]);
            adj[conn[1]].push_back(conn[0]);
        }
        
        vector<bool> visited(n, false);
        int components = 0;
        
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                components++;
                Wired_Connections_DFS_Helper(i, adj, visited);
            }
        }
        
        return components - 1;
    }

    int Find_Parent(vector<int>& parent, int x) {
        if (parent[x] != x) {
            parent[x] = Find_Parent(parent, parent[x]);
        }
        return parent[x];
    }

    void Union_Set(vector<int>& parent, vector<int>& rank, int x, int y) {
        int px = Find_Parent(parent, x);
        int py = Find_Parent(parent, y);
        
        if (px == py) return;
        
        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
    }

    int Wired_Connections_Union_Find(int n, vector<vector<int>>& connections) {
        /*
        DSU-based
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        if (connections.size() < n - 1) {
            return -1;
        }
        
        vector<int> parent(n);
        vector<int> rank(n, 0);
        
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
        
        for (auto& conn : connections) {
            Union_Set(parent, rank, conn[0], conn[1]);
        }
        
        int components = 0;
        for (int i = 0; i < n; i++) {
            if (parent[i] == i) {
                components++;
            }
        }
        
        return components - 1;
    }
};

void Test_Making_Wired_Connections() {
    Solution solution;
    
    cout << "Test 1: n=4, connections=[[0,1],[0,2],[1,2]]" << endl;
    int n1 = 4;
    vector<vector<int>> connections1 = {{0,1}, {0,2}, {1,2}};
    int result1 = solution.Wired_Connections_DFS(n1, connections1);
    cout << "Minimum operations (DFS): " << result1 << endl;
    
    int result1_uf = solution.Wired_Connections_Union_Find(n1, connections1);
    cout << "Minimum operations (Union-Find): " << result1_uf << endl;
    
    cout << "\nTest 2: n=6, connections=[[0,1],[0,2],[0,3],[1,2]]" << endl;
    int n2 = 6;
    vector<vector<int>> connections2 = {{0,1}, {0,2}, {0,3}, {1,2}};
    int result2 = solution.Wired_Connections_DFS(n2, connections2);
    cout << "Minimum operations (DFS): " << result2 << endl;
    
    int result2_uf = solution.Wired_Connections_Union_Find(n2, connections2);
    cout << "Minimum operations (Union-Find): " << result2_uf << endl;
    
    cout << "\nTest 3: Not enough cables" << endl;
    int n3 = 5;
    vector<vector<int>> connections3 = {{0,1}, {0,2}};
    int result3 = solution.Wired_Connections_DFS(n3, connections3);
    cout << "Minimum operations: " << result3 << endl;
}

int main() {
    Test_Making_Wired_Connections();
    return 0;
}
