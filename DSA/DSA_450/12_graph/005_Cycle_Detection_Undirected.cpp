/*
Problem: Detect Cycle in an Undirected Graph
URL: https://practice.geeksforgeeks.org/problems/detect-cycle-in-an-undirected-graph/1

Problem Statement:
Detect if an undirected graph contains a cycle.

Sample Input/Output:
Input: Graph with cycle: 0-1-2-0
Output: Cycle detected: true
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Cycle_Undirected_DFS_Helper(int node, int parent, vector<vector<int>>& adj, vector<bool>& visited) {
        visited[node] = true;
        
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                if (Cycle_Undirected_DFS_Helper(neighbor, node, adj, visited)) {
                    return true;
                }
            } else if (neighbor != parent) {
                return true;
            }
        }
        
        return false;
    }

    bool Cycle_Undirected_DFS(int V, vector<vector<int>>& adj) {
        /*
        DFS with Parent Tracking
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<bool> visited(V, false);
        
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                if (Cycle_Undirected_DFS_Helper(i, -1, adj, visited)) {
                    return true;
                }
            }
        }
        
        return false;
    }

    bool Cycle_Undirected_BFS(int V, vector<vector<int>>& adj) {
        /*
        BFS with Parent Tracking
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<bool> visited(V, false);
        
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                queue<pair<int, int>> q;
                visited[i] = true;
                q.push({i, -1});
                
                while (!q.empty()) {
                    int node = q.front().first;
                    int parent = q.front().second;
                    q.pop();
                    
                    for (int neighbor : adj[node]) {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push({neighbor, node});
                        } else if (neighbor != parent) {
                            return true;
                        }
                    }
                }
            }
        }
        
        return false;
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

    bool Cycle_Undirected_Union_Find(int V, vector<vector<int>>& adj) {
        /*
        Union-Find / DSU
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<int> parent(V);
        vector<int> rank(V, 0);
        
        for (int i = 0; i < V; i++) {
            parent[i] = i;
        }
        
        for (int u = 0; u < V; u++) {
            for (int v : adj[u]) {
                if (u < v) {
                    int pu = Find_Parent(parent, u);
                    int pv = Find_Parent(parent, v);
                    
                    if (pu == pv) {
                        return true;
                    }
                    
                    Union_Set(parent, rank, u, v);
                }
            }
        }
        
        return false;
    }
};

void Test_Cycle_Detection_Undirected() {
    Solution solution;
    
    cout << "Test 1: Graph with Cycle" << endl;
    int V1 = 4;
    vector<vector<int>> adj1(V1);
    adj1[0] = {1, 2};
    adj1[1] = {0, 2};
    adj1[2] = {0, 1, 3};
    adj1[3] = {2};
    
    bool hasCycle1 = solution.Cycle_Undirected_DFS(V1, adj1);
    cout << "Cycle detected (DFS): " << (hasCycle1 ? "Yes" : "No") << endl;
    
    bool hasCycle1_bfs = solution.Cycle_Undirected_BFS(V1, adj1);
    cout << "Cycle detected (BFS): " << (hasCycle1_bfs ? "Yes" : "No") << endl;
    
    bool hasCycle1_uf = solution.Cycle_Undirected_Union_Find(V1, adj1);
    cout << "Cycle detected (Union-Find): " << (hasCycle1_uf ? "Yes" : "No") << endl;
    
    cout << "\nTest 2: Graph without Cycle" << endl;
    int V2 = 4;
    vector<vector<int>> adj2(V2);
    adj2[0] = {1};
    adj2[1] = {0, 2};
    adj2[2] = {1, 3};
    adj2[3] = {2};
    
    bool hasCycle2 = solution.Cycle_Undirected_DFS(V2, adj2);
    cout << "Cycle detected (DFS): " << (hasCycle2 ? "Yes" : "No") << endl;
    
    bool hasCycle2_bfs = solution.Cycle_Undirected_BFS(V2, adj2);
    cout << "Cycle detected (BFS): " << (hasCycle2_bfs ? "Yes" : "No") << endl;
    
    bool hasCycle2_uf = solution.Cycle_Undirected_Union_Find(V2, adj2);
    cout << "Cycle detected (Union-Find): " << (hasCycle2_uf ? "Yes" : "No") << endl;
}

int main() {
    Test_Cycle_Detection_Undirected();
    return 0;
}
