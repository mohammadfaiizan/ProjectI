/*
Problem: Vertex Cover Problem
URL: https://www.geeksforgeeks.org/vertex-cover-problem-set-1-introduction-approximate-algorithm-2/

Problem Statement:
Find an approximate minimum vertex cover (set of vertices that covers all edges).

Sample Input/Output:
Input: Graph with edges
Output: Vertex cover set
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Vertex_Cover_Approximate(int V, vector<pair<int, int>>& edges) {
        /*
        Greedy: pick edge, add both endpoints, remove covered edges
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        vector<vector<int>> adj(V);
        vector<bool> inCover(V, false);
        vector<bool> edgeCovered(edges.size(), false);
        
        for (int i = 0; i < edges.size(); i++) {
            int u = edges[i].first;
            int v = edges[i].second;
            adj[u].push_back(i);
            adj[v].push_back(i);
        }
        
        for (int i = 0; i < edges.size(); i++) {
            if (edgeCovered[i]) continue;
            
            int u = edges[i].first;
            int v = edges[i].second;
            
            inCover[u] = true;
            inCover[v] = true;
            
            for (int e : adj[u]) {
                edgeCovered[e] = true;
            }
            for (int e : adj[v]) {
                edgeCovered[e] = true;
            }
        }
        
        vector<int> cover;
        for (int i = 0; i < V; i++) {
            if (inCover[i]) {
                cover.push_back(i);
            }
        }
        return cover;
    }

    int Vertex_Cover_Tree_DP(int V, vector<pair<int, int>>& edges, int root) {
        /*
        DP on tree (for trees only)
        Time Complexity: O(V)
        Space Complexity: O(V)
        */
        vector<vector<int>> adj(V);
        for (auto& edge : edges) {
            int u = edge.first;
            int v = edge.second;
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        
        vector<vector<int>> dp(V, vector<int>(2, -1));
        
        function<int(int, int, bool)> dfs = [&](int u, int parent, bool include) -> int {
            if (dp[u][include] != -1) return dp[u][include];
            
            int result = include ? 1 : 0;
            
            for (int v : adj[u]) {
                if (v == parent) continue;
                
                if (include) {
                    result += min(dfs(v, u, true), dfs(v, u, false));
                } else {
                    result += dfs(v, u, true);
                }
            }
            
            return dp[u][include] = result;
        };
        
        return min(dfs(root, -1, true), dfs(root, -1, false));
    }
};

void Test_Vertex_Cover() {
    Solution solution;
    
    cout << "Test Case 1: General Graph" << endl;
    int V1 = 7;
    vector<pair<int, int>> edges1 = {{0, 1}, {0, 2}, {1, 3}, {2, 4}, {2, 5}, {4, 6}};
    vector<int> cover1 = solution.Vertex_Cover_Approximate(V1, edges1);
    cout << "Vertex Cover: ";
    for (int v : cover1) {
        cout << v << " ";
    }
    cout << endl;
    cout << "Size: " << cover1.size() << endl;
    cout << endl;
    
    cout << "Test Case 2: Tree Graph" << endl;
    int V2 = 5;
    vector<pair<int, int>> edges2 = {{0, 1}, {0, 2}, {1, 3}, {1, 4}};
    vector<int> cover2 = solution.Vertex_Cover_Approximate(V2, edges2);
    cout << "Approximate Vertex Cover: ";
    for (int v : cover2) {
        cout << v << " ";
    }
    cout << endl;
    cout << "Size: " << cover2.size() << endl;
    int optimal2 = solution.Vertex_Cover_Tree_DP(V2, edges2, 0);
    cout << "Optimal Tree DP Size: " << optimal2 << endl;
    cout << endl;
    
    cout << "Test Case 3: Complete Graph K3" << endl;
    int V3 = 3;
    vector<pair<int, int>> edges3 = {{0, 1}, {1, 2}, {2, 0}};
    vector<int> cover3 = solution.Vertex_Cover_Approximate(V3, edges3);
    cout << "Vertex Cover: ";
    for (int v : cover3) {
        cout << v << " ";
    }
    cout << endl;
    cout << "Size: " << cover3.size() << endl;
}

int main() {
    Test_Vertex_Cover();
    return 0;
}
