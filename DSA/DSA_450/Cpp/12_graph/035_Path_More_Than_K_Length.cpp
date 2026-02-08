/*
Problem: Find if There is a Path of More Than K Length From a Source
URL: https://www.geeksforgeeks.org/find-if-there-is-a-path-of-more-than-k-length-from-a-source/

Problem Statement:
Check if there exists a simple path (no repeated vertices) of total weight > K from a source vertex in a weighted graph.

Sample Input/Output:
Input: V=9, edges = [[0,1,4],[0,7,8],[1,2,8],[1,7,11],[2,3,7],[2,8,2],[2,5,4],[3,4,9],[3,5,14],[4,5,10],[5,6,2],[6,7,1],[6,8,6],[7,8,7]], src=0, k=58
Output: true
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Path_K_Backtracking(int V, vector<vector<int>>& edges, int src, int k) {
        /*
        DFS backtracking, avoid revisiting
        Time Complexity: O(V!)
        Space Complexity: O(V+E)
        */
        vector<vector<pair<int, int>>> adj(V);
        for (auto& e : edges) {
            adj[e[0]].push_back({e[1], e[2]});
            adj[e[1]].push_back({e[0], e[2]});
        }
        
        vector<bool> visited(V, false);
        
        function<bool(int, int)> dfs = [&](int u, int pathLen) {
            if (pathLen > k) {
                return true;
            }
            
            visited[u] = true;
            for (auto& [v, w] : adj[u]) {
                if (!visited[v]) {
                    if (dfs(v, pathLen + w)) {
                        return true;
                    }
                }
            }
            visited[u] = false;
            
            return false;
        };
        
        return dfs(src, 0);
    }
};

void Test_Path_K_Backtracking() {
    Solution solution;
    
    int V1 = 9;
    vector<vector<int>> edges1 = {{0,1,4},{0,7,8},{1,2,8},{1,7,11},{2,3,7},{2,8,2},{2,5,4},{3,4,9},{3,5,14},{4,5,10},{5,6,2},{6,7,1},{6,8,6},{7,8,7}};
    cout << "Test 1 (k=58): " << (solution.Path_K_Backtracking(V1, edges1, 0, 58) ? "true" : "false") << endl;
    
    int V2 = 4;
    vector<vector<int>> edges2 = {{0,1,10},{1,2,20},{2,3,30},{0,3,40}};
    cout << "Test 2 (k=50): " << (solution.Path_K_Backtracking(V2, edges2, 0, 50) ? "true" : "false") << endl;
    
    int V3 = 3;
    vector<vector<int>> edges3 = {{0,1,5},{1,2,5},{0,2,5}};
    cout << "Test 3 (k=15): " << (solution.Path_K_Backtracking(V3, edges3, 0, 15) ? "true" : "false") << endl;
}

int main() {
    Test_Path_K_Backtracking();
    return 0;
}
