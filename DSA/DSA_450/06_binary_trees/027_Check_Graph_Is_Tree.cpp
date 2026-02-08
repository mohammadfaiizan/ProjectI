/*
Problem: Check Graph Is Tree
URL: https://www.geeksforgeeks.org/check-given-graph-tree/

Problem Statement:
Check if an undirected graph is a tree (no cycle + all connected).

Sample Input/Output:
Input: 
Vertices: 5, Edges: 4
Edges: (0,1), (0,2), (0,3), (1,4)

Output: true
Explanation: Graph is connected and has no cycles.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Has_Cycle_DFS(vector<vector<int>>& graph, int node, int parent, vector<bool>& visited) {
        visited[node] = true;
        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                if (Has_Cycle_DFS(graph, neighbor, node, visited)) {
                    return true;
                }
            } else if (neighbor != parent) {
                return true;
            }
        }
        return false;
    }

    bool Is_Tree_DFS(vector<vector<int>>& graph, int V) {
        /*
        DFS cycle detection + connectivity check: Check for cycles and connectivity
        Time Complexity: O(V+E) where V is vertices and E is edges
        Space Complexity: O(V) for visited array and recursion stack
        */
        vector<bool> visited(V, false);
        if (Has_Cycle_DFS(graph, 0, -1, visited)) {
            return false;
        }
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                return false;
            }
        }
        return true;
    }

    bool Is_Tree_BFS(vector<vector<int>>& graph, int V) {
        /*
        BFS approach: Use BFS to check cycles and connectivity
        Time Complexity: O(V+E) where V is vertices and E is edges
        Space Complexity: O(V) for visited array and queue
        */
        vector<bool> visited(V, false);
        queue<pair<int, int>> q;
        q.push({0, -1});
        visited[0] = true;
        while (!q.empty()) {
            int node = q.front().first;
            int parent = q.front().second;
            q.pop();
            for (int neighbor : graph[node]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push({neighbor, node});
                } else if (neighbor != parent) {
                    return false;
                }
            }
        }
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                return false;
            }
        }
        return true;
    }
};

void Test_Check_Graph_Is_Tree() {
    Solution solution;
    
    int V1 = 5;
    vector<vector<int>> graph1(V1);
    graph1[0].push_back(1);
    graph1[0].push_back(2);
    graph1[0].push_back(3);
    graph1[1].push_back(0);
    graph1[1].push_back(4);
    graph1[2].push_back(0);
    graph1[3].push_back(0);
    graph1[4].push_back(1);
    cout << "Graph 1 (DFS): " << solution.Is_Tree_DFS(graph1, V1) << endl;
    cout << "Graph 1 (BFS): " << solution.Is_Tree_BFS(graph1, V1) << endl;
    
    int V2 = 3;
    vector<vector<int>> graph2(V2);
    graph2[0].push_back(1);
    graph2[1].push_back(0);
    graph2[1].push_back(2);
    graph2[2].push_back(1);
    graph2[0].push_back(2);
    graph2[2].push_back(0);
    cout << "Graph 2 (DFS): " << solution.Is_Tree_DFS(graph2, V2) << endl;
    cout << "Graph 2 (BFS): " << solution.Is_Tree_BFS(graph2, V2) << endl;
}

int main() {
    Test_Check_Graph_Is_Tree();
    return 0;
}
