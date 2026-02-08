/*
Problem: Topological Sort
URL: https://practice.geeksforgeeks.org/problems/topological-sort/1

Problem Statement:
Given a Directed Acyclic Graph (DAG), perform topological sort to find a linear ordering of vertices such that for every directed edge (u, v), vertex u comes before v in the ordering.

Sample Input/Output:
Input: Graph with 6 vertices, edges: 5->0, 5->2, 4->0, 4->1, 2->3, 3->1
Output: 5 4 2 3 1 0 (or other valid topological order)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Topological_Sort_DFS(int V, vector<int> adj[]) {
        /*
        DFS with stack
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<bool> visited(V, false);
        stack<int> st;
        
        function<void(int)> dfs = [&](int u) {
            visited[u] = true;
            for (int v : adj[u]) {
                if (!visited[v]) {
                    dfs(v);
                }
            }
            st.push(u);
        };
        
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                dfs(i);
            }
        }
        
        vector<int> result;
        while (!st.empty()) {
            result.push_back(st.top());
            st.pop();
        }
        return result;
    }
    
    vector<int> Topological_Sort_BFS_Kahn(int V, vector<int> adj[]) {
        /*
        Kahn's algorithm with in-degree
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<int> inDegree(V, 0);
        for (int u = 0; u < V; u++) {
            for (int v : adj[u]) {
                inDegree[v]++;
            }
        }
        
        queue<int> q;
        for (int i = 0; i < V; i++) {
            if (inDegree[i] == 0) {
                q.push(i);
            }
        }
        
        vector<int> result;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            result.push_back(u);
            
            for (int v : adj[u]) {
                inDegree[v]--;
                if (inDegree[v] == 0) {
                    q.push(v);
                }
            }
        }
        
        return result;
    }
};

void Test_Topological_Sort() {
    Solution solution;
    
    int V = 6;
    vector<int> adj[6];
    adj[5].push_back(0);
    adj[5].push_back(2);
    adj[4].push_back(0);
    adj[4].push_back(1);
    adj[2].push_back(3);
    adj[3].push_back(1);
    
    cout << "Test Case 1: DAG with 6 vertices" << endl;
    
    vector<int> result1 = solution.Topological_Sort_DFS(V, adj);
    cout << "DFS Topological Sort: ";
    for (int x : result1) cout << x << " ";
    cout << endl;
    
    vector<int> result2 = solution.Topological_Sort_BFS_Kahn(V, adj);
    cout << "BFS Kahn Topological Sort: ";
    for (int x : result2) cout << x << " ";
    cout << endl;
    
    cout << "Test Case 2: Simple linear DAG" << endl;
    int V2 = 4;
    vector<int> adj2[4];
    adj2[0].push_back(1);
    adj2[1].push_back(2);
    adj2[2].push_back(3);
    
    vector<int> result3 = solution.Topological_Sort_DFS(V2, adj2);
    cout << "DFS Result: ";
    for (int x : result3) cout << x << " ";
    cout << endl;
}

int main() {
    Test_Topological_Sort();
    return 0;
}
