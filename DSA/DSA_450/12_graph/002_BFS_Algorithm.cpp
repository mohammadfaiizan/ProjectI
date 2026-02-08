/*
Problem: Implement BFS Algorithm
URL: https://practice.geeksforgeeks.org/problems/bfs-traversal-of-graph/1

Problem Statement:
Implement Breadth-First Search traversal for both connected and disconnected graphs.

Sample Input/Output:
Input: Connected graph with 5 vertices
Output: BFS traversal: 0 1 2 3 4
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> BFS_Connected(int V, vector<vector<int>>& adj, int start) {
        /*
        Single Source BFS
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<int> result;
        vector<bool> visited(V, false);
        queue<int> q;
        
        visited[start] = true;
        q.push(start);
        
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            result.push_back(node);
            
            for (int neighbor : adj[node]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
        
        return result;
    }

    vector<int> BFS_Disconnected(int V, vector<vector<int>>& adj) {
        /*
        BFS for Disconnected Graph
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<int> result;
        vector<bool> visited(V, false);
        
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                queue<int> q;
                visited[i] = true;
                q.push(i);
                
                while (!q.empty()) {
                    int node = q.front();
                    q.pop();
                    result.push_back(node);
                    
                    for (int neighbor : adj[node]) {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
            }
        }
        
        return result;
    }
};

void Test_BFS_Algorithm() {
    Solution solution;
    
    cout << "Test 1: Connected Graph" << endl;
    int V1 = 5;
    vector<vector<int>> adj1(V1);
    adj1[0] = {1, 2};
    adj1[1] = {0, 3, 4};
    adj1[2] = {0};
    adj1[3] = {1};
    adj1[4] = {1};
    
    vector<int> bfs1 = solution.BFS_Connected(V1, adj1, 0);
    cout << "BFS Traversal: ";
    for (int node : bfs1) {
        cout << node << " ";
    }
    cout << endl;
    
    cout << "\nTest 2: Disconnected Graph" << endl;
    int V2 = 6;
    vector<vector<int>> adj2(V2);
    adj2[0] = {1};
    adj2[1] = {0};
    adj2[2] = {3};
    adj2[3] = {2};
    adj2[4] = {5};
    adj2[5] = {4};
    
    vector<int> bfs2 = solution.BFS_Disconnected(V2, adj2);
    cout << "BFS Traversal: ";
    for (int node : bfs2) {
        cout << node << " ";
    }
    cout << endl;
}

int main() {
    Test_BFS_Algorithm();
    return 0;
}
