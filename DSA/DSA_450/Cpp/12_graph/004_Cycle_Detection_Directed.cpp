/*
Problem: Detect Cycle in a Directed Graph
URL: https://practice.geeksforgeeks.org/problems/detect-cycle-in-a-directed-graph/1

Problem Statement:
Detect if a directed graph contains a cycle.

Sample Input/Output:
Input: Graph with cycle: 0->1->2->0
Output: Cycle detected: true
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Cycle_Directed_DFS_Helper(int node, vector<vector<int>>& adj, vector<bool>& visited, vector<bool>& recStack) {
        visited[node] = true;
        recStack[node] = true;
        
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                if (Cycle_Directed_DFS_Helper(neighbor, adj, visited, recStack)) {
                    return true;
                }
            } else if (recStack[neighbor]) {
                return true;
            }
        }
        
        recStack[node] = false;
        return false;
    }

    bool Cycle_Directed_DFS(int V, vector<vector<int>>& adj) {
        /*
        DFS with Recursion Stack Tracking
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<bool> visited(V, false);
        vector<bool> recStack(V, false);
        
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                if (Cycle_Directed_DFS_Helper(i, adj, visited, recStack)) {
                    return true;
                }
            }
        }
        
        return false;
    }

    bool Cycle_Directed_BFS_Kahn(int V, vector<vector<int>>& adj) {
        /*
        Kahn's Algorithm - Topological Sort
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<int> inDegree(V, 0);
        
        for (int i = 0; i < V; i++) {
            for (int neighbor : adj[i]) {
                inDegree[neighbor]++;
            }
        }
        
        queue<int> q;
        for (int i = 0; i < V; i++) {
            if (inDegree[i] == 0) {
                q.push(i);
            }
        }
        
        int count = 0;
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            count++;
            
            for (int neighbor : adj[node]) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    q.push(neighbor);
                }
            }
        }
        
        return count != V;
    }
};

void Test_Cycle_Detection_Directed() {
    Solution solution;
    
    cout << "Test 1: Graph with Cycle" << endl;
    int V1 = 4;
    vector<vector<int>> adj1(V1);
    adj1[0] = {1};
    adj1[1] = {2};
    adj1[2] = {3};
    adj1[3] = {1};
    
    bool hasCycle1 = solution.Cycle_Directed_DFS(V1, adj1);
    cout << "Cycle detected (DFS): " << (hasCycle1 ? "Yes" : "No") << endl;
    
    bool hasCycle1_bfs = solution.Cycle_Directed_BFS_Kahn(V1, adj1);
    cout << "Cycle detected (BFS/Kahn): " << (hasCycle1_bfs ? "Yes" : "No") << endl;
    
    cout << "\nTest 2: Graph without Cycle" << endl;
    int V2 = 4;
    vector<vector<int>> adj2(V2);
    adj2[0] = {1};
    adj2[1] = {2};
    adj2[2] = {3};
    
    bool hasCycle2 = solution.Cycle_Directed_DFS(V2, adj2);
    cout << "Cycle detected (DFS): " << (hasCycle2 ? "Yes" : "No") << endl;
    
    bool hasCycle2_bfs = solution.Cycle_Directed_BFS_Kahn(V2, adj2);
    cout << "Cycle detected (BFS/Kahn): " << (hasCycle2_bfs ? "Yes" : "No") << endl;
}

int main() {
    Test_Cycle_Detection_Directed();
    return 0;
}
