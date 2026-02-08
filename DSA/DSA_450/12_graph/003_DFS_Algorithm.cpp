/*
Problem: Implement DFS Algorithm
URL: https://practice.geeksforgeeks.org/problems/depth-first-traversal-for-a-graph/1

Problem Statement:
Implement Depth-First Search traversal for both connected and disconnected graphs.

Sample Input/Output:
Input: Connected graph with 5 vertices
Output: DFS traversal: 0 1 3 4 2
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void DFS_Recursive_Helper(int node, vector<vector<int>>& adj, vector<bool>& visited, vector<int>& result) {
        visited[node] = true;
        result.push_back(node);
        
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                DFS_Recursive_Helper(neighbor, adj, visited, result);
            }
        }
    }

    vector<int> DFS_Recursive(int V, vector<vector<int>>& adj, int start) {
        /*
        Recursive DFS
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<int> result;
        vector<bool> visited(V, false);
        DFS_Recursive_Helper(start, adj, visited, result);
        return result;
    }

    vector<int> DFS_Iterative(int V, vector<vector<int>>& adj, int start) {
        /*
        Iterative DFS using Stack
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        */
        vector<int> result;
        vector<bool> visited(V, false);
        stack<int> st;
        
        st.push(start);
        
        while (!st.empty()) {
            int node = st.top();
            st.pop();
            
            if (!visited[node]) {
                visited[node] = true;
                result.push_back(node);
                
                for (auto it = adj[node].rbegin(); it != adj[node].rend(); ++it) {
                    if (!visited[*it]) {
                        st.push(*it);
                    }
                }
            }
        }
        
        return result;
    }
};

void Test_DFS_Algorithm() {
    Solution solution;
    
    cout << "Test 1: Connected Graph (Recursive)" << endl;
    int V1 = 5;
    vector<vector<int>> adj1(V1);
    adj1[0] = {1, 2};
    adj1[1] = {0, 3, 4};
    adj1[2] = {0};
    adj1[3] = {1};
    adj1[4] = {1};
    
    vector<int> dfs1 = solution.DFS_Recursive(V1, adj1, 0);
    cout << "DFS Traversal: ";
    for (int node : dfs1) {
        cout << node << " ";
    }
    cout << endl;
    
    cout << "\nTest 2: Connected Graph (Iterative)" << endl;
    vector<int> dfs2 = solution.DFS_Iterative(V1, adj1, 0);
    cout << "DFS Traversal: ";
    for (int node : dfs2) {
        cout << node << " ";
    }
    cout << endl;
    
    cout << "\nTest 3: Disconnected Graph" << endl;
    int V2 = 6;
    vector<vector<int>> adj2(V2);
    adj2[0] = {1};
    adj2[1] = {0};
    adj2[2] = {3};
    adj2[3] = {2};
    adj2[4] = {5};
    adj2[5] = {4};
    
    vector<bool> visited(V2, false);
    vector<int> result;
    for (int i = 0; i < V2; i++) {
        if (!visited[i]) {
            solution.DFS_Recursive_Helper(i, adj2, visited, result);
        }
    }
    cout << "DFS Traversal: ";
    for (int node : result) {
        cout << node << " ";
    }
    cout << endl;
}

int main() {
    Test_DFS_Algorithm();
    return 0;
}
