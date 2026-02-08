/*
Problem: Prim's Minimum Spanning Tree Algorithm
URL: https://practice.geeksforgeeks.org/problems/minimum-spanning-tree/1

Problem Statement:
Find the Minimum Spanning Tree (MST) of a weighted undirected graph using Prim's algorithm. The algorithm starts from any vertex and greedily adds the minimum weight edge connecting a vertex in MST to a vertex outside MST.

Sample Input/Output:
Input: Graph with edges (0,1,10), (0,2,6), (0,3,5), (1,3,15), (2,3,4)
Output: MST weight = 19
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Prim_MST_Priority_Queue(int V, vector<vector<pair<int, int>>>& adj) {
        /*
        Min-heap based
        Time Complexity: O((V+E) log V)
        Space Complexity: O(V)
        */
        vector<bool> inMST(V, false);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        
        pq.push({0, 0});
        int mstWeight = 0;
        
        while (!pq.empty()) {
            auto [weight, u] = pq.top();
            pq.pop();
            
            if (inMST[u]) continue;
            
            inMST[u] = true;
            mstWeight += weight;
            
            for (auto [v, w] : adj[u]) {
                if (!inMST[v]) {
                    pq.push({w, v});
                }
            }
        }
        
        return mstWeight;
    }
    
    int Prim_MST_Adjacency_Matrix(int V, vector<vector<int>>& graph) {
        /*
        Brute force with adjacency matrix
        Time Complexity: O(V^2)
        Space Complexity: O(V)
        */
        vector<bool> inMST(V, false);
        vector<int> key(V, INT_MAX);
        key[0] = 0;
        int mstWeight = 0;
        
        for (int count = 0; count < V; count++) {
            int u = -1;
            for (int i = 0; i < V; i++) {
                if (!inMST[i] && (u == -1 || key[i] < key[u])) {
                    u = i;
                }
            }
            
            inMST[u] = true;
            mstWeight += key[u];
            
            for (int v = 0; v < V; v++) {
                if (graph[u][v] != 0 && !inMST[v] && graph[u][v] < key[v]) {
                    key[v] = graph[u][v];
                }
            }
        }
        
        return mstWeight;
    }
};

void Test_Prim() {
    Solution solution;
    
    cout << "Test Case 1: Weighted graph with adjacency list" << endl;
    int V1 = 4;
    vector<vector<pair<int, int>>> adj1(4);
    adj1[0].push_back({1, 10});
    adj1[0].push_back({2, 6});
    adj1[0].push_back({3, 5});
    adj1[1].push_back({0, 10});
    adj1[1].push_back({3, 15});
    adj1[2].push_back({0, 6});
    adj1[2].push_back({3, 4});
    adj1[3].push_back({0, 5});
    adj1[3].push_back({1, 15});
    adj1[3].push_back({2, 4});
    cout << "Priority Queue MST Weight: " << solution.Prim_MST_Priority_Queue(V1, adj1) << endl;
    
    cout << "\nTest Case 2: Weighted graph with adjacency matrix" << endl;
    int V2 = 4;
    vector<vector<int>> graph2 = {
        {0, 10, 6, 5},
        {10, 0, 0, 15},
        {6, 0, 0, 4},
        {5, 15, 4, 0}
    };
    cout << "Adjacency Matrix MST Weight: " << solution.Prim_MST_Adjacency_Matrix(V2, graph2) << endl;
    
    cout << "\nTest Case 3: Complete graph" << endl;
    int V3 = 5;
    vector<vector<pair<int, int>>> adj3(5);
    adj3[0].push_back({1, 2});
    adj3[0].push_back({2, 3});
    adj3[0].push_back({3, 6});
    adj3[0].push_back({4, 5});
    adj3[1].push_back({0, 2});
    adj3[1].push_back({2, 5});
    adj3[1].push_back({3, 3});
    adj3[1].push_back({4, 4});
    adj3[2].push_back({0, 3});
    adj3[2].push_back({1, 5});
    adj3[2].push_back({3, 1});
    adj3[2].push_back({4, 2});
    adj3[3].push_back({0, 6});
    adj3[3].push_back({1, 3});
    adj3[3].push_back({2, 1});
    adj3[3].push_back({4, 3});
    adj3[4].push_back({0, 5});
    adj3[4].push_back({1, 4});
    adj3[4].push_back({2, 2});
    adj3[4].push_back({3, 3});
    cout << "Priority Queue MST Weight: " << solution.Prim_MST_Priority_Queue(V3, adj3) << endl;
}

int main() {
    Test_Prim();
    return 0;
}
