/*
Problem: Dijkstra's Shortest Path Algorithm
URL: https://practice.geeksforgeeks.org/problems/implementing-dijkstra-set-1-adjacency-matrix/1

Problem Statement:
Find shortest distance from source to all vertices in a weighted graph with non-negative weights.

Sample Input/Output:
Input: Weighted graph with 5+ nodes
Output: Shortest distances from source to all vertices
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> Dijkstra_Priority_Queue(int V, vector<vector<pair<int, int>>>& adj, int src) {
        /*
        Min-Heap Based
        Time Complexity: O((V+E) log V)
        Space Complexity: O(V)
        */
        vector<int> dist(V, INT_MAX);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        
        dist[src] = 0;
        pq.push({0, src});
        
        while (!pq.empty()) {
            int u = pq.top().second;
            int d = pq.top().first;
            pq.pop();
            
            if (d > dist[u]) continue;
            
            for (auto& edge : adj[u]) {
                int v = edge.first;
                int weight = edge.second;
                
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});
                }
            }
        }
        
        return dist;
    }

    vector<int> Dijkstra_Set(int V, vector<vector<pair<int, int>>>& adj, int src) {
        /*
        Set-Based
        Time Complexity: O((V+E) log V)
        Space Complexity: O(V)
        */
        vector<int> dist(V, INT_MAX);
        set<pair<int, int>> s;
        
        dist[src] = 0;
        s.insert({0, src});
        
        while (!s.empty()) {
            auto it = s.begin();
            int u = it->second;
            int d = it->first;
            s.erase(it);
            
            for (auto& edge : adj[u]) {
                int v = edge.first;
                int weight = edge.second;
                
                if (dist[u] + weight < dist[v]) {
                    if (dist[v] != INT_MAX) {
                        s.erase({dist[v], v});
                    }
                    dist[v] = dist[u] + weight;
                    s.insert({dist[v], v});
                }
            }
        }
        
        return dist;
    }
};

void Test_Dijkstra_Algorithm() {
    Solution solution;
    
    cout << "Test: Dijkstra's Algorithm" << endl;
    int V = 5;
    vector<vector<pair<int, int>>> adj(V);
    
    adj[0].push_back({1, 4});
    adj[0].push_back({2, 1});
    adj[1].push_back({3, 1});
    adj[2].push_back({1, 2});
    adj[2].push_back({3, 5});
    adj[3].push_back({4, 3});
    
    int src = 0;
    
    vector<int> dist1 = solution.Dijkstra_Priority_Queue(V, adj, src);
    cout << "Shortest distances from source " << src << " (Priority Queue):" << endl;
    for (int i = 0; i < V; i++) {
        cout << "Distance to " << i << ": " << (dist1[i] == INT_MAX ? -1 : dist1[i]) << endl;
    }
    
    vector<int> dist2 = solution.Dijkstra_Set(V, adj, src);
    cout << "\nShortest distances from source " << src << " (Set):" << endl;
    for (int i = 0; i < V; i++) {
        cout << "Distance to " << i << ": " << (dist2[i] == INT_MAX ? -1 : dist2[i]) << endl;
    }
    
    cout << "\nTest 2: Larger Graph" << endl;
    int V2 = 6;
    vector<vector<pair<int, int>>> adj2(V2);
    
    adj2[0].push_back({1, 5});
    adj2[0].push_back({2, 3});
    adj2[1].push_back({3, 6});
    adj2[1].push_back({2, 2});
    adj2[2].push_back({4, 4});
    adj2[2].push_back({5, 2});
    adj2[3].push_back({4, 1});
    adj2[4].push_back({5, 3});
    
    int src2 = 0;
    vector<int> dist3 = solution.Dijkstra_Priority_Queue(V2, adj2, src2);
    cout << "Shortest distances from source " << src2 << ":" << endl;
    for (int i = 0; i < V2; i++) {
        cout << "Distance to " << i << ": " << (dist3[i] == INT_MAX ? -1 : dist3[i]) << endl;
    }
}

int main() {
    Test_Dijkstra_Algorithm();
    return 0;
}
