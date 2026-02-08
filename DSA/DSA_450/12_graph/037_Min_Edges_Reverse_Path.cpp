/*
Problem: Minimum Edges to Reverse to Make Path from Source to Destination
URL: https://www.geeksforgeeks.org/minimum-edges-reverse-make-path-source-destination/

Problem Statement:
Given a directed graph and a source vertex and destination vertex, find the minimum number of edges that need to be reversed to make a path from source to destination.

Sample Input/Output:
Input: Directed graph with edges, src=0, dst=6
Output: Minimum edges to reverse
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Edges_Reverse_BFS_01(int V, vector<pair<int, int>>& edges, int src, int dst) {
        /*
        0-1 BFS using deque
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        vector<vector<pair<int, int>>> adj(V);
        for (auto& edge : edges) {
            int u = edge.first;
            int v = edge.second;
            adj[u].push_back({v, 0});
            adj[v].push_back({u, 1});
        }
        
        deque<pair<int, int>> dq;
        vector<int> dist(V, INT_MAX);
        dist[src] = 0;
        dq.push_front({src, 0});
        
        while (!dq.empty()) {
            int u = dq.front().first;
            int cost = dq.front().second;
            dq.pop_front();
            
            if (u == dst) return cost;
            
            for (auto& neighbor : adj[u]) {
                int v = neighbor.first;
                int weight = neighbor.second;
                
                if (dist[v] > dist[u] + weight) {
                    dist[v] = dist[u] + weight;
                    if (weight == 0) {
                        dq.push_front({v, dist[v]});
                    } else {
                        dq.push_back({v, dist[v]});
                    }
                }
            }
        }
        
        return -1;
    }

    int Min_Edges_Reverse_Dijkstra(int V, vector<pair<int, int>>& edges, int src, int dst) {
        /*
        Dijkstra on modified graph
        Time Complexity: O((V+E) log V)
        Space Complexity: O(V+E)
        */
        vector<vector<pair<int, int>>> adj(V);
        for (auto& edge : edges) {
            int u = edge.first;
            int v = edge.second;
            adj[u].push_back({v, 0});
            adj[v].push_back({u, 1});
        }
        
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        vector<int> dist(V, INT_MAX);
        dist[src] = 0;
        pq.push({0, src});
        
        while (!pq.empty()) {
            int u = pq.top().second;
            int cost = pq.top().first;
            pq.pop();
            
            if (u == dst) return cost;
            if (cost > dist[u]) continue;
            
            for (auto& neighbor : adj[u]) {
                int v = neighbor.first;
                int weight = neighbor.second;
                
                if (dist[v] > dist[u] + weight) {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});
                }
            }
        }
        
        return -1;
    }
};

void Test_Min_Edges_Reverse() {
    Solution solution;
    int V = 7;
    vector<pair<int, int>> edges = {{0, 1}, {2, 1}, {2, 3}, {5, 1}, {4, 5}, {6, 4}, {6, 3}};
    int src = 0, dst = 6;
    
    cout << "Test Case 1:" << endl;
    cout << "Source: " << src << ", Destination: " << dst << endl;
    int result1 = solution.Min_Edges_Reverse_BFS_01(V, edges, src, dst);
    cout << "0-1 BFS Result: " << result1 << " edges to reverse" << endl;
    
    int result2 = solution.Min_Edges_Reverse_Dijkstra(V, edges, src, dst);
    cout << "Dijkstra Result: " << result2 << " edges to reverse" << endl;
    cout << endl;
    
    V = 4;
    edges = {{0, 1}, {2, 0}, {2, 3}, {3, 1}};
    src = 0;
    dst = 3;
    cout << "Test Case 2:" << endl;
    cout << "Source: " << src << ", Destination: " << dst << endl;
    result1 = solution.Min_Edges_Reverse_BFS_01(V, edges, src, dst);
    cout << "0-1 BFS Result: " << result1 << " edges to reverse" << endl;
    result2 = solution.Min_Edges_Reverse_Dijkstra(V, edges, src, dst);
    cout << "Dijkstra Result: " << result2 << " edges to reverse" << endl;
}

int main() {
    Test_Min_Edges_Reverse();
    return 0;
}
