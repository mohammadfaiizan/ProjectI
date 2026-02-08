/*
Problem: Cheapest Flights Within K Stops
URL: https://leetcode.com/problems/cheapest-flights-within-k-stops/

Problem Statement:
Find the cheapest flight from src to dst with at most K stops. Given flights array where flights[i] = [fromi, toi, pricei].

Sample Input/Output:
Input: n=4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src=0, dst=3, k=1
Output: 700
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Cheapest_Flights_Bellman_Ford(int n, vector<vector<int>>& flights, int src, int dst, int k) {
        /*
        Modified Bellman-Ford with K+1 iterations
        Time Complexity: O(K*E)
        Space Complexity: O(V)
        */
        vector<int> dist(n, INT_MAX);
        dist[src] = 0;
        
        for (int i = 0; i <= k; i++) {
            vector<int> temp = dist;
            for (auto& flight : flights) {
                int u = flight[0], v = flight[1], w = flight[2];
                if (dist[u] != INT_MAX) {
                    temp[v] = min(temp[v], dist[u] + w);
                }
            }
            dist = temp;
        }
        
        return dist[dst] == INT_MAX ? -1 : dist[dst];
    }
    
    int Cheapest_Flights_BFS(int n, vector<vector<int>>& flights, int src, int dst, int k) {
        /*
        BFS with cost tracking
        Time Complexity: O(V*E)
        Space Complexity: O(V)
        */
        vector<vector<pair<int, int>>> adj(n);
        for (auto& flight : flights) {
            adj[flight[0]].push_back({flight[1], flight[2]});
        }
        
        queue<pair<int, int>> q;
        vector<int> cost(n, INT_MAX);
        q.push({src, 0});
        cost[src] = 0;
        int stops = 0;
        
        while (!q.empty() && stops <= k) {
            int sz = q.size();
            while (sz--) {
                auto [u, c] = q.front();
                q.pop();
                
                for (auto& [v, w] : adj[u]) {
                    if (c + w < cost[v]) {
                        cost[v] = c + w;
                        q.push({v, cost[v]});
                    }
                }
            }
            stops++;
        }
        
        return cost[dst] == INT_MAX ? -1 : cost[dst];
    }
    
    int Cheapest_Flights_Dijkstra(int n, vector<vector<int>>& flights, int src, int dst, int k) {
        /*
        Priority queue with stops constraint
        Time Complexity: O(E log V)
        Space Complexity: O(V)
        */
        vector<vector<pair<int, int>>> adj(n);
        for (auto& flight : flights) {
            adj[flight[0]].push_back({flight[1], flight[2]});
        }
        
        priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, greater<tuple<int, int, int>>> pq;
        pq.push({0, src, 0});
        vector<vector<int>> dist(n, vector<int>(k + 2, INT_MAX));
        dist[src][0] = 0;
        
        while (!pq.empty()) {
            auto [cost, u, stops] = pq.top();
            pq.pop();
            
            if (u == dst) {
                return cost;
            }
            
            if (stops > k) continue;
            
            for (auto& [v, w] : adj[u]) {
                int newCost = cost + w;
                if (newCost < dist[v][stops + 1]) {
                    dist[v][stops + 1] = newCost;
                    pq.push({newCost, v, stops + 1});
                }
            }
        }
        
        return -1;
    }
};

void Test_Cheapest_Flights_Bellman_Ford() {
    Solution solution;
    
    int n1 = 4;
    vector<vector<int>> flights1 = {{0,1,100},{1,2,100},{2,0,100},{1,3,600},{2,3,200}};
    cout << "Test 1 Bellman-Ford: " << solution.Cheapest_Flights_Bellman_Ford(n1, flights1, 0, 3, 1) << endl;
    
    int n2 = 3;
    vector<vector<int>> flights2 = {{0,1,100},{1,2,100},{0,2,500}};
    cout << "Test 2 BFS: " << solution.Cheapest_Flights_BFS(n2, flights2, 0, 2, 1) << endl;
    
    int n3 = 3;
    vector<vector<int>> flights3 = {{0,1,100},{1,2,100},{0,2,500}};
    cout << "Test 3 Dijkstra: " << solution.Cheapest_Flights_Dijkstra(n3, flights3, 0, 2, 0) << endl;
}

int main() {
    Test_Cheapest_Flights_Bellman_Ford();
    return 0;
}
