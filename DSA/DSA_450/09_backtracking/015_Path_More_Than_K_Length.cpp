/*
Problem: Path More Than K Length
URL: https://www.geeksforgeeks.org/find-if-there-is-a-path-of-more-than-k-length-from-a-source/

Problem Statement:
Given a weighted graph and a source vertex, determine if there exists a path of length >= K starting from the source. Graph is represented as adjacency list with weights.

Sample Input/Output:
Input: Graph edges: (0,1,4), (0,7,8), (1,2,8), (1,7,11), (2,3,7), (2,8,2), (2,5,4), (3,4,9), (3,5,14), (4,5,10), (5,6,2), (6,7,1), (6,8,6), (7,8,7)
       Source = 0, K = 58
Output: true
Explanation: Path exists with length >= 58
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    bool Path_More_Than_K_Length_DFS(vector<vector<pair<int, int>>>& graph, int source, int k) {
        /*
        DFS backtracking with weight accumulation
        Time Complexity: O(V!)
        Space Complexity: O(V)
        */
        int V = graph.size();
        vector<bool> visited(V, false);
        
        function<bool(int, int)> dfs = [&](int u, int current_length) -> bool {
            if (current_length >= k) {
                return true;
            }
            
            visited[u] = true;
            
            for (auto& [v, weight] : graph[u]) {
                if (!visited[v]) {
                    if (dfs(v, current_length + weight)) {
                        return true;
                    }
                }
            }
            
            visited[u] = false;
            return false;
        };
        
        return dfs(source, 0);
    }
};

void Test_Path_More_Than_K_Length() {
    Solution solution;
    int V = 9;
    vector<vector<pair<int, int>>> graph(V);
    
    graph[0].push_back({1, 4});
    graph[0].push_back({7, 8});
    graph[1].push_back({0, 4});
    graph[1].push_back({2, 8});
    graph[1].push_back({7, 11});
    graph[2].push_back({1, 8});
    graph[2].push_back({3, 7});
    graph[2].push_back({8, 2});
    graph[2].push_back({5, 4});
    graph[3].push_back({2, 7});
    graph[3].push_back({4, 9});
    graph[3].push_back({5, 14});
    graph[4].push_back({3, 9});
    graph[4].push_back({5, 10});
    graph[5].push_back({2, 4});
    graph[5].push_back({3, 14});
    graph[5].push_back({4, 10});
    graph[5].push_back({6, 2});
    graph[6].push_back({5, 2});
    graph[6].push_back({7, 1});
    graph[6].push_back({8, 6});
    graph[7].push_back({0, 8});
    graph[7].push_back({1, 11});
    graph[7].push_back({6, 1});
    graph[7].push_back({8, 7});
    graph[8].push_back({2, 2});
    graph[8].push_back({6, 6});
    graph[8].push_back({7, 7});
    
    int source = 0;
    int k = 58;
    bool result = solution.Path_More_Than_K_Length_DFS(graph, source, k);
    cout << "Path with length >= " << k << " exists: " << (result ? "true" : "false") << endl;
}

int main() {
    Test_Path_More_Than_K_Length();
    return 0;
}
