/*
Problem: Oliver and the Game (Euler Tour / Ancestor Check)
URL: https://www.hackerearth.com/practice/algorithms/graphs/topological-sort/practice-problems/algorithm/oliver-and-the-game-3/

Problem Statement:
Given a rooted tree, answer queries: is node X an ancestor of node Y? Use Euler tour (in-time/out-time) to check subtree relationship.

Sample Input/Output:
Input: tree edges = [[0,1],[0,2],[1,3],[1,4]], queries = [[0,3],[2,4]]
Output: [true, false]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<bool> Oliver_Game_Euler_Tour(int n, vector<vector<int>>& edges, vector<vector<int>>& queries, int root) {
        /*
        DFS to compute in/out times, check subtree relationship
        Time Complexity: O(N+Q)
        Space Complexity: O(N)
        */
        vector<vector<int>> adj(n);
        for (auto& e : edges) {
            adj[e[0]].push_back(e[1]);
            adj[e[1]].push_back(e[0]);
        }
        
        vector<int> inTime(n), outTime(n);
        int timer = 0;
        
        function<void(int, int)> dfs = [&](int u, int parent) {
            inTime[u] = timer++;
            for (int v : adj[u]) {
                if (v != parent) {
                    dfs(v, u);
                }
            }
            outTime[u] = timer++;
        };
        
        dfs(root, -1);
        
        vector<bool> results;
        for (auto& q : queries) {
            int x = q[0], y = q[1];
            bool isAncestor = (inTime[x] <= inTime[y] && outTime[x] >= outTime[y]);
            results.push_back(isAncestor);
        }
        
        return results;
    }
};

void Test_Oliver_Game_Euler_Tour() {
    Solution solution;
    
    int n1 = 5;
    vector<vector<int>> edges1 = {{0,1},{0,2},{1,3},{1,4}};
    vector<vector<int>> queries1 = {{0,3},{2,4},{1,4}};
    vector<bool> result1 = solution.Oliver_Game_Euler_Tour(n1, edges1, queries1, 0);
    cout << "Test 1: ";
    for (bool r : result1) {
        cout << (r ? "true " : "false ");
    }
    cout << endl;
    
    int n2 = 4;
    vector<vector<int>> edges2 = {{0,1},{0,2},{2,3}};
    vector<vector<int>> queries2 = {{0,3},{1,2}};
    vector<bool> result2 = solution.Oliver_Game_Euler_Tour(n2, edges2, queries2, 0);
    cout << "Test 2: ";
    for (bool r : result2) {
        cout << (r ? "true " : "false ");
    }
    cout << endl;
    
    int n3 = 3;
    vector<vector<int>> edges3 = {{0,1},{0,2}};
    vector<vector<int>> queries3 = {{0,1},{0,2},{1,2}};
    vector<bool> result3 = solution.Oliver_Game_Euler_Tour(n3, edges3, queries3, 0);
    cout << "Test 3: ";
    for (bool r : result3) {
        cout << (r ? "true " : "false ");
    }
    cout << endl;
}

int main() {
    Test_Oliver_Game_Euler_Tour();
    return 0;
}
