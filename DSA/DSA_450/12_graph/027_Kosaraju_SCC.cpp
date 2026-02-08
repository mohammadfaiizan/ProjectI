/*
Problem: Strongly Connected Components (Kosaraju's Algorithm)
URL: https://practice.geeksforgeeks.org/problems/strongly-connected-components-kosarajus-algo/1

Problem Statement:
Find all strongly connected components (SCCs) in a directed graph. A strongly connected component is a maximal set of vertices such that every vertex can reach every other vertex in the set.

Sample Input/Output:
Input: V=5, edges = [[1,0],[0,2],[2,1],[0,3],[3,4]]
Output: 3 SCCs: [[0,1,2],[3],[4]]
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> SCC_Kosaraju(int V, vector<vector<int>>& edges) {
        /*
        Finish order DFS + transpose + DFS
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        vector<vector<int>> adj(V);
        vector<vector<int>> adjT(V);
        
        for (auto& e : edges) {
            adj[e[0]].push_back(e[1]);
            adjT[e[1]].push_back(e[0]);
        }
        
        vector<bool> visited(V, false);
        stack<int> st;
        
        function<void(int)> dfs1 = [&](int u) {
            visited[u] = true;
            for (int v : adj[u]) {
                if (!visited[v]) {
                    dfs1(v);
                }
            }
            st.push(u);
        };
        
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                dfs1(i);
            }
        }
        
        fill(visited.begin(), visited.end(), false);
        vector<vector<int>> sccs;
        
        function<void(int, vector<int>&)> dfs2 = [&](int u, vector<int>& comp) {
            visited[u] = true;
            comp.push_back(u);
            for (int v : adjT[u]) {
                if (!visited[v]) {
                    dfs2(v, comp);
                }
            }
        };
        
        while (!st.empty()) {
            int u = st.top();
            st.pop();
            if (!visited[u]) {
                vector<int> comp;
                dfs2(u, comp);
                sccs.push_back(comp);
            }
        }
        
        return sccs;
    }
    
    vector<vector<int>> SCC_Tarjan(int V, vector<vector<int>>& edges) {
        /*
        Tarjan's algorithm with low-link
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        vector<vector<int>> adj(V);
        for (auto& e : edges) {
            adj[e[0]].push_back(e[1]);
        }
        
        vector<int> disc(V, -1);
        vector<int> low(V, -1);
        vector<bool> inStack(V, false);
        stack<int> st;
        vector<vector<int>> sccs;
        int time = 0;
        
        function<void(int)> dfs = [&](int u) {
            disc[u] = low[u] = time++;
            st.push(u);
            inStack[u] = true;
            
            for (int v : adj[u]) {
                if (disc[v] == -1) {
                    dfs(v);
                    low[u] = min(low[u], low[v]);
                } else if (inStack[v]) {
                    low[u] = min(low[u], disc[v]);
                }
            }
            
            if (low[u] == disc[u]) {
                vector<int> comp;
                while (st.top() != u) {
                    int v = st.top();
                    st.pop();
                    inStack[v] = false;
                    comp.push_back(v);
                }
                st.pop();
                inStack[u] = false;
                comp.push_back(u);
                sccs.push_back(comp);
            }
        };
        
        for (int i = 0; i < V; i++) {
            if (disc[i] == -1) {
                dfs(i);
            }
        }
        
        return sccs;
    }
};

void Test_SCC_Kosaraju() {
    Solution solution;
    
    int V1 = 5;
    vector<vector<int>> edges1 = {{1,0},{0,2},{2,1},{0,3},{3,4}};
    vector<vector<int>> result1 = solution.SCC_Kosaraju(V1, edges1);
    cout << "Test 1 Kosaraju SCCs: " << result1.size() << endl;
    for (auto& scc : result1) {
        cout << "[";
        for (int v : scc) cout << v << " ";
        cout << "] ";
    }
    cout << endl;
    
    int V2 = 4;
    vector<vector<int>> edges2 = {{0,1},{1,2},{2,3},{3,0}};
    vector<vector<int>> result2 = solution.SCC_Kosaraju(V2, edges2);
    cout << "Test 2 Kosaraju SCCs: " << result2.size() << endl;
    
    int V3 = 3;
    vector<vector<int>> edges3 = {{0,1},{1,2}};
    vector<vector<int>> result3 = solution.SCC_Tarjan(V3, edges3);
    cout << "Test 3 Tarjan SCCs: " << result3.size() << endl;
}

int main() {
    Test_SCC_Kosaraju();
    return 0;
}
