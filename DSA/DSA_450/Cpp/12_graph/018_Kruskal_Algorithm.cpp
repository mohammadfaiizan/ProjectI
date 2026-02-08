/*
Problem: Kruskal's Minimum Spanning Tree Algorithm
URL: https://practice.geeksforgeeks.org/problems/minimum-spanning-tree/1

Problem Statement:
Find the Minimum Spanning Tree (MST) of a weighted undirected graph using Kruskal's algorithm. The algorithm sorts all edges by weight and uses Union-Find to avoid cycles.

Sample Input/Output:
Input: Graph with edges (0,1,10), (0,2,6), (0,3,5), (1,3,15), (2,3,4)
Output: MST weight = 19
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Kruskal_MST(int V, vector<vector<int>>& edges) {
        /*
        Sort edges by weight, union-find with path compression and rank
        Time Complexity: O(E log E)
        Space Complexity: O(V)
        */
        sort(edges.begin(), edges.end(), [](const vector<int>& a, const vector<int>& b) {
            return a[2] < b[2];
        });
        
        vector<int> parent(V);
        vector<int> rank(V, 0);
        for (int i = 0; i < V; i++) {
            parent[i] = i;
        }
        
        function<int(int)> find = [&](int x) -> int {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        };
        
        function<bool(int, int)> unite = [&](int x, int y) -> bool {
            x = find(x);
            y = find(y);
            if (x == y) return false;
            if (rank[x] < rank[y]) swap(x, y);
            parent[y] = x;
            if (rank[x] == rank[y]) rank[x]++;
            return true;
        };
        
        int mstWeight = 0;
        int edgesAdded = 0;
        
        for (auto& edge : edges) {
            int u = edge[0];
            int v = edge[1];
            int w = edge[2];
            
            if (unite(u, v)) {
                mstWeight += w;
                edgesAdded++;
                if (edgesAdded == V - 1) break;
            }
        }
        
        return mstWeight;
    }
};

void Test_Kruskal() {
    Solution solution;
    
    cout << "Test Case 1: Weighted graph with 5 vertices" << endl;
    int V1 = 4;
    vector<vector<int>> edges1 = {
        {0, 1, 10},
        {0, 2, 6},
        {0, 3, 5},
        {1, 3, 15},
        {2, 3, 4}
    };
    cout << "MST Weight: " << solution.Kruskal_MST(V1, edges1) << endl;
    
    cout << "\nTest Case 2: Complete graph" << endl;
    int V2 = 5;
    vector<vector<int>> edges2 = {
        {0, 1, 2},
        {0, 2, 3},
        {0, 3, 6},
        {0, 4, 5},
        {1, 2, 5},
        {1, 3, 3},
        {1, 4, 4},
        {2, 3, 1},
        {2, 4, 2},
        {3, 4, 3}
    };
    cout << "MST Weight: " << solution.Kruskal_MST(V2, edges2) << endl;
    
    cout << "\nTest Case 3: Simple triangle" << endl;
    int V3 = 3;
    vector<vector<int>> edges3 = {
        {0, 1, 1},
        {1, 2, 2},
        {0, 2, 3}
    };
    cout << "MST Weight: " << solution.Kruskal_MST(V3, edges3) << endl;
}

int main() {
    Test_Kruskal();
    return 0;
}
