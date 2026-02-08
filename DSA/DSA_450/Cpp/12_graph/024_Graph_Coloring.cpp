/*
Problem: Graph Coloring Problem
URL: https://www.geeksforgeeks.org/graph-coloring-applications/

Problem Statement:
Assign colors to vertices such that no two adjacent vertices share the same color. Find the minimum number of colors needed (chromatic number approximation) or check if a graph can be colored with m colors.

Sample Input/Output:
Input: Graph with edges, number of colors
Output: Color assignment or whether coloring is possible
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Graph_Coloring_Greedy(int V, vector<vector<int>>& adj) {
        /*
        Greedy coloring
        Time Complexity: O(V^2)
        Space Complexity: O(V)
        */
        vector<int> color(V, -1);
        color[0] = 0;
        
        vector<bool> available(V, false);
        
        for (int u = 1; u < V; u++) {
            for (int v : adj[u]) {
                if (color[v] != -1) {
                    available[color[v]] = true;
                }
            }
            
            int cr;
            for (cr = 0; cr < V; cr++) {
                if (!available[cr]) break;
            }
            
            color[u] = cr;
            
            for (int v : adj[u]) {
                if (color[v] != -1) {
                    available[color[v]] = false;
                }
            }
        }
        
        int maxColor = *max_element(color.begin(), color.end());
        return maxColor + 1;
    }
    
    bool Graph_Coloring_Backtracking(int V, vector<vector<int>>& adj, int m) {
        /*
        Try m colors with backtracking
        Time Complexity: O(m^V)
        Space Complexity: O(V)
        */
        vector<int> color(V, -1);
        
        function<bool(int)> isSafe = [&](int u) -> bool {
            for (int v : adj[u]) {
                if (color[v] != -1 && color[v] == color[u]) {
                    return false;
                }
            }
            return true;
        };
        
        function<bool(int)> solve = [&](int u) -> bool {
            if (u == V) return true;
            
            for (int c = 0; c < m; c++) {
                color[u] = c;
                if (isSafe(u) && solve(u + 1)) {
                    return true;
                }
                color[u] = -1;
            }
            
            return false;
        };
        
        return solve(0);
    }
    
    vector<int> Get_Coloring_Result(int V, vector<vector<int>>& adj, int m) {
        vector<int> color(V, -1);
        
        function<bool(int)> isSafe = [&](int u) -> bool {
            for (int v : adj[u]) {
                if (color[v] != -1 && color[v] == color[u]) {
                    return false;
                }
            }
            return true;
        };
        
        function<bool(int)> solve = [&](int u) -> bool {
            if (u == V) return true;
            
            for (int c = 0; c < m; c++) {
                color[u] = c;
                if (isSafe(u) && solve(u + 1)) {
                    return true;
                }
                color[u] = -1;
            }
            
            return false;
        };
        
        solve(0);
        return color;
    }
};

void Test_Graph_Coloring() {
    Solution solution;
    
    cout << "Test Case 1: Simple graph" << endl;
    int V1 = 4;
    vector<vector<int>> adj1(4);
    adj1[0].push_back(1);
    adj1[0].push_back(2);
    adj1[1].push_back(0);
    adj1[1].push_back(3);
    adj1[2].push_back(0);
    adj1[2].push_back(3);
    adj1[3].push_back(1);
    adj1[3].push_back(2);
    
    int minColors1 = solution.Graph_Coloring_Greedy(V1, adj1);
    cout << "Greedy minimum colors: " << minColors1 << endl;
    
    bool canColor1 = solution.Graph_Coloring_Backtracking(V1, adj1, 2);
    cout << "Can color with 2 colors: " << (canColor1 ? "Yes" : "No") << endl;
    
    cout << "\nTest Case 2: Complete graph K4" << endl;
    int V2 = 4;
    vector<vector<int>> adj2(4);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i != j) adj2[i].push_back(j);
        }
    }
    
    int minColors2 = solution.Graph_Coloring_Greedy(V2, adj2);
    cout << "Greedy minimum colors: " << minColors2 << endl;
    
    bool canColor2 = solution.Graph_Coloring_Backtracking(V2, adj2, 4);
    cout << "Can color with 4 colors: " << (canColor2 ? "Yes" : "No") << endl;
    
    cout << "\nTest Case 3: Bipartite graph" << endl;
    int V3 = 6;
    vector<vector<int>> adj3(6);
    adj3[0].push_back(1);
    adj3[0].push_back(3);
    adj3[1].push_back(0);
    adj3[1].push_back(2);
    adj3[2].push_back(1);
    adj3[2].push_back(4);
    adj3[3].push_back(0);
    adj3[3].push_back(4);
    adj3[4].push_back(2);
    adj3[4].push_back(3);
    adj3[4].push_back(5);
    adj3[5].push_back(4);
    
    int minColors3 = solution.Graph_Coloring_Greedy(V3, adj3);
    cout << "Greedy minimum colors: " << minColors3 << endl;
    
    bool canColor3 = solution.Graph_Coloring_Backtracking(V3, adj3, 2);
    cout << "Can color with 2 colors: " << (canColor3 ? "Yes" : "No") << endl;
    
    if (canColor3) {
        vector<int> coloring = solution.Get_Coloring_Result(V3, adj3, 2);
        cout << "Coloring: ";
        for (int c : coloring) cout << c << " ";
        cout << endl;
    }
}

int main() {
    Test_Graph_Coloring();
    return 0;
}
