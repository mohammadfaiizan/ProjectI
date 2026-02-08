/*
Problem: Journey to the Moon
URL: https://www.hackerrank.com/challenges/journey-to-the-moon/

Problem Statement:
Given N astronauts and pairs of astronauts from the same country, count the number of ways to choose 2 astronauts from different countries.

Sample Input/Output:
Input: n=5, pairs = [[0,1],[2,3],[0,4]]
Output: 6
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Journey_Moon_DFS(int n, vector<vector<int>>& pairs) {
        /*
        Find connected component sizes via DFS, compute pairs combinatorially
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        */
        vector<vector<int>> adj(n);
        for (auto& p : pairs) {
            adj[p[0]].push_back(p[1]);
            adj[p[1]].push_back(p[0]);
        }
        
        vector<bool> visited(n, false);
        vector<int> componentSizes;
        
        function<int(int)> dfs = [&](int u) {
            visited[u] = true;
            int size = 1;
            for (int v : adj[u]) {
                if (!visited[v]) {
                    size += dfs(v);
                }
            }
            return size;
        };
        
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                componentSizes.push_back(dfs(i));
            }
        }
        
        long long totalPairs = (long long)n * (n - 1) / 2;
        long long sameCountryPairs = 0;
        
        for (int size : componentSizes) {
            sameCountryPairs += (long long)size * (size - 1) / 2;
        }
        
        return totalPairs - sameCountryPairs;
    }
};

void Test_Journey_Moon_DFS() {
    Solution solution;
    
    int n1 = 5;
    vector<vector<int>> pairs1 = {{0,1},{2,3},{0,4}};
    cout << "Test 1: " << solution.Journey_Moon_DFS(n1, pairs1) << endl;
    
    int n2 = 4;
    vector<vector<int>> pairs2 = {{0,2}};
    cout << "Test 2: " << solution.Journey_Moon_DFS(n2, pairs2) << endl;
    
    int n3 = 6;
    vector<vector<int>> pairs3 = {{0,1},{2,3},{4,5}};
    cout << "Test 3: " << solution.Journey_Moon_DFS(n3, pairs3) << endl;
    
    int n4 = 3;
    vector<vector<int>> pairs4 = {};
    cout << "Test 4: " << solution.Journey_Moon_DFS(n4, pairs4) << endl;
}

int main() {
    Test_Journey_Moon_DFS();
    return 0;
}
