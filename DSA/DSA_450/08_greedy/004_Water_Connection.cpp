/*
Problem: Water Connection
URL: https://practice.geeksforgeeks.org/problems/water-connection-problem5822/1

Problem Statement:
There are n houses and p water pipes in Geek Colony. Each house has at most one pipe going into it and at most one pipe going out of it. Geek Colony needs to install water tanks and taps in the colony. Houses with no incoming pipe get a water tank and houses with no outgoing pipe get a tap. Find the source houses (with tanks), destination houses (with taps), and minimum diameter along each path.

Sample Input/Output:
Input: n = 9, p = 6, a[] = {7,5,4,2,9,3}, b[] = {4,9,6,8,7,1}, d[] = {98,72,10,22,17,66}
Output: 3
        2 8 22
        3 1 66
        5 6 10
Explanation: Three paths: 2->8 (diameter 22), 3->1 (diameter 66), 5->6 (diameter 10)
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> Solve_Water_Connection_DFS(int n, int p, vector<int>& a, vector<int>& b, vector<int>& d) {
        /*
        DFS traversal to find source-destination paths with minimum diameter
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        vector<int> in_degree(n + 1, 0);
        vector<int> out_degree(n + 1, 0);
        vector<pair<int, int>> graph(n + 1, {-1, -1});
        
        for (int i = 0; i < p; i++) {
            graph[a[i]] = {b[i], d[i]};
            in_degree[b[i]]++;
            out_degree[a[i]]++;
        }
        
        vector<vector<int>> result;
        
        for (int i = 1; i <= n; i++) {
            if (in_degree[i] == 0 && out_degree[i] > 0) {
                int start = i;
                int end_node = i;
                int min_diameter = INT_MAX;
                
                while (graph[end_node].first != -1) {
                    min_diameter = min(min_diameter, graph[end_node].second);
                    end_node = graph[end_node].first;
                }
                
                if (end_node != start) {
                    result.push_back({start, end_node, min_diameter});
                }
            }
        }
        
        return result;
    }
};

void Test_Water_Connection() {
    Solution solution;
    int n = 9, p = 6;
    vector<int> a = {7, 5, 4, 2, 9, 3};
    vector<int> b = {4, 9, 6, 8, 7, 1};
    vector<int> d = {98, 72, 10, 22, 17, 66};
    vector<vector<int>> result = solution.Solve_Water_Connection_DFS(n, p, a, b, d);
    cout << "Number of paths: " << result.size() << endl;
    for (auto& path : result) {
        cout << path[0] << " " << path[1] << " " << path[2] << endl;
    }
}

int main() {
    Test_Water_Connection();
    return 0;
}
