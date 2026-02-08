/*
Problem: Travelling Salesman Problem
URL: https://www.geeksforgeeks.org/travelling-salesman-problem-using-dynamic-programming-solution/

Problem Statement:
Find the shortest route that visits all cities exactly once and returns to the starting city. Given a distance matrix representing distances between cities.

Sample Input/Output:
Input: 4 cities, distance matrix
Output: Minimum cost to visit all cities and return
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int TSP_DP_Bitmask(vector<vector<int>>& dist) {
        /*
        DP with bitmask
        Time Complexity: O(2^n * n^2)
        Space Complexity: O(2^n * n)
        */
        int n = dist.size();
        int totalStates = 1 << n;
        vector<vector<int>> dp(totalStates, vector<int>(n, INT_MAX));
        
        dp[1][0] = 0;
        
        for (int mask = 1; mask < totalStates; mask++) {
            for (int u = 0; u < n; u++) {
                if (!(mask & (1 << u))) continue;
                if (dp[mask][u] == INT_MAX) continue;
                
                for (int v = 0; v < n; v++) {
                    if (mask & (1 << v)) continue;
                    int newMask = mask | (1 << v);
                    if (dist[u][v] > 0) {
                        dp[newMask][v] = min(dp[newMask][v], dp[mask][u] + dist[u][v]);
                    }
                }
            }
        }
        
        int finalMask = totalStates - 1;
        int result = INT_MAX;
        for (int u = 1; u < n; u++) {
            if (dist[u][0] > 0 && dp[finalMask][u] != INT_MAX) {
                result = min(result, dp[finalMask][u] + dist[u][0]);
            }
        }
        
        return result;
    }
    
    int TSP_Brute_Force(vector<vector<int>>& dist) {
        /*
        Permutation-based
        Time Complexity: O(n!)
        Space Complexity: O(n)
        */
        int n = dist.size();
        vector<int> cities;
        for (int i = 1; i < n; i++) {
            cities.push_back(i);
        }
        
        int minCost = INT_MAX;
        do {
            int cost = dist[0][cities[0]];
            for (int i = 0; i < cities.size() - 1; i++) {
                cost += dist[cities[i]][cities[i + 1]];
            }
            cost += dist[cities.back()][0];
            minCost = min(minCost, cost);
        } while (next_permutation(cities.begin(), cities.end()));
        
        return minCost;
    }
};

void Test_TSP() {
    Solution solution;
    
    cout << "Test Case 1: 4 cities distance matrix" << endl;
    vector<vector<int>> dist1 = {
        {0, 10, 15, 20},
        {10, 0, 35, 25},
        {15, 35, 0, 30},
        {20, 25, 30, 0}
    };
    cout << "DP Bitmask Result: " << solution.TSP_DP_Bitmask(dist1) << endl;
    cout << "Brute Force Result: " << solution.TSP_Brute_Force(dist1) << endl;
    
    cout << "\nTest Case 2: 3 cities" << endl;
    vector<vector<int>> dist2 = {
        {0, 1, 2},
        {1, 0, 3},
        {2, 3, 0}
    };
    cout << "DP Bitmask Result: " << solution.TSP_DP_Bitmask(dist2) << endl;
    cout << "Brute Force Result: " << solution.TSP_Brute_Force(dist2) << endl;
    
    cout << "\nTest Case 3: 5 cities" << endl;
    vector<vector<int>> dist3 = {
        {0, 2, 9, 10, 7},
        {2, 0, 6, 4, 3},
        {9, 6, 0, 8, 5},
        {10, 4, 8, 0, 1},
        {7, 3, 5, 1, 0}
    };
    cout << "DP Bitmask Result: " << solution.TSP_DP_Bitmask(dist3) << endl;
}

int main() {
    Test_TSP();
    return 0;
}
