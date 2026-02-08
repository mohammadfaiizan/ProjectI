/*
Problem: K Centers Problem
URL: https://www.geeksforgeeks.org/k-centers-problem-set-1-greedy-approximate-algorithm/

Problem Statement:
Given N cities with distances, select K centers to minimize max distance from any city to nearest center (2-approximation).

Sample Input/Output:
Input: N=4, K=2, distances matrix
Output: Centers selected and max distance
Explanation: Greedy farthest-first traversal selects optimal centers.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    pair<vector<int>, int> Select_K_Centers(int N, int K, vector<vector<int>>& distances) {
        /*
        Greedy farthest-first traversal approach
        Time Complexity: O(n*k)
        Space Complexity: O(n)
        */
        vector<int> centers;
        vector<int> min_dist(N, INT_MAX);
        
        centers.push_back(0);
        
        for (int i = 0; i < N; i++) {
            min_dist[i] = distances[0][i];
        }
        
        for (int k = 1; k < K; k++) {
            int farthest_city = -1;
            int max_dist = 0;
            
            for (int i = 0; i < N; i++) {
                if (min_dist[i] > max_dist) {
                    max_dist = min_dist[i];
                    farthest_city = i;
                }
            }
            
            if (farthest_city == -1) break;
            
            centers.push_back(farthest_city);
            
            for (int i = 0; i < N; i++) {
                min_dist[i] = min(min_dist[i], distances[farthest_city][i]);
            }
        }
        
        int max_min_dist = 0;
        for (int i = 0; i < N; i++) {
            max_min_dist = max(max_min_dist, min_dist[i]);
        }
        
        return {centers, max_min_dist};
    }
};

void Test_K_Centers() {
    Solution solution;
    
    vector<vector<int>> dist1 = {
        {0, 10, 7, 6},
        {10, 0, 8, 5},
        {7, 8, 0, 12},
        {6, 5, 12, 0}
    };
    
    auto result1 = solution.Select_K_Centers(4, 2, dist1);
    cout << "Test 1 - Centers: ";
    for (int c : result1.first) cout << c << " ";
    cout << ", Max Distance: " << result1.second << endl;
    
    vector<vector<int>> dist2 = {
        {0, 1, 2},
        {1, 0, 3},
        {2, 3, 0}
    };
    
    auto result2 = solution.Select_K_Centers(3, 2, dist2);
    cout << "Test 2 - Centers: ";
    for (int c : result2.first) cout << c << " ";
    cout << ", Max Distance: " << result2.second << endl;
}

int main() {
    Test_K_Centers();
    return 0;
}
