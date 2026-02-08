/*
Problem: Chocolate Breaking
URL: https://www.spoj.com/problems/CHOCOLA/

Problem Statement:
Break an M x N chocolate bar into 1x1 squares. Each horizontal/vertical cut has a cost. Cost of a cut = cost * number of segments being cut. Find minimum total cost.

Sample Input/Output:
Input: M=4, N=6, horizontal costs=[2,1,3,1,4], vertical costs=[4,1,2]
Output: 42
Explanation: Sort all cuts descending, greedily pick most expensive cuts first.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Min_Cost_To_Break_Chocolate(int M, int N, vector<int>& horizontal_costs, vector<int>& vertical_costs) {
        /*
        Sort all cuts descending, greedily pick most expensive
        Time Complexity: O((m+n) log(m+n))
        Space Complexity: O(1)
        */
        sort(horizontal_costs.rbegin(), horizontal_costs.rend());
        sort(vertical_costs.rbegin(), vertical_costs.rend());
        
        int h_pieces = 1;
        int v_pieces = 1;
        int h_idx = 0;
        int v_idx = 0;
        int total_cost = 0;
        
        while (h_idx < horizontal_costs.size() || v_idx < vertical_costs.size()) {
            if (h_idx < horizontal_costs.size() && 
                (v_idx >= vertical_costs.size() || horizontal_costs[h_idx] >= vertical_costs[v_idx])) {
                total_cost += horizontal_costs[h_idx] * v_pieces;
                h_pieces++;
                h_idx++;
            } else {
                total_cost += vertical_costs[v_idx] * h_pieces;
                v_pieces++;
                v_idx++;
            }
        }
        
        return total_cost;
    }
};

void Test_Chocolate_Breaking() {
    Solution solution;
    
    vector<int> h_costs1 = {2, 1, 3, 1, 4};
    vector<int> v_costs1 = {4, 1, 2};
    cout << "Test 1: " << solution.Min_Cost_To_Break_Chocolate(4, 6, h_costs1, v_costs1) << endl;
    
    vector<int> h_costs2 = {1, 1};
    vector<int> v_costs2 = {1};
    cout << "Test 2: " << solution.Min_Cost_To_Break_Chocolate(2, 2, h_costs2, v_costs2) << endl;
}

int main() {
    Test_Chocolate_Breaking();
    return 0;
}
