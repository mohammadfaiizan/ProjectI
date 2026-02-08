/*
Problem: Wine Trading in Gergovia
URL: https://www.spoj.com/problems/GERGOVIA/

Problem Statement:
N houses in a row, each buys or sells wine. Transport cost is 1 per unit per house. Find minimum total transport cost.

Sample Input/Output:
Input: [5, -4, 1, -3, 1]
Output: 9
Explanation: Prefix sum / greedy matching approach minimizes transport cost.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Min_Transport_Cost(vector<int>& houses) {
        /*
        Prefix sum / greedy matching approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        long long cost = 0;
        long long prefix_sum = 0;
        
        for (int i = 0; i < houses.size(); i++) {
            prefix_sum += houses[i];
            cost += abs(prefix_sum);
        }
        
        return cost;
    }
};

void Test_Wine_Trading_Gergovia() {
    Solution solution;
    
    vector<int> houses1 = {5, -4, 1, -3, 1};
    cout << "Test 1: " << solution.Min_Transport_Cost(houses1) << endl;
    
    vector<int> houses2 = {-1000, -1000, -1000, 1000, 1000, 1000};
    cout << "Test 2: " << solution.Min_Transport_Cost(houses2) << endl;
    
    vector<int> houses3 = {1, -1};
    cout << "Test 3: " << solution.Min_Transport_Cost(houses3) << endl;
}

int main() {
    Test_Wine_Trading_Gergovia();
    return 0;
}
