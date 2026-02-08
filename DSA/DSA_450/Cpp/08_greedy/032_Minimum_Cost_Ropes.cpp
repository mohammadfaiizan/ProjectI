/*
Problem: Minimum Cost of Ropes
URL: https://practice.geeksforgeeks.org/problems/minimum-cost-of-ropes-1587115620/1

Problem Statement:
Connect N ropes with minimum cost. Cost of connecting two ropes = sum of their lengths.

Sample Input/Output:
Input: [4,3,2,6]
Output: 29
Explanation: Min-heap approach minimizes total connection cost.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long Min_Cost_To_Connect_Ropes(vector<int>& ropes) {
        /*
        Min-heap approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        priority_queue<int, vector<int>, greater<int>> min_heap;
        
        for (int rope : ropes) {
            min_heap.push(rope);
        }
        
        long long total_cost = 0;
        
        while (min_heap.size() > 1) {
            int first = min_heap.top();
            min_heap.pop();
            int second = min_heap.top();
            min_heap.pop();
            
            int cost = first + second;
            total_cost += cost;
            min_heap.push(cost);
        }
        
        return total_cost;
    }
};

void Test_Minimum_Cost_Ropes() {
    Solution solution;
    
    vector<int> ropes1 = {4, 3, 2, 6};
    cout << "Test 1: " << solution.Min_Cost_To_Connect_Ropes(ropes1) << endl;
    
    vector<int> ropes2 = {4, 2, 7, 6, 9};
    cout << "Test 2: " << solution.Min_Cost_To_Connect_Ropes(ropes2) << endl;
    
    vector<int> ropes3 = {1, 2, 3};
    cout << "Test 3: " << solution.Min_Cost_To_Connect_Ropes(ropes3) << endl;
}

int main() {
    Test_Minimum_Cost_Ropes();
    return 0;
}
