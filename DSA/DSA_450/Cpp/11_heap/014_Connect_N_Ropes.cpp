/*
Problem: Connect N Ropes with Minimum Cost
URL: https://practice.geeksforgeeks.org/problems/minimum-cost-of-ropes-1587115620/1

Problem Statement:
Given N ropes of different lengths, connect them into one rope with minimum total cost. Cost to connect two ropes = sum of their lengths.

Sample Input/Output:
Input: [4,3,2,6]
Output: 29
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Connect_Ropes_Min_Heap(vector<int>& ropes) {
        /*
        Min Heap Approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        priority_queue<int, vector<int>, greater<int>> pq;
        for (int rope : ropes) {
            pq.push(rope);
        }
        
        int totalCost = 0;
        while (pq.size() > 1) {
            int first = pq.top();
            pq.pop();
            int second = pq.top();
            pq.pop();
            int cost = first + second;
            totalCost += cost;
            pq.push(cost);
        }
        
        return totalCost;
    }
    
    int Connect_Ropes_Sort(vector<int>& ropes) {
        /*
        Sort Approach
        Time Complexity: O(n^2 log n)
        Space Complexity: O(1)
        */
        vector<int> arr = ropes;
        int totalCost = 0;
        
        while (arr.size() > 1) {
            sort(arr.begin(), arr.end());
            int first = arr[0];
            int second = arr[1];
            int cost = first + second;
            totalCost += cost;
            arr.erase(arr.begin(), arr.begin() + 2);
            arr.push_back(cost);
        }
        
        return totalCost;
    }
};

void Test_Connect_Ropes() {
    Solution solution;
    
    vector<int> ropes1 = {4, 3, 2, 6};
    cout << "Test 1 Min Heap: " << solution.Connect_Ropes_Min_Heap(ropes1) << endl;
    vector<int> ropes1b = {4, 3, 2, 6};
    cout << "Test 1 Sort: " << solution.Connect_Ropes_Sort(ropes1b) << endl;
    
    vector<int> ropes2 = {1, 2, 3, 4, 5};
    cout << "Test 2 Min Heap: " << solution.Connect_Ropes_Min_Heap(ropes2) << endl;
    vector<int> ropes2b = {1, 2, 3, 4, 5};
    cout << "Test 2 Sort: " << solution.Connect_Ropes_Sort(ropes2b) << endl;
    
    vector<int> ropes3 = {5, 4, 2, 8};
    cout << "Test 3 Min Heap: " << solution.Connect_Ropes_Min_Heap(ropes3) << endl;
    vector<int> ropes3b = {5, 4, 2, 8};
    cout << "Test 3 Sort: " << solution.Connect_Ropes_Sort(ropes3b) << endl;
}

int main() {
    Test_Connect_Ropes();
    return 0;
}
