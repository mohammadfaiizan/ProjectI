/*
Problem: Fractional Knapsack
URL: https://practice.geeksforgeeks.org/problems/fractional-knapsack-1587115620/1

Problem Statement:
Given weights and values of N items, put these items in a knapsack of capacity W to get the maximum total value. Items can be broken into fractions.

Sample Input/Output:
Input: N = 3, W = 50, values[] = {60,100,120}, weight[] = {10,20,30}
Output: 240.00
Explanation: Take items of weight 10 and 20 kg and 2/3rd of the item with weight 30 kg. Total value = 60 + 100 + 120*(2/3) = 240.
*/

#include <bits/stdc++.h>
using namespace std;

struct Item {
    int value;
    int weight;
};

class Solution {
public:
    double Fractional_Knapsack_Greedy(int W, Item arr[], int n) {
        /*
        Sort by value/weight ratio descending, greedily take items
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        */
        sort(arr, arr + n, [](Item a, Item b) {
            return (double)a.value / a.weight > (double)b.value / b.weight;
        });
        
        double total_value = 0.0;
        int remaining = W;
        
        for (int i = 0; i < n && remaining > 0; i++) {
            if (arr[i].weight <= remaining) {
                total_value += arr[i].value;
                remaining -= arr[i].weight;
            } else {
                total_value += (double)arr[i].value * remaining / arr[i].weight;
                remaining = 0;
            }
        }
        
        return total_value;
    }
};

void Test_Fractional_Knapsack() {
    Solution solution;
    int W = 50;
    Item arr[] = {{60, 10}, {100, 20}, {120, 30}};
    int n = 3;
    cout << "Maximum value: " << fixed << setprecision(2) << solution.Fractional_Knapsack_Greedy(W, arr, n) << endl;
}

int main() {
    Test_Fractional_Knapsack();
    return 0;
}
