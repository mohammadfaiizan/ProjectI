"""
Problem: Fractional Knapsack
URL: https://practice.geeksforgeeks.org/problems/fractional-knapsack-1587115620/1

Problem Statement:
Given weights and values of N items, put these items in a knapsack of capacity W to get the maximum total value. Items can be broken into fractions.

Sample Input/Output:
Input: N = 3, W = 50, values[] = {60,100,120}, weight[] = {10,20,30}
Output: 240.00
Explanation: Take items of weight 10 and 20 kg and 2/3rd of the item with weight 30 kg. Total value = 60 + 100 + 120*(2/3) = 240.
"""


class Solution:
    def Fractional_Knapsack_Greedy(self, W, arr, n):
        """
        Sort by value/weight ratio descending, greedily take items
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr.sort(key=lambda x: x.value / x.weight, reverse=True)
        
        total_value = 0.0
        remaining = W
        
        for i in range(n):
            if remaining <= 0:
                break
            if arr[i].weight <= remaining:
                total_value += arr[i].value
                remaining -= arr[i].weight
            else:
                total_value += arr[i].value * remaining / arr[i].weight
                remaining = 0
        
        return total_value


class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight


def Test_Fractional_Knapsack():
    solution = Solution()
    W = 50
    arr = [Item(60, 10), Item(100, 20), Item(120, 30)]
    n = 3
    print(f"Maximum value: {solution.Fractional_Knapsack_Greedy(W, arr, n):.2f}")


if __name__ == "__main__":
    Test_Fractional_Knapsack()
