"""
Problem: Candy Store
URL: https://practice.geeksforgeeks.org/problems/shop-in-candy-store1145/1

Problem Statement:
In a candy store, there are N different types of candies available and the prices of all the N different types of candies are given to you. You are now provided with an attractive offer. For every candy you buy from the store and get K other candies (all are different types) for free. Find the minimum and maximum amount you have to spend to buy all the N different candies.

Sample Input/Output:
Input: N = 4, K = 2, candies[] = {3, 2, 1, 4}
Output: 3 7
Explanation: Minimum: Buy 1, get 2 free. Buy 1 more. Total = 3. Maximum: Buy 4, get 2 free. Buy 2, get 2 free. Total = 7.
"""


class Solution:
    def Candy_Store_Greedy_Both_Ends(self, N, K, candies):
        """
        Sort candies, buy from start for min (get free from end), buy from end for max (get free from start)
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        candies.sort()
        
        min_cost = 0
        max_cost = 0
        i = 0
        j = N - 1
        
        while i <= j:
            min_cost += candies[i]
            i += 1
            j -= K
        
        i = 0
        j = N - 1
        
        while i <= j:
            max_cost += candies[j]
            j -= 1
            i += K
        
        return [min_cost, max_cost]


def Test_Candy_Store():
    solution = Solution()
    N, K = 4, 2
    candies = [3, 2, 1, 4]
    result = solution.Candy_Store_Greedy_Both_Ends(N, K, candies)
    print(f"Minimum cost: {result[0]}, Maximum cost: {result[1]}")


if __name__ == "__main__":
    Test_Candy_Store()
