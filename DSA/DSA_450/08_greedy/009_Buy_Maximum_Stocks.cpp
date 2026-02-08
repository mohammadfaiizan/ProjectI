/*
Problem: Buy Maximum Stocks
URL: https://www.geeksforgeeks.org/buy-maximum-stocks-stocks-can-bought-th-day/

Problem Statement:
In a stock market, there is a product with its infinite stocks. The stock prices are given for N days, where arr[i] denotes the price of the stock on the ith day. There is a rule that a customer can buy at most i stock on the ith day. If the customer has an amount of k amount of money initially, find out the maximum number of stocks a customer can buy.

Sample Input/Output:
Input: price[] = {10, 7, 19}, k = 45
Output: 4
Explanation: Day 1: Buy 1 stock at 10, Day 2: Buy 2 stocks at 7 each, Day 3: Buy 1 stock at 19. Total = 1 + 2 + 1 = 4 stocks.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Buy_Maximum_Stocks_Sort_Greedy(vector<int>& price, int n, int k) {
        /*
        Sort by price, greedily buy maximum stocks at lowest prices
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        */
        vector<pair<int, int>> stocks;
        for (int i = 0; i < n; i++) {
            stocks.push_back({price[i], i + 1});
        }
        
        sort(stocks.begin(), stocks.end());
        
        int stocks_bought = 0;
        int remaining = k;
        
        for (auto& stock : stocks) {
            int max_buy = min(stock.second, remaining / stock.first);
            stocks_bought += max_buy;
            remaining -= max_buy * stock.first;
            if (remaining < stock.first) break;
        }
        
        return stocks_bought;
    }
};

void Test_Buy_Maximum_Stocks() {
    Solution solution;
    vector<int> price = {10, 7, 19};
    int k = 45;
    cout << "Maximum stocks: " << solution.Buy_Maximum_Stocks_Sort_Greedy(price, price.size(), k) << endl;
}

int main() {
    Test_Buy_Maximum_Stocks();
    return 0;
}
