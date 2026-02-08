/*
Problem: Best Time to Buy and Sell Stock
URL: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

Problem Statement:
Given an array prices[] where prices[i] is the price of a stock on the ith day.
Maximize profit by choosing a single day to buy and a different day in the future to sell.
Return max profit, or 0 if no profit is possible.

Sample Input/Output:
Input: prices = [7, 1, 5, 3, 6, 4]
Output: 5
Explanation: Buy on day 2 (price=1), sell on day 5 (price=6), profit = 5.

Input: prices = [7, 6, 4, 3, 1]
Output: 0
Explanation: No profitable transaction possible.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Profit_Single_Pass_Optimal(vector<int>& prices) {
        /*
        Single Pass - Track minimum price and maximum profit
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int min_price = INT_MAX, max_profit = 0;
        for (int price : prices) {
            min_price = min(min_price, price);
            max_profit = max(max_profit, price - min_price);
        }
        return max_profit;
    }

    int Max_Profit_Kadane_Variant(vector<int>& prices) {
        /*
        Kadane's Variant - Max subarray on price differences
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int local = 0, global = 0;
        for (int i = 1; i < (int)prices.size(); i++) {
            local = max(0, local + prices[i] - prices[i - 1]);
            global = max(local, global);
        }
        return global;
    }

    int Max_Profit_Brute_Force(vector<int>& prices) {
        /*
        Brute Force - Check all pairs of buy and sell days
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        */
        int max_profit = 0;
        for (int i = 0; i < (int)prices.size(); i++)
            for (int j = i + 1; j < (int)prices.size(); j++)
                max_profit = max(max_profit, prices[j] - prices[i]);
        return max_profit;
    }
};

void Test_Best_Time_Buy_Sell_Stock() {
    Solution solution;

    struct TestCase {
        vector<int> prices;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{7, 1, 5, 3, 6, 4}, 5},
        {{7, 6, 4, 3, 1}, 0},
        {{2, 4, 1}, 2},
        {{1, 2}, 1}
    };

    for (auto& tc : test_cases) {
        cout << "Prices: ";
        for (int x : tc.prices) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        cout << "Single Pass: " << solution.Max_Profit_Single_Pass_Optimal(tc.prices) << endl;
        cout << "Kadane's Variant: " << solution.Max_Profit_Kadane_Variant(tc.prices) << endl;
        cout << "Brute Force: " << solution.Max_Profit_Brute_Force(tc.prices) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Best_Time_Buy_Sell_Stock();
    return 0;
}
