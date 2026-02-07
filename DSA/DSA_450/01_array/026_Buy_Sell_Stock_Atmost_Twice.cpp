/*
Problem: Buy and Sell Stock at Most Twice
URL: https://www.geeksforgeeks.org/maximum-profit-by-buying-and-selling-a-share-at-most-twice/

Problem Statement:
Given an array price[] where price[i] is the stock price on day i, find the maximum profit
achievable by buying and selling at most twice. You must sell before buying again.

Sample Input/Output:
Input: prices = [2, 30, 15, 10, 8, 25, 80]
Output: 100
Explanation: Buy at 2, sell at 30 (profit 28). Buy at 8, sell at 80 (profit 72). Total = 100.

Input: prices = [100, 30, 15, 10, 8, 25, 80]
Output: 72
Explanation: Buy at 8, sell at 80. Only one transaction needed.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Max_Profit_Two_Pass_DP_Optimal(vector<int>& prices) {
        /*
        Two Pass DP - Forward pass for first transaction, backward for second
        Time Complexity: O(n)
        Space Complexity: O(n)
        */
        int n = prices.size();
        if (n < 2) return 0;
        vector<int> profit(n, 0);
        int max_price = prices[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            max_price = max(max_price, prices[i]);
            profit[i] = max(profit[i + 1], max_price - prices[i]);
        }
        int min_price = prices[0];
        for (int i = 1; i < n; i++) {
            min_price = min(min_price, prices[i]);
            profit[i] = max(profit[i - 1], profit[i] + (prices[i] - min_price));
        }
        return profit[n - 1];
    }

    int Max_Profit_State_Machine(vector<int>& prices) {
        /*
        State Machine - Track 4 states (buy1, sell1, buy2, sell2)
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int buy1 = INT_MIN, sell1 = 0;
        int buy2 = INT_MIN, sell2 = 0;
        for (int price : prices) {
            buy1 = max(buy1, -price);
            sell1 = max(sell1, buy1 + price);
            buy2 = max(buy2, sell1 - price);
            sell2 = max(sell2, buy2 + price);
        }
        return sell2;
    }

    int Max_Profit_Valley_Peak_Unlimited(vector<int>& prices) {
        /*
        Valley Peak (Unlimited Transactions) - Sum all positive differences
        Time Complexity: O(n)
        Space Complexity: O(1)
        */
        int profit = 0;
        for (int i = 1; i < (int)prices.size(); i++) {
            int diff = prices[i] - prices[i - 1];
            if (diff > 0) profit += diff;
        }
        return profit;
    }
};

void Test_Buy_Sell_Stock_Twice() {
    Solution solution;

    struct TestCase {
        vector<int> prices;
        int expected;
    };

    vector<TestCase> test_cases = {
        {{2, 30, 15, 10, 8, 25, 80}, 100},
        {{100, 30, 15, 10, 8, 25, 80}, 72},
        {{10, 22, 5, 75, 65, 80}, 87},
        {{1, 2, 3, 4, 5}, 4}
    };

    for (auto& tc : test_cases) {
        cout << "Prices: ";
        for (int x : tc.prices) cout << x << " ";
        cout << ", Expected: " << tc.expected << endl;

        cout << "Two Pass DP: " << solution.Max_Profit_Two_Pass_DP_Optimal(tc.prices) << endl;
        cout << "State Machine: " << solution.Max_Profit_State_Machine(tc.prices) << endl;
        cout << "Valley Peak (unlimited): " << solution.Max_Profit_Valley_Peak_Unlimited(tc.prices) << endl;

        cout << string(50, '-') << endl;
    }
}

int main() {
    Test_Buy_Sell_Stock_Twice();
    return 0;
}
