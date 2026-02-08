/*
Problem: Minimum Cash Flow
URL: https://www.geeksforgeeks.org/minimize-cash-flow-among-given-set-friends-borrowed-money/

Problem Statement:
Given a number of friends who have to give or take some amount of money from one another. Design an algorithm by which the total cash flow among all the friends is minimized.

Sample Input/Output:
Input: graph[][] = {{0, 1000, 2000}, {0, 0, 5000}, {0, 0, 0}}
Output: Person 1 pays 4000 to Person 2
        Person 0 pays 3000 to Person 2
Explanation: Net amounts: Person 0 = -3000, Person 1 = -4000, Person 2 = 7000. Settle by paying max creditor.
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    void Min_Cash_Flow_Net_Amount_Greedy(vector<vector<int>>& graph, int n) {
        /*
        Calculate net amount for each person, greedily settle max creditor and debtor
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        */
        vector<int> net_amount(n, 0);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                net_amount[i] -= graph[i][j];
                net_amount[j] += graph[i][j];
            }
        }
        
        while (true) {
            int max_creditor = -1;
            int max_debtor = -1;
            int max_credit = INT_MIN;
            int max_debt = INT_MAX;
            
            for (int i = 0; i < n; i++) {
                if (net_amount[i] > max_credit) {
                    max_credit = net_amount[i];
                    max_creditor = i;
                }
                if (net_amount[i] < max_debt) {
                    max_debt = net_amount[i];
                    max_debtor = i;
                }
            }
            
            if (max_credit == 0 && max_debt == 0) break;
            
            int settle_amount = min(max_credit, -max_debt);
            net_amount[max_creditor] -= settle_amount;
            net_amount[max_debtor] += settle_amount;
            
            cout << "Person " << max_debtor << " pays " << settle_amount << " to Person " << max_creditor << endl;
        }
    }
};

void Test_Min_Cash_Flow() {
    Solution solution;
    vector<vector<int>> graph = {{0, 1000, 2000}, {0, 0, 5000}, {0, 0, 0}};
    int n = 3;
    solution.Min_Cash_Flow_Net_Amount_Greedy(graph, n);
}

int main() {
    Test_Min_Cash_Flow();
    return 0;
}
