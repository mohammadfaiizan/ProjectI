/*
Problem: Minimize Cash Flow Among Friends
URL: https://www.geeksforgeeks.org/minimize-cash-flow-among-given-set-friends-borrowed-money/

Problem Statement:
Given a graph of debts among friends, minimize the number of transactions to settle all debts.

Sample Input/Output:
Input: Number of friends and debts
Output: Minimum transactions needed
*/

#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int Minimize_Cash_Flow_Greedy(int N, vector<vector<int>>& graph) {
        /*
        Compute net amounts, repeatedly settle between max creditor and max debtor
        Time Complexity: O(N^2)
        Space Complexity: O(N)
        */
        vector<int> netAmount(N, 0);
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                netAmount[i] += graph[j][i] - graph[i][j];
            }
        }
        
        int transactions = 0;
        
        while (true) {
            int maxCreditor = -1, maxDebtor = -1;
            int maxCredit = INT_MIN, maxDebt = INT_MAX;
            
            for (int i = 0; i < N; i++) {
                if (netAmount[i] > maxCredit) {
                    maxCredit = netAmount[i];
                    maxCreditor = i;
                }
                if (netAmount[i] < maxDebt) {
                    maxDebt = netAmount[i];
                    maxDebtor = i;
                }
            }
            
            if (maxCredit == 0 && maxDebt == 0) break;
            
            int settleAmount = min(maxCredit, -maxDebt);
            netAmount[maxCreditor] -= settleAmount;
            netAmount[maxDebtor] += settleAmount;
            transactions++;
        }
        
        return transactions;
    }
};

void Test_Minimize_Cash_Flow() {
    Solution solution;
    
    cout << "Test Case 1: 3 Friends" << endl;
    int N1 = 3;
    vector<vector<int>> graph1 = {
        {0, 1000, 2000},
        {0, 0, 5000},
        {0, 0, 0}
    };
    int result1 = solution.Minimize_Cash_Flow_Greedy(N1, graph1);
    cout << "Minimum transactions: " << result1 << endl;
    cout << endl;
    
    cout << "Test Case 2: 4 Friends" << endl;
    int N2 = 4;
    vector<vector<int>> graph2 = {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    };
    graph2[0][1] = 1000;
    graph2[0][2] = 2000;
    graph2[1][2] = 5000;
    graph2[2][3] = 3000;
    int result2 = solution.Minimize_Cash_Flow_Greedy(N2, graph2);
    cout << "Minimum transactions: " << result2 << endl;
    cout << endl;
    
    cout << "Test Case 3: Simple Case" << endl;
    int N3 = 3;
    vector<vector<int>> graph3 = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };
    graph3[0][1] = 100;
    graph3[1][2] = 100;
    int result3 = solution.Minimize_Cash_Flow_Greedy(N3, graph3);
    cout << "Minimum transactions: " << result3 << endl;
}

int main() {
    Test_Minimize_Cash_Flow();
    return 0;
}
