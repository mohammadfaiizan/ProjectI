"""
Problem: Minimize Cash Flow Among Friends
URL: https://www.geeksforgeeks.org/minimize-cash-flow-among-given-set-friends-borrowed-money/

Problem Statement:
Given a graph of debts among friends, minimize the number of transactions to settle all debts.

Sample Input/Output:
Input: Number of friends and debts
Output: Minimum transactions needed
"""


class Solution:
    def Minimize_Cash_Flow_Greedy(self, N, graph):
        """
        Compute net amounts, repeatedly settle between max creditor and max debtor
        Time Complexity: O(N^2)
        Space Complexity: O(N)
        """
        netAmount = [0] * N
        
        for i in range(N):
            for j in range(N):
                netAmount[i] += graph[j][i] - graph[i][j]
        
        transactions = 0
        
        while True:
            maxCreditor = -1
            maxDebtor = -1
            maxCredit = float('-inf')
            maxDebt = float('inf')
            
            for i in range(N):
                if netAmount[i] > maxCredit:
                    maxCredit = netAmount[i]
                    maxCreditor = i
                if netAmount[i] < maxDebt:
                    maxDebt = netAmount[i]
                    maxDebtor = i
            
            if maxCredit == 0 and maxDebt == 0:
                break
            
            settleAmount = min(maxCredit, -maxDebt)
            netAmount[maxCreditor] -= settleAmount
            netAmount[maxDebtor] += settleAmount
            transactions += 1
        
        return transactions


def Test_Minimize_Cash_Flow():
    solution = Solution()
    
    print("Test Case 1: 3 Friends")
    N1 = 3
    graph1 = [
        [0, 1000, 2000],
        [0, 0, 5000],
        [0, 0, 0]
    ]
    result1 = solution.Minimize_Cash_Flow_Greedy(N1, graph1)
    print(f"Minimum transactions: {result1}")
    print()
    
    print("Test Case 2: 4 Friends")
    N2 = 4
    graph2 = [[0] * N2 for _ in range(N2)]
    graph2[0][1] = 1000
    graph2[0][2] = 2000
    graph2[1][2] = 5000
    graph2[2][3] = 3000
    result2 = solution.Minimize_Cash_Flow_Greedy(N2, graph2)
    print(f"Minimum transactions: {result2}")
    print()
    
    print("Test Case 3: Simple Case")
    N3 = 3
    graph3 = [[0] * N3 for _ in range(N3)]
    graph3[0][1] = 100
    graph3[1][2] = 100
    result3 = solution.Minimize_Cash_Flow_Greedy(N3, graph3)
    print(f"Minimum transactions: {result3}")


if __name__ == "__main__":
    Test_Minimize_Cash_Flow()
