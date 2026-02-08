"""
Problem: Minimum Cash Flow
URL: https://www.geeksforgeeks.org/minimize-cash-flow-among-given-set-friends-borrowed-money/

Problem Statement:
Given a number of friends who have to give or take some amount of money from one another. Design an algorithm by which the total cash flow among all the friends is minimized.

Sample Input/Output:
Input: graph[][] = {{0, 1000, 2000}, {0, 0, 5000}, {0, 0, 0}}
Output: Person 1 pays 4000 to Person 2
        Person 0 pays 3000 to Person 2
Explanation: Net amounts: Person 0 = -3000, Person 1 = -4000, Person 2 = 7000. Settle by paying max creditor.
"""


class Solution:
    def Min_Cash_Flow_Net_Amount_Greedy(self, graph, n):
        """
        Calculate net amount for each person, greedily settle max creditor and debtor
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        net_amount = [0] * n
        
        for i in range(n):
            for j in range(n):
                net_amount[i] -= graph[i][j]
                net_amount[j] += graph[i][j]
        
        while True:
            max_creditor = -1
            max_debtor = -1
            max_credit = float('-inf')
            max_debt = float('inf')
            
            for i in range(n):
                if net_amount[i] > max_credit:
                    max_credit = net_amount[i]
                    max_creditor = i
                if net_amount[i] < max_debt:
                    max_debt = net_amount[i]
                    max_debtor = i
            
            if max_credit == 0 and max_debt == 0:
                break
            
            settle_amount = min(max_credit, -max_debt)
            net_amount[max_creditor] -= settle_amount
            net_amount[max_debtor] += settle_amount
            
            print(f"Person {max_debtor} pays {settle_amount} to Person {max_creditor}")


def Test_Min_Cash_Flow():
    solution = Solution()
    graph = [[0, 1000, 2000], [0, 0, 5000], [0, 0, 0]]
    n = 3
    solution.Min_Cash_Flow_Net_Amount_Greedy(graph, n)


if __name__ == "__main__":
    Test_Min_Cash_Flow()
