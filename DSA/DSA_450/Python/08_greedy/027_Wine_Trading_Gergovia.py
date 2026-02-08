"""
Problem: Wine Trading in Gergovia
URL: https://www.spoj.com/problems/GERGOVIA/

Problem Statement:
N houses in a row, each buys or sells wine. Transport cost is 1 per unit per house. Find minimum total transport cost.

Sample Input/Output:
Input: [5, -4, 1, -3, 1]
Output: 9
Explanation: Prefix sum / greedy matching approach minimizes transport cost.
"""


class Solution:
    def Min_Transport_Cost(self, houses):
        """
        Prefix sum / greedy matching approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        cost = 0
        prefix_sum = 0
        
        for i in range(len(houses)):
            prefix_sum += houses[i]
            cost += abs(prefix_sum)
        
        return cost


def Test_Wine_Trading_Gergovia():
    solution = Solution()
    
    houses1 = [5, -4, 1, -3, 1]
    print(f"Test 1: {solution.Min_Transport_Cost(houses1)}")
    
    houses2 = [-1000, -1000, -1000, 1000, 1000, 1000]
    print(f"Test 2: {solution.Min_Transport_Cost(houses2)}")
    
    houses3 = [1, -1]
    print(f"Test 3: {solution.Min_Transport_Cost(houses3)}")


if __name__ == "__main__":
    Test_Wine_Trading_Gergovia()
