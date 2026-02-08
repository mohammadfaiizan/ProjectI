"""
Problem: Chocolate Breaking
URL: https://www.spoj.com/problems/CHOCOLA/

Problem Statement:
Break an M x N chocolate bar into 1x1 squares. Each horizontal/vertical cut has a cost. Cost of a cut = cost * number of segments being cut. Find minimum total cost.

Sample Input/Output:
Input: M=4, N=6, horizontal costs=[2,1,3,1,4], vertical costs=[4,1,2]
Output: 42
Explanation: Sort all cuts descending, greedily pick most expensive cuts first.
"""


class Solution:
    def Min_Cost_To_Break_Chocolate(self, M, N, horizontal_costs, vertical_costs):
        """
        Sort all cuts descending, greedily pick most expensive
        Time Complexity: O((m+n) log(m+n))
        Space Complexity: O(1)
        """
        horizontal_costs.sort(reverse=True)
        vertical_costs.sort(reverse=True)
        
        h_pieces = 1
        v_pieces = 1
        h_idx = 0
        v_idx = 0
        total_cost = 0
        
        while h_idx < len(horizontal_costs) or v_idx < len(vertical_costs):
            if h_idx < len(horizontal_costs) and \
               (v_idx >= len(vertical_costs) or horizontal_costs[h_idx] >= vertical_costs[v_idx]):
                total_cost += horizontal_costs[h_idx] * v_pieces
                h_pieces += 1
                h_idx += 1
            else:
                total_cost += vertical_costs[v_idx] * h_pieces
                v_pieces += 1
                v_idx += 1
        
        return total_cost


def Test_Chocolate_Breaking():
    solution = Solution()
    
    h_costs1 = [2, 1, 3, 1, 4]
    v_costs1 = [4, 1, 2]
    print(f"Test 1: {solution.Min_Cost_To_Break_Chocolate(4, 6, h_costs1, v_costs1)}")
    
    h_costs2 = [1, 1]
    v_costs2 = [1]
    print(f"Test 2: {solution.Min_Cost_To_Break_Chocolate(2, 2, h_costs2, v_costs2)}")


if __name__ == "__main__":
    Test_Chocolate_Breaking()
