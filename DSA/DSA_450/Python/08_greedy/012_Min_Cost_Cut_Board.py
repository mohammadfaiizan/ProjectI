"""
Problem: Minimum Cost to Cut Board into Squares
URL: https://www.geeksforgeeks.org/minimum-cost-cut-board-squares/

Problem Statement:
A board of length m and width n is given, the task is to break this board into m*n squares such that cost of breaking is minimum. The cutting cost for each edge will be given. In short, we need to choose such a sequence of cutting such that cost is minimized.

Sample Input/Output:
Input: m = 6, n = 4, X[] = {2, 1, 3, 1, 4}, Y[] = {4, 1, 2}
Output: 42
Explanation: Cut horizontally first (cost 4*6=24), then vertically (cost 1*4+2*4+1*4+3*4=28). Total = 52. Better: Cut vertical first.
"""


class Solution:
    def Min_Cost_Cut_Board_Sort_Greedy(self, m, n, X, Y):
        """
        Sort cuts in descending order, greedily make expensive cuts first when more pieces exist
        Time Complexity: O((m+n) log(m+n))
        Space Complexity: O(1)
        """
        X.sort(reverse=True)
        Y.sort(reverse=True)
        
        horizontal_pieces = 1
        vertical_pieces = 1
        cost = 0
        i = 0
        j = 0
        
        while i < len(X) and j < len(Y):
            if X[i] > Y[j]:
                cost += X[i] * vertical_pieces
                horizontal_pieces += 1
                i += 1
            else:
                cost += Y[j] * horizontal_pieces
                vertical_pieces += 1
                j += 1
        
        while i < len(X):
            cost += X[i] * vertical_pieces
            i += 1
        
        while j < len(Y):
            cost += Y[j] * horizontal_pieces
            j += 1
        
        return cost


def Test_Min_Cost_Cut_Board():
    solution = Solution()
    m, n = 6, 4
    X = [2, 1, 3, 1, 4]
    Y = [4, 1, 2]
    print(f"Minimum cost: {solution.Min_Cost_Cut_Board_Sort_Greedy(m, n, X, Y)}")


if __name__ == "__main__":
    Test_Min_Cost_Cut_Board()
