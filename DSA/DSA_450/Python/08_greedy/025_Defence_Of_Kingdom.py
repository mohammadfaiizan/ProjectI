"""
Problem: Defence Of Kingdom
URL: https://www.spoj.com/problems/DEFKIN/

Problem Statement:
Given a rectangular grid of W x H with some fortified cells, find the largest undefended rectangular area.

Sample Input/Output:
Input: W=15, H=8, fortified cells: [(3,8), (11,2), (8,6)]
Output: 12
Explanation: Add boundary 0 and W+1/H+1, sort coordinates, find max gap in x and y, multiply.
"""


class Solution:
    def Largest_Undefended_Area(self, W, H, fortified):
        """
        Sort + max gap approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        x_coords = [0, W + 1]
        y_coords = [0, H + 1]
        
        for p in fortified:
            x_coords.append(p[0])
            y_coords.append(p[1])
        
        x_coords.sort()
        y_coords.sort()
        
        max_x_gap = 0
        max_y_gap = 0
        
        for i in range(1, len(x_coords)):
            max_x_gap = max(max_x_gap, x_coords[i] - x_coords[i-1] - 1)
        
        for i in range(1, len(y_coords)):
            max_y_gap = max(max_y_gap, y_coords[i] - y_coords[i-1] - 1)
        
        return max_x_gap * max_y_gap


def Test_Defence_Of_Kingdom():
    solution = Solution()
    
    fortified1 = [(3, 8), (11, 2), (8, 6)]
    print(f"Test 1: {solution.Largest_Undefended_Area(15, 8, fortified1)}")
    
    fortified2 = [(2, 2)]
    print(f"Test 2: {solution.Largest_Undefended_Area(5, 5, fortified2)}")


if __name__ == "__main__":
    Test_Defence_Of_Kingdom()
