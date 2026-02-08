"""
Problem: Water Jug Problem
URL: https://www.geeksforgeeks.org/water-jug-problem-using-bfs/

Problem Statement:
Given two jugs of capacities a and b, measure exactly d liters using BFS. You can fill, empty, or pour water between jugs.

Sample Input/Output:
Input: jug1=4, jug2=3, target=2
Output: true (can measure 2 liters)
"""

from collections import deque


class Solution:
    def Water_Jug_BFS(self, a, b, d):
        """
        BFS with state (x,y) and all 6 operations
        Time Complexity: O(a*b)
        Space Complexity: O(a*b)
        """
        if d > max(a, b) or d < 0:
            return False
        
        if d == 0:
            return True
        
        visited = set()
        q = deque()
        q.append((0, 0))
        visited.add((0, 0))
        
        while q:
            x, y = q.popleft()
            
            if x == d or y == d or x + y == d:
                return True
            
            nextStates = [
                (a, y),
                (x, b),
                (0, y),
                (x, 0),
                (x - min(x, b - y), y + min(x, b - y)),
                (x + min(y, a - x), y - min(y, a - x))
            ]
            
            for nx, ny in nextStates:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        
        return False


def Test_Water_Jug_BFS():
    solution = Solution()
    
    print(f"Test 1 (4,3,2): {solution.Water_Jug_BFS(4, 3, 2)}")
    print(f"Test 2 (5,3,4): {solution.Water_Jug_BFS(5, 3, 4)}")
    print(f"Test 3 (3,5,4): {solution.Water_Jug_BFS(3, 5, 4)}")
    print(f"Test 4 (8,5,3): {solution.Water_Jug_BFS(8, 5, 3)}")
    print(f"Test 5 (2,6,5): {solution.Water_Jug_BFS(2, 6, 5)}")


if __name__ == "__main__":
    Test_Water_Jug_BFS()
