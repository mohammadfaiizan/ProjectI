"""
Problem: Optimum Location of Point to Minimize Total Distance
URL: https://www.geeksforgeeks.org/optimum-location-point-minimize-total-distance/

Problem Statement:
Given a set of points, find the point on a line that minimizes the total distance to all given points. The line is given by ax + by + c = 0.

Sample Input/Output:
Input: points = [(0, 0), (2, 0), (3, 0), (5, 0)], line: y = 0 (a=0, b=1, c=0)
Output: (2.5, 0)

Input: points = [(1, 1), (2, 2), (3, 3)], line: x - y = 0 (a=1, b=-1, c=0)
Output: (2, 2)
"""

import math


class Solution:
    def Optimum_Location_Ternary_Search(self, points, a, b, c):
        """
        Ternary search to find optimal point on line that minimizes total distance
        Time Complexity: O(n * log(range))
        Space Complexity: O(1)
        """
        def distance(x):
            if a == 0:
                y = -c / b
            else:
                y = (-a * x - c) / b
            total = 0.0
            for p in points:
                dx = p[0] - x
                dy = p[1] - y
                total += math.sqrt(dx * dx + dy * dy)
            return total
        
        left = -1e6
        right = 1e6
        eps = 1e-6
        
        while right - left > eps:
            m1 = left + (right - left) / 3.0
            m2 = right - (right - left) / 3.0
            
            if distance(m1) < distance(m2):
                right = m2
            else:
                left = m1
        
        return (left + right) / 2.0


def Test_Optimum_Location_Min_Distance():
    sol = Solution()
    tests = [
        ([(0, 0), (2, 0), (3, 0), (5, 0)], [0.0, 1.0, 0.0]),
        ([(1, 1), (2, 2), (3, 3)], [1.0, -1.0, 0.0]),
        ([(0, 1), (1, 0), (2, 1)], [0.0, 1.0, -1.0])
    ]

    for test in tests:
        points = test[0]
        a, b, c = test[1][0], test[1][1], test[1][2]
        
        print("Points:", end=" ")
        for p in points:
            print(f"({p[0]},{p[1]})", end=" ")
        print()
        print(f"Line: {a}x + {b}y + {c} = 0")
        
        res = sol.Optimum_Location_Ternary_Search(points, a, b, c)
        if a == 0:
            y = -c / b
        else:
            y = (-a * res - c) / b
        print(f"Optimum Location: ({res}, {y})")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Optimum_Location_Min_Distance()
