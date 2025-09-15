"""
Problem: Josephus Problem
URL: https://www.geeksforgeeks.org/dsa/josephus-problem/

Problem Statement:
There are n people standing in a circle waiting to be executed. Counting begins at a specified point in the circle and proceeds around the circle in a specified direction. After a specified number of people are skipped, the next person is executed. The procedure is repeated with the remaining people, starting after the person who was just executed, until only one person remains, and is freed.

Sample Input/Output:
Input: n = 5, k = 2
Output: 3 (0-indexed), 4 (1-indexed)
Explanation: People at positions 1,3,0,4 are eliminated, person at position 2 survives

Input: n = 7, k = 3
Output: 4 (0-indexed), 5 (1-indexed)
"""

from typing import List

class Solution:
    def Josephus_Simulation(self, n: int, k: int) -> int:
        """
        Simulation Approach - Simulate the elimination process
        Time Complexity: O(n * k)
        Space Complexity: O(n)
        """
        people = list(range(n))
        index = 0
        
        while len(people) > 1:
            index = (index + k - 1) % len(people)
            people.pop(index)
            index = index % len(people) if people else 0
        
        return people[0]
    
    def Josephus_Recursive_Optimal(self, n: int, k: int) -> int:
        """
        Recursive Formula - Optimal recursive solution
        Time Complexity: O(n)
        Space Complexity: O(n) - recursion stack
        """
        def Josephus_Helper(num_people: int) -> int:
            if num_people == 1:
                return 0
            return (Josephus_Helper(num_people - 1) + k) % num_people
        
        return Josephus_Helper(n)
    
    def Josephus_Iterative_Optimal(self, n: int, k: int) -> int:
        """
        Iterative Formula - Convert recursion to iteration
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        result = 0
        for i in range(2, n + 1):
            result = (result + k) % i
        return result
    
    def Josephus_Mathematical_Formula(self, n: int, k: int) -> int:
        """
        Mathematical Formula - Direct calculation for k=2
        Time Complexity: O(log n) for k=2, O(n) general
        Space Complexity: O(1)
        """
        if k == 2:
            m = n - (1 << (n.bit_length() - 1))
            return 2 * m
        else:
            result = 0
            for i in range(2, n + 1):
                result = (result + k) % i
            return result

def Test_Josephus():
    solution = Solution()
    
    test_cases = [
        (5, 2, 2),
        (7, 3, 4),
        (1, 1, 0),
        (10, 3, 3),
        (6, 2, 4)
    ]
    
    for n, k, expected in test_cases:
        result1 = solution.Josephus_Simulation(n, k)
        result2 = solution.Josephus_Recursive_Optimal(n, k)
        result3 = solution.Josephus_Iterative_Optimal(n, k)
        result4 = solution.Josephus_Mathematical_Formula(n, k)
        
        print(f"n = {n}, k = {k}")
        print(f"Expected (0-indexed): {expected}")
        print(f"Simulation: {result1}")
        print(f"Recursive Optimal: {result2}")
        print(f"Iterative Optimal: {result3}")
        print(f"Mathematical Formula: {result4}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Josephus()
