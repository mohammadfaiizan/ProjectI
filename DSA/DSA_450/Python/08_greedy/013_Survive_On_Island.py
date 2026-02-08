"""
Problem: Survive On Island
URL: https://www.geeksforgeeks.org/survival/

Problem Statement:
Given S days to survive, N units of food can be bought per day, M units needed per day, and shop closed on Sundays. Can you survive? If yes, find min buying days.

Sample Input/Output:
Input: S=10, N=16, M=2
Output: 2
Explanation: Need 20 units total. Can buy 16 units on day 1 (Monday), 16 units on day 2 (Tuesday) = 32 units. Shop closed on Sunday (day 7). Can survive with 2 buying days.
"""


class Solution:
    def Survive_On_Island_Math_Greedy(self, S, N, M):
        """
        Math/greedy approach: Calculate total food needed and check if we can buy enough before Sundays
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        total_food_needed = S * M
        sundays = S // 7
        buying_days_available = S - sundays
        
        if buying_days_available * N < total_food_needed:
            return -1
        
        return (total_food_needed + N - 1) // N


def Test_Survive_On_Island():
    solution = Solution()
    
    print(f"Test 1: S=10, N=16, M=2")
    print(f"Result: {solution.Survive_On_Island_Math_Greedy(10, 16, 2)}")
    
    print(f"\nTest 2: S=10, N=20, M=30")
    print(f"Result: {solution.Survive_On_Island_Math_Greedy(10, 20, 30)}")
    
    print(f"\nTest 3: S=6, N=10, M=2")
    print(f"Result: {solution.Survive_On_Island_Math_Greedy(6, 10, 2)}")


if __name__ == "__main__":
    Test_Survive_On_Island()
