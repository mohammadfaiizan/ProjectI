"""
Problem: Chocolate Distribution
URL: https://practice.geeksforgeeks.org/problems/chocolate-distribution-problem3825/1

Problem Statement:
Given N packets with chocolates and M children, distribute one packet each. Minimize difference between max and min chocolates given.

Sample Input/Output:
Input: packets[] = {7, 3, 2, 4, 9, 12, 56}, M = 3
Output: 2
Explanation: Distribute packets {2, 3, 4}. Difference = 4 - 2 = 2 (minimum).
"""


class Solution:
    def Chocolate_Distribution_Sort_Sliding_Window(self, packets, M):
        """
        Sort + sliding window of size M greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        packets.sort()
        n = len(packets)
        
        if M > n:
            return -1
        
        min_diff = float('inf')
        
        for i in range(n - M + 1):
            diff = packets[i + M - 1] - packets[i]
            min_diff = min(min_diff, diff)
        
        return min_diff


def Test_Chocolate_Distribution():
    solution = Solution()
    
    packets1 = [7, 3, 2, 4, 9, 12, 56]
    print(f"Test 1: {solution.Chocolate_Distribution_Sort_Sliding_Window(packets1, 3)}")
    
    packets2 = [3, 4, 1, 9, 56, 7, 9, 12]
    print(f"Test 2: {solution.Chocolate_Distribution_Sort_Sliding_Window(packets2, 5)}")
    
    packets3 = [12, 4, 7, 9, 2, 23, 25, 41, 30, 40, 28, 42, 30, 44, 48, 43, 50]
    print(f"Test 3: {solution.Chocolate_Distribution_Sort_Sliding_Window(packets3, 7)}")


if __name__ == "__main__":
    Test_Chocolate_Distribution()
