"""
Problem: Bishu and Soldiers
URL: https://www.hackerearth.com/practice/algorithms/searching/binary-search/practice-problems/algorithm/bishu-and-soldiers/

Problem Statement:
Bishu is fighting with soldiers. Each soldier has a power level.
In each round, Bishu has a power level. Find how many soldiers Bishu can defeat
(soldiers with power <= Bishu's power) and the sum of their powers.

Sample Input:
7
1 2 3 4 5 6 7
3
3
10
2

Sample Output:
3 6
7 28
2 3
"""

import bisect


class Solution:
    def Solve_Binary_Search(self, soldiers, rounds):
        """
        Approach: Sort soldiers, then for each round use binary search (bisect_right)
        to find count of soldiers with power <= round power, and sum their powers.
        Time Complexity: O(n log n + q log n) where n = soldiers, q = rounds
        Space Complexity: O(n) for sorted array
        """
        soldiers_sorted = sorted(soldiers)
        prefix_sum = [0] * (len(soldiers_sorted) + 1)
        for i in range(len(soldiers_sorted)):
            prefix_sum[i + 1] = prefix_sum[i] + soldiers_sorted[i]
        
        result = []
        for power in rounds:
            idx = bisect.bisect_right(soldiers_sorted, power)
            count = idx
            sum_val = prefix_sum[idx]
            result.append((count, sum_val))
        return result
    
    def Solve_Prefix_Sum_Binary_Search(self, soldiers, rounds):
        """
        Approach: Sort soldiers, build prefix sum array, then binary search for each round.
        Time Complexity: O(n log n + q log n)
        Space Complexity: O(n) for prefix sum
        """
        soldiers_sorted = sorted(soldiers)
        prefix_sum = [0] * (len(soldiers_sorted) + 1)
        for i in range(len(soldiers_sorted)):
            prefix_sum[i + 1] = prefix_sum[i] + soldiers_sorted[i]
        
        result = []
        for power in rounds:
            idx = bisect.bisect_right(soldiers_sorted, power)
            result.append((idx, prefix_sum[idx]))
        return result


def Test_Bishu_And_Soldiers():
    sol = Solution()
    
    soldiers1 = [1, 2, 3, 4, 5, 6, 7]
    rounds1 = [3, 10, 2]
    result1 = sol.Solve_Binary_Search(soldiers1, rounds1)
    assert result1[0] == (3, 6)
    assert result1[1] == (7, 28)
    assert result1[2] == (2, 3)
    
    soldiers2 = [5, 3, 1, 4, 2]
    rounds2 = [3, 6]
    result2 = sol.Solve_Prefix_Sum_Binary_Search(soldiers2, rounds2)
    assert result2[0] == (3, 6)
    assert result2[1] == (5, 15)
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Bishu_And_Soldiers()
