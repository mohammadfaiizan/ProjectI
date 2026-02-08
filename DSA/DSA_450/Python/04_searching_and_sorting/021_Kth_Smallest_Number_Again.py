"""
Problem: Kth Smallest Number Again
URL: https://www.hackerearth.com/practice/algorithms/searching/binary-search/practice-problems/algorithm/kth-smallest-number-again-2/

Problem Statement:
Given N ranges [a,b], merge overlapping ranges and find kth smallest number
across all ranges.

Sample Input:
2
1 3
4 6
2
2
5

Sample Output:
2
-1
"""


class Solution:
    def Solve_Sort_Merge_Intervals(self, ranges, k):
        """
        Approach: Sort ranges by start, merge overlapping intervals,
        then linearly scan to find which merged interval contains kth element.
        Time Complexity: O(n log n + q*n) where n = ranges, q = queries
        Space Complexity: O(n) for merged intervals
        """
        if not ranges:
            return -1
        
        ranges_sorted = sorted(ranges)
        merged = [ranges_sorted[0]]
        
        for i in range(1, len(ranges_sorted)):
            if ranges_sorted[i][0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], ranges_sorted[i][1]))
            else:
                merged.append(ranges_sorted[i])
        
        current = 1
        for interval in merged:
            count = interval[1] - interval[0] + 1
            if current <= k <= current + count - 1:
                return interval[0] + (k - current)
            current += count
        
        return -1
    
    def Solve_Linear_Scan(self, ranges, k):
        """
        Approach: Merge intervals first, then scan linearly to find kth element.
        Time Complexity: O(n log n + q*n)
        Space Complexity: O(n)
        """
        return self.Solve_Sort_Merge_Intervals(ranges, k)


def Test_Kth_Smallest_Number_Again():
    sol = Solution()
    
    ranges1 = [(1, 3), (4, 6)]
    assert sol.Solve_Sort_Merge_Intervals(ranges1, 2) == 2
    assert sol.Solve_Sort_Merge_Intervals(ranges1, 5) == -1
    
    ranges2 = [(1, 5), (3, 7)]
    assert sol.Solve_Linear_Scan(ranges2, 4) == 4
    assert sol.Solve_Linear_Scan(ranges2, 8) == 7
    
    ranges3 = [(10, 12)]
    assert sol.Solve_Sort_Merge_Intervals(ranges3, 1) == 10
    assert sol.Solve_Sort_Merge_Intervals(ranges3, 3) == 12
    assert sol.Solve_Sort_Merge_Intervals(ranges3, 4) == -1
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Kth_Smallest_Number_Again()
