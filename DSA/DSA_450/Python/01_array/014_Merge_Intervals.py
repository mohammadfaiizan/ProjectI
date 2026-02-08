"""
Problem: Merge Intervals
URL: https://leetcode.com/problems/merge-intervals/

Problem Statement:
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping
intervals, and return an array of non-overlapping intervals that cover all intervals.

Sample Input/Output:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Intervals [1,3] and [2,6] overlap, merged into [1,6].

Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] overlap.
"""


class Solution:
    def Merge_Intervals_Sort_And_Merge_Optimal(self, intervals):
        """
        Sort and Merge - Sort by start, merge overlapping
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        intervals_sorted = sorted(intervals)
        result = [intervals_sorted[0]]
        for i in range(1, len(intervals_sorted)):
            if intervals_sorted[i][0] <= result[-1][1]:
                result[-1][1] = max(result[-1][1], intervals_sorted[i][1])
            else:
                result.append(intervals_sorted[i])
        return result

    def Merge_Intervals_In_Place(self, intervals):
        """
        In-Place Merge - Track merge index in sorted array
        Time Complexity: O(n log n)
        Space Complexity: O(1) excluding result
        """
        intervals_sorted = sorted(intervals)
        idx = 0
        for i in range(1, len(intervals_sorted)):
            if intervals_sorted[i][0] <= intervals_sorted[idx][1]:
                intervals_sorted[idx][1] = max(intervals_sorted[idx][1], intervals_sorted[i][1])
            else:
                idx += 1
                intervals_sorted[idx] = intervals_sorted[i]
        return intervals_sorted[:idx + 1]


def Test_Merge_Intervals():
    solution = Solution()

    test_cases = [
        [[1, 3], [2, 6], [8, 10], [15, 18]],
        [[1, 4], [4, 5]],
        [[1, 4], [0, 4]],
        [[1, 4], [2, 3]]
    ]

    for intervals in test_cases:
        print("Intervals:", intervals)

        r1 = solution.Merge_Intervals_Sort_And_Merge_Optimal([x[:] for x in intervals])
        print("Sort & Merge:", r1)

        r2 = solution.Merge_Intervals_In_Place([x[:] for x in intervals])
        print("In-Place:", r2)

        print("-" * 50)


if __name__ == "__main__":
    Test_Merge_Intervals()
