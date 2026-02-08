"""
Problem: Merge Overlapping Intervals
URL: https://practice.geeksforgeeks.org/problems/overlapping-intervals--170633/1

Problem Statement:
Given intervals as pairs, merge all overlapping intervals.

Sample Input/Output:
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
"""


class Solution:
    def Merge_Overlapping_Intervals_Sort(self, intervals):
        """
        Merge overlapping intervals using sorting.
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if not intervals:
            return []
        intervals.sort()
        merged = [intervals[0]]
        for i in range(1, len(intervals)):
            if merged[-1][1] >= intervals[i][0]:
                merged[-1][1] = max(merged[-1][1], intervals[i][1])
            else:
                merged.append(intervals[i])
        return merged

    def Merge_Overlapping_Intervals_Stack(self, intervals):
        """
        Merge overlapping intervals using stack.
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if not intervals:
            return []
        intervals.sort()
        st = [intervals[0]]
        for i in range(1, len(intervals)):
            if st[-1][1] >= intervals[i][0]:
                top = st.pop()
                top[1] = max(top[1], intervals[i][1])
                st.append(top)
            else:
                st.append(intervals[i])
        merged = []
        while st:
            merged.append(st.pop())
        merged.reverse()
        return merged


def Test_Merge_Overlapping_Intervals():
    solution = Solution()
    
    print("=== Sort Approach ===")
    intervals1 = [[1,3], [2,6], [8,10], [15,18]]
    print(f"Input: [1,3] [2,6] [8,10] [15,18]")
    result1 = solution.Merge_Overlapping_Intervals_Sort(intervals1)
    print(f"Output: {result1}")
    
    intervals2 = [[1,4], [4,5]]
    print(f"\nInput: [1,4] [4,5]")
    result2 = solution.Merge_Overlapping_Intervals_Sort(intervals2)
    print(f"Output: {result2}")
    
    intervals3 = [[1,9], [2,4], [4,7], [6,8]]
    print(f"\nInput: [1,9] [2,4] [4,7] [6,8]")
    result3 = solution.Merge_Overlapping_Intervals_Sort(intervals3)
    print(f"Output: {result3}")
    
    print("\n=== Stack Approach ===")
    intervals4 = [[1,3], [2,6], [8,10], [15,18]]
    print(f"Input: [1,3] [2,6] [8,10] [15,18]")
    result4 = solution.Merge_Overlapping_Intervals_Stack(intervals4)
    print(f"Output: {result4}")


if __name__ == "__main__":
    Test_Merge_Overlapping_Intervals()
