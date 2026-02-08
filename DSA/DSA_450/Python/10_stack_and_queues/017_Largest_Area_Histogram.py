"""
Problem: Largest Rectangular Area in a Histogram
URL: https://practice.geeksforgeeks.org/problems/maximum-rectangular-area-in-a-histogram-1587115620/1

Problem Statement:
Find the largest rectangular area possible in a histogram where bars have unit width.

Sample Input/Output:
Input: [6,2,5,4,5,1,6]
Output: 12
Input: [2,1,5,6,2,3]
Output: 10
"""


class Solution:
    def Largest_Area_Histogram_Stack(self, heights):
        """
        Find largest area using stack approach.
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        st = []
        maxArea = 0
        n = len(heights)
        for i in range(n + 1):
            while st and (i == n or heights[st[-1]] >= heights[i]):
                height = heights[st.pop()]
                width = i if not st else i - st[-1] - 1
                maxArea = max(maxArea, height * width)
            st.append(i)
        return maxArea

    def Largest_Area_Histogram_Divide_Conquer(self, heights, left, right):
        """
        Find largest area using divide and conquer.
        Time Complexity: O(n log n)
        Space Complexity: O(log n)
        """
        if left > right:
            return 0
        if left == right:
            return heights[left]
        minIdx = left
        for i in range(left, right + 1):
            if heights[i] < heights[minIdx]:
                minIdx = i
        area = heights[minIdx] * (right - left + 1)
        leftArea = self.Largest_Area_Histogram_Divide_Conquer(heights, left, minIdx - 1)
        rightArea = self.Largest_Area_Histogram_Divide_Conquer(heights, minIdx + 1, right)
        return max(area, leftArea, rightArea)


def Test_Largest_Area_Histogram():
    solution = Solution()
    
    print("=== Stack Approach ===")
    heights1 = [6, 2, 5, 4, 5, 1, 6]
    print(f"Input: {heights1}")
    print(f"Output: {solution.Largest_Area_Histogram_Stack(heights1)}")
    
    heights2 = [2, 1, 5, 6, 2, 3]
    print(f"\nInput: {heights2}")
    print(f"Output: {solution.Largest_Area_Histogram_Stack(heights2)}")
    
    heights3 = [1, 2, 3, 4, 5]
    print(f"\nInput: {heights3}")
    print(f"Output: {solution.Largest_Area_Histogram_Stack(heights3)}")
    
    print("\n=== Divide and Conquer Approach ===")
    heights4 = [6, 2, 5, 4, 5, 1, 6]
    print(f"Input: {heights4}")
    print(f"Output: {solution.Largest_Area_Histogram_Divide_Conquer(heights4, 0, len(heights4) - 1)}")
    
    heights5 = [2, 1, 5, 6, 2, 3]
    print(f"\nInput: {heights5}")
    print(f"Output: {solution.Largest_Area_Histogram_Divide_Conquer(heights5, 0, len(heights5) - 1)}")


if __name__ == "__main__":
    Test_Largest_Area_Histogram()
