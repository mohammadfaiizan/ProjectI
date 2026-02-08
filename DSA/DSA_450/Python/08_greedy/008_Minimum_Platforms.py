"""
Problem: Minimum Platforms
URL: https://practice.geeksforgeeks.org/problems/minimum-platforms-1587115620/1

Problem Statement:
Given arrival and departure times of all trains that reach a railway station, find the minimum number of platforms required for the railway station so that no train waits.

Sample Input/Output:
Input: arr[] = {900, 940, 950, 1100, 1500, 1800}, dep[] = {910, 1200, 1120, 1130, 1900, 2000}
Output: 3
Explanation: Minimum 3 platforms are required to accommodate all trains.
"""


class Solution:
    def Find_Platform_Sort_Two_Pointer(self, arr, dep, n):
        """
        Sort both arrays, use two pointers to track overlapping trains
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr.sort()
        dep.sort()
        
        platforms = 1
        max_platforms = 1
        i = 1
        j = 0
        
        while i < n and j < n:
            if arr[i] <= dep[j]:
                platforms += 1
                i += 1
            else:
                platforms -= 1
                j += 1
            max_platforms = max(max_platforms, platforms)
        
        return max_platforms


def Test_Minimum_Platforms():
    solution = Solution()
    arr = [900, 940, 950, 1100, 1500, 1800]
    dep = [910, 1200, 1120, 1130, 1900, 2000]
    n = 6
    print(f"Minimum platforms: {solution.Find_Platform_Sort_Two_Pointer(arr, dep, n)}")


if __name__ == "__main__":
    Test_Minimum_Platforms()
