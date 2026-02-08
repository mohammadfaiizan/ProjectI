"""
Problem: Minimum Swaps to Sort
URL: https://practice.geeksforgeeks.org/problems/minimum-swaps/1

Problem Statement:
Given an array of n distinct elements. Find the minimum number of swaps required to sort the array in strictly increasing order.

Sample Input/Output:
Input: nums[] = {2, 8, 5, 4}
Output: 1

Input: nums[] = {10, 19, 6, 3, 5}
Output: 2
"""


class Solution:
    def Min_Swaps_Graph_Cycle_Detection(self, nums, n):
        """
        Create graph of cycles where each element should be at its sorted position
        Count cycles and swaps needed = n - number of cycles
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        arr_pos = []
        for i in range(n):
            arr_pos.append((nums[i], i))
        
        arr_pos.sort()
        
        visited = [False] * n
        swaps = 0
        
        for i in range(n):
            if visited[i] or arr_pos[i][1] == i:
                continue
            
            cycle_size = 0
            j = i
            
            while not visited[j]:
                visited[j] = True
                j = arr_pos[j][1]
                cycle_size += 1
            
            if cycle_size > 0:
                swaps += (cycle_size - 1)
        
        return swaps

    def Min_Swaps_HashMap_Tracking(self, nums, n):
        """
        Use hash map to track correct positions and count swaps needed
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        pos_map = {}
        for i in range(n):
            pos_map[nums[i]] = i
        
        sorted_nums = sorted(nums)
        
        swaps = 0
        visited = [False] * n
        
        for i in range(n):
            if visited[i] or nums[i] == sorted_nums[i]:
                continue
            
            cycle_size = 0
            j = i
            
            while not visited[j]:
                visited[j] = True
                j = pos_map[sorted_nums[j]]
                cycle_size += 1
            
            swaps += (cycle_size - 1)
        
        return swaps


def Test_Min_Swaps_To_Sort():
    sol = Solution()
    tests = [
        [2, 8, 5, 4],
        [10, 19, 6, 3, 5],
        [1, 5, 4, 3, 2],
        [1, 2, 3, 4, 5],
        [4, 3, 2, 1]
    ]

    for nums in tests:
        n = len(nums)
        print("Array:", end=" ")
        for num in nums:
            print(num, end=" ")
        print()
        
        nums1 = nums[:]
        nums2 = nums[:]
        res1 = sol.Min_Swaps_Graph_Cycle_Detection(nums1, n)
        res2 = sol.Min_Swaps_HashMap_Tracking(nums2, n)
        
        print(f"Graph Cycle Detection: {res1}")
        print(f"HashMap Tracking: {res2}")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Min_Swaps_To_Sort()
