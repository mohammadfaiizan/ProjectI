"""
Problem: K-th Element of Two Sorted Arrays
URL: https://practice.geeksforgeeks.org/problems/k-th-element-of-two-sorted-array1317/1

Problem Statement:
Given two sorted arrays of size m and n, find the kth element
in the union of the two arrays.

Sample Input:
arr1 = [2, 3, 6, 7, 9]
arr2 = [1, 4, 8, 10]
k = 5

Sample Output:
6
"""

import sys


class Solution:
    def Kth_Element_Merge(self, arr1, arr2, k):
        """
        Approach: Merge both arrays until kth element is found.
        Use two pointers to traverse both arrays simultaneously.
        Time Complexity: O(k)
        Space Complexity: O(1)
        """
        i = 0
        j = 0
        count = 0
        m = len(arr1)
        n = len(arr2)
        
        while i < m and j < n:
            if arr1[i] <= arr2[j]:
                count += 1
                if count == k:
                    return arr1[i]
                i += 1
            else:
                count += 1
                if count == k:
                    return arr2[j]
                j += 1
        
        while i < m:
            count += 1
            if count == k:
                return arr1[i]
            i += 1
        
        while j < n:
            count += 1
            if count == k:
                return arr2[j]
            j += 1
        
        return -1
    
    def Kth_Element_Binary_Search(self, arr1, arr2, k):
        """
        Approach: Binary search on smaller array. Partition both arrays
        such that left partition has k elements total.
        Time Complexity: O(log(min(m,n)))
        Space Complexity: O(1)
        """
        m = len(arr1)
        n = len(arr2)
        if m > n:
            return self.Kth_Element_Binary_Search(arr2, arr1, k)
        
        left = max(0, k - n)
        right = min(k, m)
        
        while left <= right:
            partition1 = (left + right) // 2
            partition2 = k - partition1
            
            left1 = -sys.maxsize if partition1 == 0 else arr1[partition1 - 1]
            right1 = sys.maxsize if partition1 == m else arr1[partition1]
            left2 = -sys.maxsize if partition2 == 0 else arr2[partition2 - 1]
            right2 = sys.maxsize if partition2 == n else arr2[partition2]
            
            if left1 <= right2 and left2 <= right1:
                return max(left1, left2)
            elif left1 > right2:
                right = partition1 - 1
            else:
                left = partition1 + 1
        
        return -1
    
    def Kth_Element_Min_Heap(self, arr1, arr2, k):
        """
        Approach: Use min heap to merge and find kth element.
        Push elements from both arrays and pop k times.
        Time Complexity: O(k log k)
        Space Complexity: O(k)
        """
        import heapq
        
        pq = []
        
        for num in arr1:
            heapq.heappush(pq, num)
        
        for num in arr2:
            heapq.heappush(pq, num)
        
        for i in range(k - 1):
            heapq.heappop(pq)
        
        return heapq.heappop(pq)


def Test_Kth_Element_Two_Sorted_Arrays():
    sol = Solution()
    
    arr1 = [2, 3, 6, 7, 9]
    arr2 = [1, 4, 8, 10]
    assert sol.Kth_Element_Merge(arr1, arr2, 5) == 6
    assert sol.Kth_Element_Binary_Search(arr1, arr2, 5) == 6
    assert sol.Kth_Element_Min_Heap(arr1, arr2, 5) == 6
    
    arr3 = [1, 3, 5]
    arr4 = [2, 4, 6]
    assert sol.Kth_Element_Merge(arr3, arr4, 4) == 4
    assert sol.Kth_Element_Binary_Search(arr3, arr4, 4) == 4
    
    arr5 = [1]
    arr6 = [2, 3, 4]
    assert sol.Kth_Element_Merge(arr5, arr6, 2) == 2
    assert sol.Kth_Element_Binary_Search(arr5, arr6, 2) == 2
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Kth_Element_Two_Sorted_Arrays()
