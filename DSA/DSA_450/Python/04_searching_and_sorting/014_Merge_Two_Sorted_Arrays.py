"""
Problem: Merge Two Sorted Arrays Without Extra Space
URL: https://practice.geeksforgeeks.org/problems/merge-two-sorted-arrays5135/1

Problem Statement:
Given two sorted arrays arr1[] and arr2[] of sizes n and m in non-decreasing order. Merge them in sorted order without using any extra space.

Sample Input/Output:
Input: n = 4, arr1[] = {1, 3, 5, 7}, m = 5, arr2[] = {0, 2, 6, 8, 9}
Output: arr1[] = {0, 1, 2, 3}, arr2[] = {5, 6, 7, 8, 9}

Input: n = 2, arr1[] = {10, 12}, m = 3, arr2[] = {5, 18, 20}
Output: arr1[] = {5, 10}, arr2[] = {12, 18, 20}
"""


class Solution:
    def Merge_Gap_Method(self, arr1, arr2, n, m):
        """
        Gap method based on shell sort - compare elements at gap distance and swap if needed
        Time Complexity: O((n+m) * log(n+m))
        Space Complexity: O(1)
        """
        total = n + m
        gap = (total + 1) // 2
        
        while gap > 0:
            i = 0
            j = gap
            
            while j < total:
                if i < n and j < n:
                    if arr1[i] > arr1[j]:
                        arr1[i], arr1[j] = arr1[j], arr1[i]
                elif i < n and j >= n:
                    if arr1[i] > arr2[j - n]:
                        arr1[i], arr2[j - n] = arr2[j - n], arr1[i]
                else:
                    if arr2[i - n] > arr2[j - n]:
                        arr2[i - n], arr2[j - n] = arr2[j - n], arr2[i - n]
                i += 1
                j += 1
            
            if gap == 1:
                break
            gap = (gap + 1) // 2

    def Merge_Insertion_Based(self, arr1, arr2, n, m):
        """
        Compare last element of arr1 with first element of arr2 and insert appropriately
        Time Complexity: O(n * m)
        Space Complexity: O(1)
        """
        for i in range(n - 1, -1, -1):
            last = arr1[i]
            j = m - 2
            
            while j >= 0 and arr2[j] > last:
                arr2[j + 1] = arr2[j]
                j -= 1
            
            if j != m - 2 or arr2[j + 1] > last:
                arr2[j + 1] = last
                arr1[i] = arr2[0]
                
                first = arr2[0]
                k = 1
                while k < m and arr2[k] < first:
                    arr2[k - 1] = arr2[k]
                    k += 1
                arr2[k - 1] = first


def Test_Merge_Two_Sorted_Arrays():
    sol = Solution()
    tests = [
        (([1, 3, 5, 7], [0, 2, 6, 8, 9]), (4, 5)),
        (([10, 12], [5, 18, 20]), (2, 3)),
        (([1, 2], [3, 4]), (2, 2)),
        (([1], [1]), (1, 1))
    ]

    for test in tests:
        arr1 = list(test[0][0])
        arr2 = list(test[0][1])
        n = test[1][0]
        m = test[1][1]
        
        print("arr1:", end=" ")
        for num in arr1:
            print(num, end=" ")
        print(", arr2:", end=" ")
        for num in arr2:
            print(num, end=" ")
        print()
        
        arr1a = arr1[:]
        arr2a = arr2[:]
        arr1b = arr1[:]
        arr2b = arr2[:]
        
        sol.Merge_Gap_Method(arr1a, arr2a, n, m)
        print("Gap Method - arr1:", end=" ")
        for num in arr1a:
            print(num, end=" ")
        print(", arr2:", end=" ")
        for num in arr2a:
            print(num, end=" ")
        print()
        
        sol.Merge_Insertion_Based(arr1b, arr2b, n, m)
        print("Insertion Based - arr1:", end=" ")
        for num in arr1b:
            print(num, end=" ")
        print(", arr2:", end=" ")
        for num in arr2b:
            print(num, end=" ")
        print()
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Merge_Two_Sorted_Arrays()
