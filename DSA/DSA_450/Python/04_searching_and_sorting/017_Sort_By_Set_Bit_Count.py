"""
Problem: Sort Array by Set Bit Count
URL: https://practice.geeksforgeeks.org/problems/sort-by-set-bit-count1153/1

Problem Statement:
Given an array of integers, sort the array (in decreasing order) according to count of set bits in binary representation of array elements.

Sample Input/Output:
Input: arr[] = {5, 2, 3, 9, 4, 6, 7, 15, 32}
Output: 15 7 5 3 9 6 2 4 32

Input: arr[] = {1, 2, 3, 4, 5, 6}
Output: 3 5 6 1 2 4
"""


class Solution:
    def Sort_By_Set_Bit_Custom_Comparator(self, arr, n):
        """
        Use custom comparator with stable_sort to sort by set bit count
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        def count_bits(num):
            count = 0
            while num:
                count += num & 1
                num >>= 1
            return count
        
        arr.sort(key=lambda x: (-count_bits(x), x))

    def Sort_By_Set_Bit_Builtin_Popcount(self, arr, n):
        """
        Use bin().count('1') to count set bits efficiently
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr.sort(key=lambda x: (-bin(x).count('1'), x))


def Test_Sort_By_Set_Bit_Count():
    sol = Solution()
    tests = [
        [5, 2, 3, 9, 4, 6, 7, 15, 32],
        [1, 2, 3, 4, 5, 6],
        [1024, 512, 256, 128, 64],
        [7, 8, 9, 10, 11]
    ]

    for arr in tests:
        n = len(arr)
        print("Original Array:", end=" ")
        for num in arr:
            print(num, end=" ")
        print()
        
        arr1 = arr[:]
        arr2 = arr[:]
        
        sol.Sort_By_Set_Bit_Custom_Comparator(arr1, n)
        print("Custom Comparator:", end=" ")
        for num in arr1:
            print(num, end=" ")
        print()
        
        sol.Sort_By_Set_Bit_Builtin_Popcount(arr2, n)
        print("Builtin Popcount:", end=" ")
        for num in arr2:
            print(num, end=" ")
        print()
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Sort_By_Set_Bit_Count()
