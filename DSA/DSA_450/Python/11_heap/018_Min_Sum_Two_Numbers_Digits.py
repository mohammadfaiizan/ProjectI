"""
Problem: Minimum Sum of Two Numbers Formed from Digits of an Array
URL: https://practice.geeksforgeeks.org/problems/minimum-sum4058/1

Problem Statement:
Given an array of digits, form two numbers using all digits such that their sum is minimized.

Sample Input/Output:
Input: [6,8,4,5,2,3]
Output: "604"
"""

import heapq


class Solution:
    def Min_Sum_Sort(self, arr):
        """
        Sort Approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        arr_sorted = sorted(arr)
        num1 = ""
        num2 = ""

        for i in range(len(arr_sorted)):
            if i % 2 == 0:
                num1 += str(arr_sorted[i])
            else:
                num2 += str(arr_sorted[i])

        return self.AddStrings(num1, num2)

    def Min_Sum_Min_Heap(self, arr):
        """
        Min Heap Approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        pq = []
        for num in arr:
            heapq.heappush(pq, num)

        num1 = ""
        num2 = ""
        toggle = True

        while pq:
            digit = heapq.heappop(pq)
            if toggle:
                num1 += str(digit)
            else:
                num2 += str(digit)
            toggle = not toggle

        return self.AddStrings(num1, num2)

    def AddStrings(self, num1, num2):
        if not num1:
            return num2
        if not num2:
            return num1

        num1 = num1[::-1]
        num2 = num2[::-1]

        result = ""
        carry = 0
        i, j = 0, 0

        while i < len(num1) or j < len(num2) or carry:
            sum_val = carry
            if i < len(num1):
                sum_val += int(num1[i])
                i += 1
            if j < len(num2):
                sum_val += int(num2[j])
                j += 1
            result += str(sum_val % 10)
            carry = sum_val // 10

        result = result[::-1]

        start = 0
        while start < len(result) and result[start] == '0':
            start += 1

        return "0" if start == len(result) else result[start:]


def Test_Min_Sum():
    solution = Solution()

    arr1 = [6, 8, 4, 5, 2, 3]
    print("Test 1 Sort:", solution.Min_Sum_Sort(arr1))
    arr1b = [6, 8, 4, 5, 2, 3]
    print("Test 1 Min Heap:", solution.Min_Sum_Min_Heap(arr1b))

    arr2 = [5, 3, 0, 7, 4]
    print("Test 2 Sort:", solution.Min_Sum_Sort(arr2))
    arr2b = [5, 3, 0, 7, 4]
    print("Test 2 Min Heap:", solution.Min_Sum_Min_Heap(arr2b))

    arr3 = [1, 2, 3, 4]
    print("Test 3 Sort:", solution.Min_Sum_Sort(arr3))
    arr3b = [1, 2, 3, 4]
    print("Test 3 Min Heap:", solution.Min_Sum_Min_Heap(arr3b))


if __name__ == "__main__":
    Test_Min_Sum()
