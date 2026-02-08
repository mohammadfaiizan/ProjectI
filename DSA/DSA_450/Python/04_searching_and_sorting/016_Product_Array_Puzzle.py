"""
Problem: Product Array Puzzle
URL: https://practice.geeksforgeeks.org/problems/product-array-puzzle4525/1

Problem Statement:
Given an array nums[] of size n, construct a Product Array P (of same size n) such that P[i] is equal to the product of all the elements of nums except nums[i].

Sample Input/Output:
Input: n = 5, nums[] = {10, 3, 5, 6, 2}
Output: 180 600 360 300 900

Input: n = 2, nums[] = {12, 0}
Output: 0 12
"""


class Solution:
    def Product_Array_Left_Right(self, nums, n):
        """
        Calculate left product and right product arrays, then multiply corresponding elements
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        result = [1] * n
        
        left = 1
        for i in range(n):
            result[i] = left
            left *= nums[i]
        
        right = 1
        for i in range(n - 1, -1, -1):
            result[i] *= right
            right *= nums[i]
        
        return result

    def Product_Array_Zero_Counting(self, nums, n):
        """
        Count zeros and handle division with zero cases
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        product = 1
        zero_count = 0
        zero_index = -1
        
        for i in range(n):
            if nums[i] == 0:
                zero_count += 1
                zero_index = i
            else:
                product *= nums[i]
        
        result = [0] * n
        
        if zero_count > 1:
            return result
        elif zero_count == 1:
            result[zero_index] = product
            return result
        else:
            for i in range(n):
                result[i] = product // nums[i]
            return result


def Test_Product_Array_Puzzle():
    sol = Solution()
    tests = [
        [10, 3, 5, 6, 2],
        [12, 0],
        [1, 2, 3, 4],
        [0, 0, 1, 2],
        [1, 0, 3, 4]
    ]

    for nums in tests:
        n = len(nums)
        print("Array:", end=" ")
        for num in nums:
            print(num, end=" ")
        print()
        
        res1 = sol.Product_Array_Left_Right(nums[:], n)
        res2 = sol.Product_Array_Zero_Counting(nums[:], n)
        
        print("Left-Right Product:", end=" ")
        for val in res1:
            print(val, end=" ")
        print()
        
        print("Zero Counting:", end=" ")
        for val in res2:
            print(val, end=" ")
        print()
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Product_Array_Puzzle()
