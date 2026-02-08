"""
Problem: Maximum Product Subset
URL: https://www.geeksforgeeks.org/maximum-product-subset-array/

Problem Statement:
Find maximum product of a subset of an array (handles negatives and zeros).

Sample Input/Output:
Input: arr[] = {-1, -1, -2, 4, 3}
Output: 24
Explanation: Maximum product is (-1) * (-1) * (-2) * 4 * 3 = 24
"""


class Solution:
    def Maximum_Product_Subset_Count_Negatives_Zeros(self, arr):
        """
        Count negatives/zeros greedy approach: Handle negative count and zeros
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(arr)
        negative_count = 0
        zero_count = 0
        max_negative = float('-inf')
        product = 1
        
        for num in arr:
            if num == 0:
                zero_count += 1
                continue
            if num < 0:
                negative_count += 1
                max_negative = max(max_negative, num)
            product *= num
        
        if zero_count == n:
            return 0
        
        if negative_count % 2 == 1:
            if negative_count == 1 and zero_count > 0 and negative_count + zero_count == n:
                return 0
            product //= max_negative
        
        return product


def Test_Maximum_Product_Subset():
    solution = Solution()
    
    arr1 = [-1, -1, -2, 4, 3]
    print(f"Test 1: {solution.Maximum_Product_Subset_Count_Negatives_Zeros(arr1)}")
    
    arr2 = [-1, 0]
    print(f"Test 2: {solution.Maximum_Product_Subset_Count_Negatives_Zeros(arr2)}")
    
    arr3 = [0, 0, 0]
    print(f"Test 3: {solution.Maximum_Product_Subset_Count_Negatives_Zeros(arr3)}")
    
    arr4 = [-1, -2, -3]
    print(f"Test 4: {solution.Maximum_Product_Subset_Count_Negatives_Zeros(arr4)}")


if __name__ == "__main__":
    Test_Maximum_Product_Subset()
