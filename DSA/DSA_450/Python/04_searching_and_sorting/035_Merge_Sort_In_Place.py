"""
Problem: In-Place Merge Sort
URL: https://www.geeksforgeeks.org/in-place-merge-sort/

Problem Statement:
Implement merge sort with O(1) extra space using modular arithmetic trick.

Sample Input:
arr[] = {12, 11, 13, 5, 6, 7}

Sample Output:
{5, 6, 7, 11, 12, 13}
"""


class Solution:
    def Merge_In_Place(self, arr, left, mid, right):
        """
        Approach: In-place merge using modular arithmetic
        Store two values at same position: arr[i] = arr[i] + arr[j] * max_val
        Extract original values using modulo and division
        
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        max_val = max(arr[left:right + 1]) + 1
        i = left
        j = mid + 1
        k = left
        
        while i <= mid and j <= right and k <= right:
            val1 = arr[i] % max_val
            val2 = arr[j] % max_val
            
            if val1 <= val2:
                arr[k] += val1 * max_val
                i += 1
            else:
                arr[k] += val2 * max_val
                j += 1
            k += 1
        
        while i <= mid:
            val1 = arr[i] % max_val
            arr[k] += val1 * max_val
            i += 1
            k += 1
        
        while j <= right:
            val2 = arr[j] % max_val
            arr[k] += val2 * max_val
            j += 1
            k += 1
        
        for i in range(left, right + 1):
            arr[i] //= max_val

    def Merge_Sort_In_Place(self, arr, left, right):
        if left < right:
            mid = left + (right - left) // 2
            self.Merge_Sort_In_Place(arr, left, mid)
            self.Merge_Sort_In_Place(arr, mid + 1, right)
            self.Merge_In_Place(arr, left, mid, right)

    def Merge_Standard(self, arr, temp, left, mid, right):
        """
        Approach: Standard merge sort with extra space
        Use temporary array to merge two sorted halves
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        i = left
        j = mid + 1
        k = left
        
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                i += 1
            else:
                temp[k] = arr[j]
                j += 1
            k += 1
        
        while i <= mid:
            temp[k] = arr[i]
            i += 1
            k += 1
        
        while j <= right:
            temp[k] = arr[j]
            j += 1
            k += 1
        
        for i in range(left, right + 1):
            arr[i] = temp[i]

    def Merge_Sort_Standard(self, arr, temp, left, right):
        if left < right:
            mid = left + (right - left) // 2
            self.Merge_Sort_Standard(arr, temp, left, mid)
            self.Merge_Sort_Standard(arr, temp, mid + 1, right)
            self.Merge_Standard(arr, temp, left, mid, right)


def Test_Merge_Sort_In_Place():
    sol = Solution()
    
    arr1 = [12, 11, 13, 5, 6, 7]
    arr1_copy = [12, 11, 13, 5, 6, 7]
    sol.Merge_Sort_In_Place(arr1, 0, 5)
    expected1 = [5, 6, 7, 11, 12, 13]
    for i in range(6):
        assert arr1[i] == expected1[i]
    
    temp = [0] * 6
    sol.Merge_Sort_Standard(arr1_copy, temp, 0, 5)
    for i in range(6):
        assert arr1_copy[i] == expected1[i]
    
    arr2 = [5, 4, 3, 2, 1]
    arr2_copy = [5, 4, 3, 2, 1]
    sol.Merge_Sort_In_Place(arr2, 0, 4)
    expected2 = [1, 2, 3, 4, 5]
    for i in range(5):
        assert arr2[i] == expected2[i]
    
    temp2 = [0] * 5
    sol.Merge_Sort_Standard(arr2_copy, temp2, 0, 4)
    for i in range(5):
        assert arr2_copy[i] == expected2[i]
    
    arr3 = [1]
    sol.Merge_Sort_In_Place(arr3, 0, 0)
    assert arr3[0] == 1
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Merge_Sort_In_Place()
