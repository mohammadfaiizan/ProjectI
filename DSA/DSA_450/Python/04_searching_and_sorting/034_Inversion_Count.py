"""
Problem: Inversion of Array
URL: https://practice.geeksforgeeks.org/problems/inversion-of-array-1587115620/1

Problem Statement:
Count number of inversions (i < j but arr[i] > arr[j]).
Modified merge sort approach.

Sample Input:
arr[] = {2, 4, 1, 3, 5}

Sample Output:
3
"""


class Solution:
    def Merge_And_Count(self, arr, temp, left, mid, right):
        """
        Approach: Modified merge sort
        During merge, count inversions when element from right half is smaller
        Inversions = number of elements remaining in left half
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        i = left
        j = mid + 1
        k = left
        inversions = 0
        
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp[k] = arr[i]
                i += 1
            else:
                temp[k] = arr[j]
                j += 1
                inversions += (mid - i + 1)
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
        
        return inversions

    def Merge_Sort_And_Count(self, arr, temp, left, right):
        inversions = 0
        if left < right:
            mid = left + (right - left) // 2
            inversions += self.Merge_Sort_And_Count(arr, temp, left, mid)
            inversions += self.Merge_Sort_And_Count(arr, temp, mid + 1, right)
            inversions += self.Merge_And_Count(arr, temp, left, mid, right)
        return inversions

    def Inversion_Count_Merge_Sort(self, arr, n):
        temp = [0] * n
        return self.Merge_Sort_And_Count(arr, temp, 0, n - 1)

    def Inversion_Count_Brute_Force(self, arr, n):
        """
        Approach: Brute force
        Check all pairs (i, j) where i < j and arr[i] > arr[j]
        
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        count = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if arr[i] > arr[j]:
                    count += 1
        return count


def Test_Inversion_Count():
    sol = Solution()
    
    arr1 = [2, 4, 1, 3, 5]
    arr1_copy = [2, 4, 1, 3, 5]
    assert sol.Inversion_Count_Merge_Sort(arr1[:], 5) == 3
    assert sol.Inversion_Count_Brute_Force(arr1_copy[:], 5) == 3
    
    arr2 = [2, 3, 4, 5, 6]
    arr2_copy = [2, 3, 4, 5, 6]
    assert sol.Inversion_Count_Merge_Sort(arr2[:], 5) == 0
    assert sol.Inversion_Count_Brute_Force(arr2_copy[:], 5) == 0
    
    arr3 = [5, 4, 3, 2, 1]
    arr3_copy = [5, 4, 3, 2, 1]
    assert sol.Inversion_Count_Merge_Sort(arr3[:], 5) == 10
    assert sol.Inversion_Count_Brute_Force(arr3_copy[:], 5) == 10
    
    arr4 = [1]
    assert sol.Inversion_Count_Merge_Sort(arr4[:], 1) == 0
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Inversion_Count()
