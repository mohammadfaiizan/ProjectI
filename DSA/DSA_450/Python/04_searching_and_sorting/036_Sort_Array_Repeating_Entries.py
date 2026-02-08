"""
Problem: Partitioning and Sorting Array with Repeating Entries
URL: https://www.baeldung.com/java-sorting-arrays-with-repeated-entries

Problem Statement:
Sort array with many repeated entries efficiently using 3-way partitioning (Dutch National Flag).

Sample Input:
arr[] = {2, 0, 2, 1, 1, 0}

Sample Output:
{0, 0, 1, 1, 2, 2}
"""


class Solution:
    def Three_Way_Partition(self, arr, start, n, pivot):
        """
        Approach: Three-way partition (Dutch National Flag)
        Partition array into three parts: < pivot, == pivot, > pivot
        Efficient for arrays with many duplicates
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        low = start
        mid = start
        high = start + n - 1
        
        while mid <= high:
            if arr[mid] < pivot:
                arr[low], arr[mid] = arr[mid], arr[low]
                low += 1
                mid += 1
            elif arr[mid] == pivot:
                mid += 1
            else:
                arr[mid], arr[high] = arr[high], arr[mid]
                high -= 1

    def Sort_Three_Way(self, arr, left, right):
        if right - left <= 1:
            return
        
        arr_slice = arr[left:right]
        min_val = min(arr_slice)
        max_val = max(arr_slice)
        
        if min_val == max_val:
            return
        
        pivot = min_val + (max_val - min_val) // 2
        self.Three_Way_Partition(arr, left, right - left, pivot)
        
        pivot_pos = -1
        for i in range(left, right):
            if arr[i] == pivot:
                pivot_pos = i
                break
        
        if pivot_pos != -1:
            left_size = pivot_pos - left
            right_start = pivot_pos
            while right_start < right and arr[right_start] == pivot:
                right_start += 1
            
            if left_size > 0:
                self.Sort_Three_Way(arr, left, pivot_pos)
            if right_start < right:
                self.Sort_Three_Way(arr, right_start, right)

    def Sort_Counting(self, arr, n):
        """
        Approach: Counting sort
        Count frequency of each element, then reconstruct array
        Efficient when range of values is small
        
        Time Complexity: O(n + k) where k is range of values
        Space Complexity: O(k)
        """
        if n == 0:
            return
        
        min_val = min(arr)
        max_val = max(arr)
        range_val = max_val - min_val + 1
        
        count = [0] * range_val
        
        for i in range(n):
            count[arr[i] - min_val] += 1
        
        idx = 0
        for i in range(range_val):
            while count[i] > 0:
                arr[idx] = i + min_val
                idx += 1
                count[i] -= 1

    def Sort_STL(self, arr, n):
        """
        Approach: STL sort
        Use standard library sort function
        
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr.sort()


def Test_Sort_Array_Repeating_Entries():
    sol = Solution()
    
    arr1 = [2, 0, 2, 1, 1, 0]
    arr1_copy1 = [2, 0, 2, 1, 1, 0]
    arr1_copy2 = [2, 0, 2, 1, 1, 0]
    arr1_copy3 = [2, 0, 2, 1, 1, 0]
    
    sol.Sort_Three_Way(arr1, 0, 6)
    sol.Sort_Counting(arr1_copy1, 6)
    sol.Sort_STL(arr1_copy2, 6)
    
    expected = [0, 0, 1, 1, 2, 2]
    for i in range(6):
        assert arr1[i] == expected[i]
        assert arr1_copy1[i] == expected[i]
        assert arr1_copy2[i] == expected[i]
    
    arr2 = [1, 1, 1, 1]
    arr2_copy = [1, 1, 1, 1]
    sol.Sort_Three_Way(arr2, 0, 4)
    sol.Sort_Counting(arr2_copy, 4)
    for i in range(4):
        assert arr2[i] == 1
        assert arr2_copy[i] == 1
    
    arr3 = [3, 1, 2]
    arr3_copy = [3, 1, 2]
    sol.Sort_Three_Way(arr3, 0, 3)
    sol.Sort_Counting(arr3_copy, 3)
    expected3 = [1, 2, 3]
    for i in range(3):
        assert arr3[i] == expected3[i]
        assert arr3_copy[i] == expected3[i]
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Sort_Array_Repeating_Entries()
