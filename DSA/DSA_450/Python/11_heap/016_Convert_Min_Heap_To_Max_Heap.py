"""
Problem: Convert Min Heap to Max Heap
URL: https://www.geeksforgeeks.org/convert-min-heap-to-max-heap/

Problem Statement:
Given an array representing a min heap, convert it to a max heap.

Sample Input/Output:
Input: [3,5,9,6,8,20,10,12,18,9]
Output: valid max heap
"""


class Solution:
    def Convert_Min_To_Max_Heapify(self, arr):
        """
        Heapify Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            self.MaxHeapify(arr, i, n)

    def MaxHeapify(self, arr, i, n):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left

        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.MaxHeapify(arr, largest, n)


def IsMaxHeap(arr):
    n = len(arr)
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[i]:
            return False
        if right < n and arr[right] > arr[i]:
            return False
    return True


def Test_Convert_Min_To_Max():
    solution = Solution()

    arr1 = [3, 5, 9, 6, 8, 20, 10, 12, 18, 9]
    print("Original Min Heap:", arr1)

    solution.Convert_Min_To_Max_Heapify(arr1)
    print("Converted Max Heap:", arr1)
    print("Is Valid Max Heap:", IsMaxHeap(arr1))

    arr2 = [1, 2, 3, 4, 5]
    print("\nOriginal Min Heap 2:", arr2)

    solution.Convert_Min_To_Max_Heapify(arr2)
    print("Converted Max Heap 2:", arr2)
    print("Is Valid Max Heap 2:", IsMaxHeap(arr2))

    arr3 = [10, 8, 9, 5, 6, 7, 4]
    print("\nOriginal Min Heap 3:", arr3)

    solution.Convert_Min_To_Max_Heapify(arr3)
    print("Converted Max Heap 3:", arr3)
    print("Is Valid Max Heap 3:", IsMaxHeap(arr3))


if __name__ == "__main__":
    Test_Convert_Min_To_Max()
