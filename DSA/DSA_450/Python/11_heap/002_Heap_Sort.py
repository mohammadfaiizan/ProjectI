"""
Problem: Heap Sort
URL: https://www.geeksforgeeks.org/heap-sort/

Problem Statement:
Sort an array using heap sort algorithm. Build max heap, then repeatedly extract max.

Sample Input/Output:
Input: [12,11,13,5,6,7]
Output: [5,6,7,11,12,13]
"""


class Solution:
    def Heap_Sort_Max_Heap(self, arr):
        """
        Heap Sort Max Heap (Ascending Order)
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        n = len(arr)

        for i in range(n // 2 - 1, -1, -1):
            self.Max_Heapify(arr, n, i)

        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self.Max_Heapify(arr, i, 0)

    def Heap_Sort_Min_Heap(self, arr):
        """
        Heap Sort Min Heap (Descending Order)
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        n = len(arr)

        for i in range(n // 2 - 1, -1, -1):
            self.Min_Heapify(arr, n, i)

        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            self.Min_Heapify(arr, i, 0)

    def Max_Heapify(self, arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left

        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.Max_Heapify(arr, n, largest)

    def Min_Heapify(self, arr, n, i):
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] < arr[smallest]:
            smallest = left

        if right < n and arr[right] < arr[smallest]:
            smallest = right

        if smallest != i:
            arr[i], arr[smallest] = arr[smallest], arr[i]
            self.Min_Heapify(arr, n, smallest)


def Test_Heap_Sort():
    solution = Solution()

    arr1 = [12, 11, 13, 5, 6, 7]
    print("Original:", arr1)
    solution.Heap_Sort_Max_Heap(arr1)
    print("Sorted (Ascending):", arr1)

    arr2 = [4, 10, 3, 5, 1]
    print("\nOriginal:", arr2)
    solution.Heap_Sort_Max_Heap(arr2)
    print("Sorted (Ascending):", arr2)

    arr3 = [64, 34, 25, 12, 22, 11, 90]
    print("\nOriginal:", arr3)
    solution.Heap_Sort_Max_Heap(arr3)
    print("Sorted (Ascending):", arr3)

    arr4 = [5, 2, 8, 1, 9]
    print("\nOriginal:", arr4)
    solution.Heap_Sort_Min_Heap(arr4)
    print("Sorted (Descending):", arr4)


if __name__ == "__main__":
    Test_Heap_Sort()
