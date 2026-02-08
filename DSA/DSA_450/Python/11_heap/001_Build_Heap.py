"""
Problem: Build Max Heap and Min Heap
URL: https://www.geeksforgeeks.org/building-heap-from-array/

Problem Statement:
Build a max heap and min heap from an array using heapify (top-down recursive). Show both max heap and min heap construction.

Sample Input/Output:
Input: [1,3,5,4,6,13,10,9,8,15,17]
Output: Max Heap: [17,15,13,9,6,5,10,4,8,3,1]
        Min Heap: [1,3,5,4,6,13,10,9,8,15,17]
"""


class Solution:
    def Build_Max_Heap_Recursive(self, arr):
        """
        Build Max Heap Recursive
        Time Complexity: O(n)
        Space Complexity: O(log n) for recursion stack
        """
        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            self.Max_Heapify(arr, n, i)

    def Build_Min_Heap_Recursive(self, arr):
        """
        Build Min Heap Recursive
        Time Complexity: O(n)
        Space Complexity: O(log n) for recursion stack
        """
        n = len(arr)
        for i in range(n // 2 - 1, -1, -1):
            self.Min_Heapify(arr, n, i)

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


def Test_Build_Heap():
    solution = Solution()

    arr1 = [1, 3, 5, 4, 6, 13, 10, 9, 8, 15, 17]
    arr2 = arr1.copy()

    print("Original array:", arr1)

    solution.Build_Max_Heap_Recursive(arr1)
    print("Max Heap:", arr1)

    solution.Build_Min_Heap_Recursive(arr2)
    print("Min Heap:", arr2)

    arr3 = [10, 20, 15, 30, 40]
    arr4 = arr3.copy()

    print("\nOriginal array:", arr3)

    solution.Build_Max_Heap_Recursive(arr3)
    print("Max Heap:", arr3)

    solution.Build_Min_Heap_Recursive(arr4)
    print("Min Heap:", arr4)


if __name__ == "__main__":
    Test_Build_Heap()
