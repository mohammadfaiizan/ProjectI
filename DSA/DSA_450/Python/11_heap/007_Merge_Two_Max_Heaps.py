"""
Problem: Merge Two Binary Max Heaps
URL: https://practice.geeksforgeeks.org/problems/merge-two-binary-max-heap0144/1

Problem Statement:
Given two max heaps, merge them into a single max heap.

Sample Input/Output:
Input: heap1=[10,5,6,2], heap2=[12,7,9]
Output: Merged max heap
"""


class Solution:
    def Merge_Heaps_Rebuild(self, heap1, heap2):
        """
        Concatenate Arrays and Rebuild Heap
        Time Complexity: O(n+m)
        Space Complexity: O(n+m)
        """
        merged = []
        merged.extend(heap1)
        merged.extend(heap2)

        n = len(merged)
        for i in range(n // 2 - 1, -1, -1):
            self.Max_Heapify(merged, n, i)

        return merged

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


def Test_Merge_Heaps():
    solution = Solution()

    heap1 = [10, 5, 6, 2]
    heap2 = [12, 7, 9]

    print("Heap 1:", heap1)
    print("Heap 2:", heap2)

    merged = solution.Merge_Heaps_Rebuild(heap1, heap2)
    print("Merged Max Heap:", merged)

    heap3 = [20, 10, 15, 8, 5]
    heap4 = [25, 18, 12]

    print("\nHeap 1:", heap3)
    print("Heap 2:", heap4)

    merged2 = solution.Merge_Heaps_Rebuild(heap3, heap4)
    print("Merged Max Heap:", merged2)

    heap5 = [30]
    heap6 = [40, 35]

    print("\nHeap 1:", heap5)
    print("Heap 2:", heap6)

    merged3 = solution.Merge_Heaps_Rebuild(heap5, heap6)
    print("Merged Max Heap:", merged3)


if __name__ == "__main__":
    Test_Merge_Heaps()
