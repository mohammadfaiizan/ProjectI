import math

class SortingAlgorithms:
    
    # Bubble Sort
    def bubble_sort(self, arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    
    # Complexity of Bubble Sort:
    # Time Complexity: Best: O(n), Average: O(n^2), Worst: O(n^2)
    # Space Complexity: O(1) (in-place sorting)

    
    # Selection Sort
    def selection_sort(self, arr):
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i+1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return arr
    
    # Complexity of Selection Sort:
    # Time Complexity: Best: O(n^2), Average: O(n^2), Worst: O(n^2)
    # Space Complexity: O(1) (in-place sorting)


    # Insertion Sort
    def insertion_sort(self, arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
    
    # Complexity of Insertion Sort:
    # Time Complexity: Best: O(n), Average: O(n^2), Worst: O(n^2)
    # Space Complexity: O(1) (in-place sorting)


    # Merge Sort
    def merge_sort(self, arr):
        if len(arr) > 1:
            mid = len(arr) // 2
            left_half = arr[:mid]
            right_half = arr[mid:]

            self.merge_sort(left_half)
            self.merge_sort(right_half)

            i = j = k = 0
            while i < len(left_half) and j < len(right_half):
                if left_half[i] < right_half[j]:
                    arr[k] = left_half[i]
                    i += 1
                else:
                    arr[k] = right_half[j]
                    j += 1
                k += 1

            while i < len(left_half):
                arr[k] = left_half[i]
                i += 1
                k += 1

            while j < len(right_half):
                arr[k] = right_half[j]
                j += 1
                k += 1
        return arr

    # Complexity of Merge Sort:
    # Time Complexity: Best: O(n log n), Average: O(n log n), Worst: O(n log n)
    # Space Complexity: O(n) (Auxiliary space required for merging)


    # Quick Sort
    def quick_sort(self, arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return self.quick_sort(left) + middle + self.quick_sort(right)
    
    def quickSort_high(self,arr,low,high):
        # code here
        if low < high:
            pi = self.partition(arr, low, high)
            self.quickSort_high(arr, low, pi-1)
            self.quickSort_high(arr, pi+1, high)
    
    def partition(self,arr,low,high):
        # code here
        pivot = arr[high]
        i = low -1
        
        for j in range(low, high):
            if arr[j] < pivot:
                i += 1
                arr[i] , arr[j] = arr[j], arr[i]
        
        arr[i+1] , arr[high] = arr[high], arr[i+1]
        return i+1

    # Complexity of Quick Sort:
    # Time Complexity: Best: O(n log n), Average: O(n log n), Worst: O(n^2)
    # Space Complexity: O(log n) (Recursion stack space)


    # Heap Sort
    def heapify(self, arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, n, largest)

    def heap_sort(self, arr):
        n = len(arr)

        for i in range(n // 2 - 1, -1, -1):
            self.heapify(arr, n, i)

        for i in range(n-1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            self.heapify(arr, i, 0)
        return arr

    # Complexity of Heap Sort:
    # Time Complexity: Best: O(n log n), Average: O(n log n), Worst: O(n log n)
    # Space Complexity: O(1) (in-place sorting)


    # Shell Sort
    def shell_sort(self, arr):
        n = len(arr)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                temp = arr[i]
                j = i
                while j >= gap and arr[j - gap] > temp:
                    arr[j] = arr[j - gap]
                    j -= gap
                arr[j] = temp
            gap //= 2
        return arr

    # Complexity of Shell Sort:
    # Time Complexity: Best: O(n log n), Average: O(n^1.5), Worst: O(n^2)
    # Space Complexity: O(1) (in-place sorting)


    # Tim Sort (Hybrid of Merge Sort and Insertion Sort)
    def tim_sort(self, arr):
        min_run = 32
        n = len(arr)
        for i in range(0, n, min_run):
            self.insertion_sort(arr[i:i + min_run])
        
        size = min_run
        while size < n:
            for start in range(0, n, size * 2):
                mid = min(n, start + size)
                end = min(start + size * 2, n)
                merged = self.merge(arr[start:mid], arr[mid:end])
                arr[start:start + len(merged)] = merged
            size *= 2
        return arr
    
    def merge(self, left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    # Complexity of Tim Sort:
    # Time Complexity: Best: O(n), Average: O(n log n), Worst: O(n log n)
    # Space Complexity: O(n) (Auxiliary space for merging)

# Example Usage:
if __name__ == "__main__":
    sorting = SortingAlgorithms()
    
    arr = [64, 25, 12, 22, 11, 20, 55, 97, 33]
    print("Bubble Sort:", sorting.bubble_sort(arr.copy()))
    print("Selection Sort:", sorting.selection_sort(arr.copy()))
    print("Insertion Sort:", sorting.insertion_sort(arr.copy()))
    print("Merge Sort:", sorting.merge_sort(arr.copy()))
    print("Quick Sort:", sorting.quick_sort(arr.copy()))
    print("Heap Sort:", sorting.heap_sort(arr.copy()))
    print("Shell Sort:", sorting.shell_sort(arr.copy()))
    print("Tim Sort:", sorting.tim_sort(arr.copy()))
