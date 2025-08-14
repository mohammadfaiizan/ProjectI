class ArrayOperations:
    @staticmethod
    def traverse_array(arr):
        """
        Traverse an array and return its elements.
        """
        return arr

    @staticmethod
    def reverse_array(arr):
        """
        Reverse the array.
        """
        reversed_array = []
        for i in range(len(arr)-1, -1, -1):
            reversed_array.append(arr[i])
        return reversed_array

    @staticmethod
    def find_largest(arr):
        """
        Find the largest element in the array.
        """
        largest = arr[0]
        for num in arr:
            if num > largest:
                largest = num
        return largest

    @staticmethod
    def find_smallest(arr):
        """
        Find the smallest element in the array.
        """
        smallest = arr[0]
        for num in arr:
            if num < smallest:
                smallest = num
        return smallest

    @staticmethod
    def rotate_array(arr, k):
        """
        Rotate the array by k positions to the right.
        """
        n = len(arr)
        k = k % n  # Handle cases where k > len(arr)
        rotated_array = arr[-k:] + arr[:-k]
        return rotated_array

    @staticmethod
    def remove_duplicates_sorted(arr):
        """
        Remove duplicates from a sorted array.
        """
        unique = []
        for num in arr:
            if not unique or unique[-1] != num:
                unique.append(num)
        return unique

    @staticmethod
    def linear_search(arr, target):
        """
        Perform linear search to find the target element.
        """
        for i, num in enumerate(arr):
            if num == target:
                return i
        return -1

    @staticmethod
    def first_and_last_occurrence(arr, target):
        """
        Find the first and last occurrence of a target element in a sorted array.
        """
        first, last = -1, -1
        for i in range(len(arr)):
            if arr[i] == target:
                if first == -1:
                    first = i
                last = i
        return (first, last)

    @staticmethod
    def sum_of_elements(arr):
        """
        Find the sum of all elements in the array.
        """
        total = 0
        for num in arr:
            total += num
        return total

    @staticmethod
    def count_frequency(arr):
        """
        Count the frequency of each element in the array.
        """
        frequency = {}
        for num in arr:
            if num in frequency:
                frequency[num] += 1
            else:
                frequency[num] = 1
        return frequency

    @staticmethod
    def find_index(arr, target):
        """
        Find the index of a target element in the array.
        """
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1

# Example usage
if __name__ == "__main__":
    arr = [3, 1, 4, 1, 5, 9, 2]
    sorted_arr = [1, 1, 2, 3, 4, 5, 9]

    print("Traverse Array:", ArrayOperations.traverse_array(arr))
    print("Reverse Array:", ArrayOperations.reverse_array(arr))
    print("Largest Element:", ArrayOperations.find_largest(arr))
    print("Smallest Element:", ArrayOperations.find_smallest(arr))
    print("Rotate Array:", ArrayOperations.rotate_array(arr, 3))
    print("Remove Duplicates (Sorted):", ArrayOperations.remove_duplicates_sorted(sorted_arr))
    print("Linear Search:", ArrayOperations.linear_search(arr, 5))
    print("Binary Search:", ArrayOperations.binary_search(sorted_arr, 5))
    print("First and Last Occurrence:", ArrayOperations.first_and_last_occurrence(sorted_arr, 1))
    print("Bubble Sort:", ArrayOperations.bubble_sort(arr.copy()))
    print("Selection Sort:", ArrayOperations.selection_sort(arr.copy()))
    print("Insertion Sort:", ArrayOperations.insertion_sort(arr.copy()))
    print("Merge Sorted Arrays:", ArrayOperations.merge_sorted_arrays([1, 3, 5], [2, 4, 6]))
    print("Sum of Elements:", ArrayOperations.sum_of_elements(arr))
    print("Maximum Subarray Sum:", ArrayOperations.max_subarray_sum(arr))
    print("Subarray with Given Sum:", ArrayOperations.subarray_with_given_sum(arr, 9))
    print("Count Frequency:", ArrayOperations.count_frequency(arr))
    print("Find Index:", ArrayOperations.find_index(arr, 9))