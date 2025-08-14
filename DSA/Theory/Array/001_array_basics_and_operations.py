"""
Array Basics and Operations - Fundamental Concepts
=================================================

Topics: Basic operations, traversal, insertion, deletion, rotation
Companies: All major tech companies use these fundamentals
Difficulty: Easy to Medium
Time Complexity: Various (O(1) to O(n))
Space Complexity: O(1) to O(n)
"""

from typing import List, Optional, Tuple, Any
import random

class ArrayBasicsOperations:
    
    def __init__(self):
        """Initialize with sample data for demonstrations"""
        self.sample_array = [1, 2, 3, 4, 5]
    
    # ==========================================
    # 1. BASIC ARRAY OPERATIONS
    # ==========================================
    
    def traverse_array(self, arr: List[int]) -> None:
        """Traverse array and print elements
        Time: O(n), Space: O(1)
        """
        print("Forward traversal:", end=" ")
        for i in range(len(arr)):
            print(arr[i], end=" ")
        print()
        
        print("Backward traversal:", end=" ")
        for i in range(len(arr) - 1, -1, -1):
            print(arr[i], end=" ")
        print()
    
    def insert_element(self, arr: List[int], element: int, position: int) -> List[int]:
        """Insert element at specific position
        Time: O(n), Space: O(1)
        """
        if position < 0 or position > len(arr):
            raise IndexError("Position out of bounds")
        
        arr.insert(position, element)
        return arr
    
    def delete_element(self, arr: List[int], element: int) -> List[int]:
        """Delete first occurrence of element
        Time: O(n), Space: O(1)
        """
        try:
            arr.remove(element)
        except ValueError:
            print(f"Element {element} not found")
        return arr
    
    def delete_at_index(self, arr: List[int], index: int) -> List[int]:
        """Delete element at specific index
        Time: O(n), Space: O(1)
        """
        if 0 <= index < len(arr):
            return arr[:index] + arr[index + 1:]
        raise IndexError("Index out of bounds")
    
    def search_element(self, arr: List[int], target: int) -> int:
        """Linear search for element
        Time: O(n), Space: O(1)
        """
        for i, element in enumerate(arr):
            if element == target:
                return i
        return -1
    
    def update_element(self, arr: List[int], index: int, new_value: int) -> List[int]:
        """Update element at specific index
        Time: O(1), Space: O(1)
        """
        if 0 <= index < len(arr):
            arr[index] = new_value
        else:
            raise IndexError("Index out of bounds")
        return arr
    
    # ==========================================
    # 2. ARRAY ROTATION OPERATIONS
    # ==========================================
    
    def rotate_left(self, arr: List[int], k: int) -> List[int]:
        """Rotate array left by k positions
        Time: O(n), Space: O(1)
        """
        if not arr:
            return arr
        
        n = len(arr)
        k = k % n  # Handle k > n
        
        # Using reversal method
        def reverse(start: int, end: int):
            while start < end:
                arr[start], arr[end] = arr[end], arr[start]
                start += 1
                end -= 1
        
        reverse(0, k - 1)      # Reverse first k elements
        reverse(k, n - 1)      # Reverse remaining elements
        reverse(0, n - 1)      # Reverse entire array
        
        return arr
    
    def rotate_right(self, arr: List[int], k: int) -> List[int]:
        """Rotate array right by k positions
        Time: O(n), Space: O(1)
        """
        if not arr:
            return arr
        
        n = len(arr)
        k = k % n
        
        # Right rotation = Left rotation by (n - k)
        return self.rotate_left(arr, n - k)
    
    def cyclic_rotate_right(self, arr: List[int]) -> List[int]:
        """Rotate array right by one position
        Time: O(n), Space: O(1)
        """
        if len(arr) <= 1:
            return arr
        
        last_element = arr[-1]
        for i in range(len(arr) - 1, 0, -1):
            arr[i] = arr[i - 1]
        arr[0] = last_element
        
        return arr
    
    # ==========================================
    # 3. ARRAY REVERSAL OPERATIONS
    # ==========================================
    
    def reverse_array(self, arr: List[int]) -> List[int]:
        """Reverse entire array in-place
        Time: O(n), Space: O(1)
        """
        left, right = 0, len(arr) - 1
        while left < right:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1
        return arr
    
    def reverse_subarray(self, arr: List[int], start: int, end: int) -> List[int]:
        """Reverse subarray from start to end
        Time: O(n), Space: O(1)
        """
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
        return arr
    
    # ==========================================
    # 4. ARRAY STATISTICS AND ANALYSIS
    # ==========================================
    
    def find_min_max(self, arr: List[int]) -> Tuple[int, int]:
        """Find minimum and maximum elements
        Time: O(n), Space: O(1)
        """
        if not arr:
            raise ValueError("Array is empty")
        
        min_val = max_val = arr[0]
        for element in arr[1:]:
            if element < min_val:
                min_val = element
            elif element > max_val:
                max_val = element
        
        return min_val, max_val
    
    def find_second_largest(self, arr: List[int]) -> Optional[int]:
        """Find second largest element
        Time: O(n), Space: O(1)
        """
        if len(arr) < 2:
            return None
        
        first = second = float('-inf')
        
        for num in arr:
            if num > first:
                second = first
                first = num
            elif num > second and num != first:
                second = num
        
        return second if second != float('-inf') else None
    
    def find_kth_largest(self, arr: List[int], k: int) -> int:
        """Find kth largest element using quickselect
        Time: O(n) average, O(n²) worst, Space: O(1)
        """
        def quickselect(left: int, right: int, k_smallest: int) -> int:
            if left == right:
                return arr[left]
            
            # Choose random pivot
            pivot_index = random.randint(left, right)
            pivot_index = partition(left, right, pivot_index)
            
            if k_smallest == pivot_index:
                return arr[k_smallest]
            elif k_smallest < pivot_index:
                return quickselect(left, pivot_index - 1, k_smallest)
            else:
                return quickselect(pivot_index + 1, right, k_smallest)
        
        def partition(left: int, right: int, pivot_index: int) -> int:
            pivot_value = arr[pivot_index]
            arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
            
            store_index = left
            for i in range(left, right):
                if arr[i] < pivot_value:
                    arr[store_index], arr[i] = arr[i], arr[store_index]
                    store_index += 1
            
            arr[right], arr[store_index] = arr[store_index], arr[right]
            return store_index
        
        return quickselect(0, len(arr) - 1, len(arr) - k)
    
    def count_frequency(self, arr: List[int]) -> dict:
        """Count frequency of each element
        Time: O(n), Space: O(n)
        """
        frequency = {}
        for element in arr:
            frequency[element] = frequency.get(element, 0) + 1
        return frequency
    
    # ==========================================
    # 5. ARRAY DUPLICATES AND UNIQUENESS
    # ==========================================
    
    def remove_duplicates_sorted(self, arr: List[int]) -> int:
        """Remove duplicates from sorted array in-place
        Time: O(n), Space: O(1)
        Returns new length
        """
        if not arr:
            return 0
        
        write_index = 1
        for read_index in range(1, len(arr)):
            if arr[read_index] != arr[read_index - 1]:
                arr[write_index] = arr[read_index]
                write_index += 1
        
        return write_index
    
    def remove_duplicates_unsorted(self, arr: List[int]) -> List[int]:
        """Remove duplicates from unsorted array
        Time: O(n), Space: O(n)
        """
        seen = set()
        result = []
        
        for element in arr:
            if element not in seen:
                seen.add(element)
                result.append(element)
        
        return result
    
    def find_duplicates(self, arr: List[int]) -> List[int]:
        """Find all duplicate elements
        Time: O(n), Space: O(n)
        """
        seen = set()
        duplicates = set()
        
        for element in arr:
            if element in seen:
                duplicates.add(element)
            else:
                seen.add(element)
        
        return list(duplicates)
    
    def has_duplicates(self, arr: List[int]) -> bool:
        """Check if array has duplicates
        Time: O(n), Space: O(n)
        """
        return len(arr) != len(set(arr))
    
    # ==========================================
    # 6. ARRAY REARRANGEMENT
    # ==========================================
    
    def rearrange_positive_negative(self, arr: List[int]) -> List[int]:
        """Rearrange positive and negative numbers alternately
        Time: O(n), Space: O(n)
        """
        positive = [x for x in arr if x >= 0]
        negative = [x for x in arr if x < 0]
        
        result = []
        min_len = min(len(positive), len(negative))
        
        for i in range(min_len):
            result.extend([positive[i], negative[i]])
        
        # Add remaining elements
        result.extend(positive[min_len:])
        result.extend(negative[min_len:])
        
        return result
    
    def segregate_even_odd(self, arr: List[int]) -> List[int]:
        """Segregate even and odd numbers
        Time: O(n), Space: O(1)
        """
        left = 0
        right = len(arr) - 1
        
        while left < right:
            while left < right and arr[left] % 2 == 0:
                left += 1
            while left < right and arr[right] % 2 == 1:
                right -= 1
            
            if left < right:
                arr[left], arr[right] = arr[right], arr[left]
                left += 1
                right -= 1
        
        return arr
    
    def move_zeros_to_end(self, arr: List[int]) -> List[int]:
        """Move all zeros to end while maintaining order
        Time: O(n), Space: O(1)
        """
        write_index = 0
        
        # Move non-zero elements to front
        for read_index in range(len(arr)):
            if arr[read_index] != 0:
                arr[write_index] = arr[read_index]
                write_index += 1
        
        # Fill remaining positions with zeros
        while write_index < len(arr):
            arr[write_index] = 0
            write_index += 1
        
        return arr
    
    # ==========================================
    # 7. ARRAY UTILITY FUNCTIONS
    # ==========================================
    
    def is_sorted(self, arr: List[int], ascending: bool = True) -> bool:
        """Check if array is sorted
        Time: O(n), Space: O(1)
        """
        if len(arr) <= 1:
            return True
        
        if ascending:
            return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
        else:
            return all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))
    
    def array_sum(self, arr: List[int]) -> int:
        """Calculate sum of array elements
        Time: O(n), Space: O(1)
        """
        return sum(arr)
    
    def array_product(self, arr: List[int]) -> int:
        """Calculate product of array elements
        Time: O(n), Space: O(1)
        """
        result = 1
        for element in arr:
            result *= element
        return result
    
    def array_average(self, arr: List[int]) -> float:
        """Calculate average of array elements
        Time: O(n), Space: O(1)
        """
        if not arr:
            return 0.0
        return sum(arr) / len(arr)
    
    def flatten_2d_array(self, arr_2d: List[List[int]]) -> List[int]:
        """Flatten 2D array to 1D
        Time: O(m*n), Space: O(m*n)
        """
        result = []
        for row in arr_2d:
            result.extend(row)
        return result
    
    def chunk_array(self, arr: List[int], chunk_size: int) -> List[List[int]]:
        """Split array into chunks of specified size
        Time: O(n), Space: O(n)
        """
        chunks = []
        for i in range(0, len(arr), chunk_size):
            chunks.append(arr[i:i + chunk_size])
        return chunks

# Test Examples and Demonstrations
def run_examples():
    abo = ArrayBasicsOperations()
    
    print("=== ARRAY BASICS AND OPERATIONS EXAMPLES ===\n")
    
    # Basic operations
    print("1. BASIC OPERATIONS:")
    arr = [1, 2, 3, 4, 5]
    print(f"Original array: {arr}")
    abo.traverse_array(arr)
    
    # Insertion and deletion
    print("\n2. INSERTION AND DELETION:")
    arr_copy = arr.copy()
    abo.insert_element(arr_copy, 10, 2)
    print(f"After inserting 10 at position 2: {arr_copy}")
    
    abo.delete_element(arr_copy, 10)
    print(f"After deleting 10: {arr_copy}")
    
    # Rotation
    print("\n3. ROTATION:")
    arr = [1, 2, 3, 4, 5]
    print(f"Original: {arr}")
    abo.rotate_left(arr.copy(), 2)
    print(f"Rotate left by 2: {abo.rotate_left(arr.copy(), 2)}")
    print(f"Rotate right by 2: {abo.rotate_right(arr.copy(), 2)}")
    
    # Statistics
    print("\n4. STATISTICS:")
    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    print(f"Array: {arr}")
    min_val, max_val = abo.find_min_max(arr)
    print(f"Min: {min_val}, Max: {max_val}")
    print(f"Second largest: {abo.find_second_largest(arr)}")
    print(f"3rd largest: {abo.find_kth_largest(arr.copy(), 3)}")
    print(f"Frequency: {abo.count_frequency(arr)}")
    
    # Duplicates
    print("\n5. DUPLICATES:")
    arr_with_dups = [1, 2, 2, 3, 4, 4, 5]
    print(f"Array with duplicates: {arr_with_dups}")
    print(f"Duplicates found: {abo.find_duplicates(arr_with_dups)}")
    print(f"Has duplicates: {abo.has_duplicates(arr_with_dups)}")
    
    # Rearrangement
    print("\n6. REARRANGEMENT:")
    arr_mixed = [1, -2, 3, -4, 5, -6]
    print(f"Mixed array: {arr_mixed}")
    print(f"Rearranged pos/neg: {abo.rearrange_positive_negative(arr_mixed)}")
    
    arr_even_odd = [1, 2, 3, 4, 5, 6]
    print(f"Even/odd segregation: {abo.segregate_even_odd(arr_even_odd.copy())}")
    
    arr_with_zeros = [1, 0, 2, 0, 3, 4]
    print(f"Move zeros to end: {abo.move_zeros_to_end(arr_with_zeros)}")

if __name__ == "__main__":
    run_examples() 