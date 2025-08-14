"""
Non-Comparison Based Sorting Algorithms
======================================

Topics: Counting, Radix, Bucket, Pigeonhole Sort
Companies: Google, Facebook, Amazon, Microsoft
Difficulty: Medium
"""

from typing import List
import math

class NonComparisonBasedSorting:
    
    # ==========================================
    # 1. COUNTING SORT
    # ==========================================
    
    def counting_sort(self, arr: List[int], k: int = None) -> List[int]:
        """Counting Sort - Sort integers in range [0, k]
        Time: O(n + k), Space: O(k)
        Stable: Yes, In-place: No
        """
        if not arr:
            return arr
        
        # Find the range if not provided
        if k is None:
            k = max(arr)
        
        # Count occurrences
        count = [0] * (k + 1)
        for num in arr:
            count[num] += 1
        
        # Build result array
        result = []
        for i in range(k + 1):
            result.extend([i] * count[i])
        
        return result
    
    def counting_sort_stable(self, arr: List[int], k: int = None) -> List[int]:
        """Stable version of counting sort
        Time: O(n + k), Space: O(n + k)
        """
        if not arr:
            return arr
        
        if k is None:
            k = max(arr)
        
        # Count occurrences
        count = [0] * (k + 1)
        for num in arr:
            count[num] += 1
        
        # Transform count array to store actual positions
        for i in range(1, k + 1):
            count[i] += count[i - 1]
        
        # Build result array from right to left (for stability)
        result = [0] * len(arr)
        for i in range(len(arr) - 1, -1, -1):
            result[count[arr[i]] - 1] = arr[i]
            count[arr[i]] -= 1
        
        return result
    
    def counting_sort_with_range(self, arr: List[int]) -> List[int]:
        """Counting sort for arrays with negative numbers
        Time: O(n + k), Space: O(k)
        """
        if not arr:
            return arr
        
        min_val = min(arr)
        max_val = max(arr)
        range_val = max_val - min_val + 1
        
        # Count occurrences
        count = [0] * range_val
        for num in arr:
            count[num - min_val] += 1
        
        # Build result
        result = []
        for i in range(range_val):
            result.extend([i + min_val] * count[i])
        
        return result
    
    # ==========================================
    # 2. RADIX SORT
    # ==========================================
    
    def radix_sort(self, arr: List[int], base: int = 10) -> List[int]:
        """Radix Sort - Sort by processing each digit
        Time: O(d * (n + b)) where d is digits, b is base
        Space: O(n + b), Stable: Yes
        """
        if not arr:
            return arr
        
        # Find maximum number to know number of digits
        max_num = max(arr)
        
        # Apply counting sort for every digit
        exp = 1
        while max_num // exp > 0:
            arr = self._counting_sort_by_digit(arr, exp, base)
            exp *= base
        
        return arr
    
    def _counting_sort_by_digit(self, arr: List[int], exp: int, base: int) -> List[int]:
        """Counting sort based on digit represented by exp"""
        n = len(arr)
        result = [0] * n
        count = [0] * base
        
        # Count occurrences of each digit
        for num in arr:
            digit = (num // exp) % base
            count[digit] += 1
        
        # Change count[i] to actual position
        for i in range(1, base):
            count[i] += count[i - 1]
        
        # Build result array
        for i in range(n - 1, -1, -1):
            digit = (arr[i] // exp) % base
            result[count[digit] - 1] = arr[i]
            count[digit] -= 1
        
        return result
    
    def radix_sort_strings(self, strings: List[str]) -> List[str]:
        """Radix sort for strings of equal length
        Time: O(d * n) where d is string length
        """
        if not strings or not strings[0]:
            return strings
        
        max_len = len(strings[0])
        
        # Sort by each character position from right to left
        for pos in range(max_len - 1, -1, -1):
            strings = self._counting_sort_by_char(strings, pos)
        
        return strings
    
    def _counting_sort_by_char(self, strings: List[str], pos: int) -> List[str]:
        """Counting sort by character at position pos"""
        n = len(strings)
        result = [""] * n
        count = [0] * 256  # ASCII characters
        
        # Count occurrences
        for s in strings:
            count[ord(s[pos])] += 1
        
        # Transform count array
        for i in range(1, 256):
            count[i] += count[i - 1]
        
        # Build result array
        for i in range(n - 1, -1, -1):
            char_code = ord(strings[i][pos])
            result[count[char_code] - 1] = strings[i]
            count[char_code] -= 1
        
        return result
    
    # ==========================================
    # 3. BUCKET SORT
    # ==========================================
    
    def bucket_sort(self, arr: List[float], num_buckets: int = None) -> List[float]:
        """Bucket Sort - Distribute elements into buckets
        Time: O(n + k) average, O(n²) worst, Space: O(n)
        Works well for uniformly distributed data
        """
        if not arr:
            return arr
        
        if num_buckets is None:
            num_buckets = len(arr)
        
        # Create empty buckets
        buckets = [[] for _ in range(num_buckets)]
        
        # Distribute elements into buckets
        max_val = max(arr)
        min_val = min(arr)
        range_val = max_val - min_val
        
        for num in arr:
            if range_val == 0:
                bucket_index = 0
            else:
                bucket_index = int((num - min_val) / range_val * (num_buckets - 1))
            buckets[bucket_index].append(num)
        
        # Sort individual buckets and concatenate
        result = []
        for bucket in buckets:
            bucket.sort()  # Can use any sorting algorithm
            result.extend(bucket)
        
        return result
    
    def bucket_sort_integers(self, arr: List[int], bucket_size: int = 5) -> List[int]:
        """Bucket sort for integers with specified bucket size
        Time: O(n + k), Space: O(n)
        """
        if not arr:
            return arr
        
        min_val = min(arr)
        max_val = max(arr)
        bucket_count = (max_val - min_val) // bucket_size + 1
        
        # Create buckets
        buckets = [[] for _ in range(bucket_count)]
        
        # Distribute elements
        for num in arr:
            bucket_index = (num - min_val) // bucket_size
            buckets[bucket_index].append(num)
        
        # Sort buckets and combine
        result = []
        for bucket in buckets:
            bucket.sort()
            result.extend(bucket)
        
        return result
    
    # ==========================================
    # 4. PIGEONHOLE SORT
    # ==========================================
    
    def pigeonhole_sort(self, arr: List[int]) -> List[int]:
        """Pigeonhole Sort - Similar to counting sort
        Time: O(n + range), Space: O(range)
        Efficient when range is not significantly greater than n
        """
        if not arr:
            return arr
        
        min_val = min(arr)
        max_val = max(arr)
        size = max_val - min_val + 1
        
        # Create pigeonholes
        holes = [[] for _ in range(size)]
        
        # Put elements in pigeonholes
        for num in arr:
            holes[num - min_val].append(num)
        
        # Collect elements from pigeonholes
        result = []
        for hole in holes:
            result.extend(hole)
        
        return result
    
    # ==========================================
    # 5. SPECIALIZED SORTING ALGORITHMS
    # ==========================================
    
    def flash_sort(self, arr: List[int]) -> List[int]:
        """Flash Sort - Distribution-based sorting
        Time: O(n) average, Space: O(n)
        """
        if len(arr) <= 1:
            return arr.copy()
        
        arr = arr.copy()
        n = len(arr)
        m = int(0.45 * n)  # Number of classes
        
        # Find min and max
        min_val = min(arr)
        max_val = max(arr)
        
        if min_val == max_val:
            return arr
        
        # Classify elements
        L = [0] * m
        c = (m - 1) / (max_val - min_val)
        
        for i in range(n):
            k = int(c * (arr[i] - min_val))
            L[k] += 1
        
        # Accumulate
        for k in range(1, m):
            L[k] += L[k - 1]
        
        # Permute
        hold = arr[0]
        j = 0
        k1 = int(c * (hold - min_val))
        
        while j < n:
            while j >= L[k1]:
                j += 1
                if j < n:
                    hold = arr[j]
                    k1 = int(c * (hold - min_val))
            
            if j < n:
                flash = hold
                while j != L[k1]:
                    k1 = int(c * (flash - min_val))
                    hold = arr[L[k1] - 1]
                    arr[L[k1] - 1] = flash
                    L[k1] -= 1
                    flash = hold
        
        # Insertion sort for final cleanup
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        
        return arr
    
    def bead_sort(self, arr: List[int]) -> List[int]:
        """Bead Sort (Gravity Sort) - Natural sorting algorithm
        Time: O(n * max(arr)), Space: O(n * max(arr))
        Only works for positive integers
        """
        if not arr or min(arr) < 0:
            return arr
        
        max_val = max(arr)
        n = len(arr)
        
        # Create bead matrix
        beads = [[0] * max_val for _ in range(n)]
        
        # Place beads
        for i, num in enumerate(arr):
            for j in range(num):
                beads[i][j] = 1
        
        # Let gravity work (beads fall down)
        for j in range(max_val):
            sum_col = sum(beads[i][j] for i in range(n))
            for i in range(n):
                beads[i][j] = 1 if i >= n - sum_col else 0
        
        # Count beads in each row
        result = []
        for i in range(n):
            count = sum(beads[i])
            result.append(count)
        
        return result

# Test Examples
def run_examples():
    ncbs = NonComparisonBasedSorting()
    
    print("=== NON-COMPARISON BASED SORTING ===\n")
    
    # Test arrays
    integers = [4, 2, 2, 8, 3, 3, 1, 5, 7, 6]
    floats = [0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68]
    strings = ["abc", "def", "ghi", "jkl", "mno"]
    
    print("1. COUNTING SORT:")
    print("Original:", integers)
    print("Counting Sort:", ncbs.counting_sort(integers))
    print("Stable Counting Sort:", ncbs.counting_sort_stable(integers))
    
    print("\n2. RADIX SORT:")
    large_numbers = [170, 45, 75, 90, 2, 802, 24, 66]
    print("Original:", large_numbers)
    print("Radix Sort:", ncbs.radix_sort(large_numbers))
    
    print("\n3. BUCKET SORT:")
    print("Original floats:", floats)
    print("Bucket Sort:", ncbs.bucket_sort(floats))
    print("Bucket Sort integers:", ncbs.bucket_sort_integers(integers))
    
    print("\n4. PIGEONHOLE SORT:")
    small_range = [8, 3, 2, 7, 4, 6, 8]
    print("Original:", small_range)
    print("Pigeonhole Sort:", ncbs.pigeonhole_sort(small_range))
    
    print("\n5. SPECIALIZED SORTS:")
    print("Flash Sort:", ncbs.flash_sort(integers))
    print("Bead Sort:", ncbs.bead_sort([5, 3, 1, 7, 4, 1, 1, 3]))

if __name__ == "__main__":
    run_examples() 