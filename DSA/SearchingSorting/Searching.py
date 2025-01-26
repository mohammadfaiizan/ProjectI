import math
class SearchAlgorithms:
    
    def linear_search(self, arr, target):
        """Linear Search"""
        # Time complexity: O(n)
        # Space complexity: O(1)
        # Data type: Unsorted
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1

    def sentinel_linear_search(self, arr, target):
        """Sentinel Linear Search"""
        # Time complexity: O(n)
        # Space complexity: O(1)
        # Data type: Unsorted
        n = len(arr)
        last = arr[n - 1]
        arr[n - 1] = target
        i = 0
        while arr[i] != target:
            i += 1
        arr[n - 1] = last
        if i < n - 1 or arr[n - 1] == target:
            return i
        return -1

    def binary_search(self, arr, target):
        """Binary Search"""
        # Time complexity: O(log n)
        # Space complexity: O(1)
        # Data type: Sorted
        low, high = 0, len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1

    def one_sided_binary_search(self, arr, target):
        """Meta Binary Search (One-Sided Binary Search)"""
        # Time complexity: O(log n)
        # Space complexity: O(1)
        # Data type: Sorted     
        n = len(arr)
        lg = int(math.log2(n-1)) + 1;
        pos = 0
        for i in range(lg - 1, -1, -1) :
            if (arr[pos] == target):
                return pos
            
            new_pos = pos | (1 << i)

            if ((new_pos < n) and
                (arr[new_pos] <= target)):
                pos = new_pos
    
        return (pos if(arr[pos] == target) else -1)
 

    def ternary_search(self, arr, target):
        """Ternary Search"""
        # Time complexity: O(log3 n)
        # Space complexity: O(1)
        # Data type: Sorted
        low, high = 0, len(arr) - 1
        while high >= low:
            mid1 = low + (high - low) // 3
            mid2 = high - (high - low) // 3
            if arr[mid1] == target:
                return mid1
            if arr[mid2] == target:
                return mid2
            if target < arr[mid1]:
                high = mid1 - 1
            elif target > arr[mid2]:
                low = mid2 + 1
            else:
                low = mid1 + 1
                high = mid2 - 1
        return -1

    def jump_search(self, arr, target):
        """Jump Search"""
        # Time complexity: O(√n)
        # Space complexity: O(1)
        # Data type: Sorted
        n = len(arr)
        step = int(n ** 0.5)
        prev = 0
        while arr[min(step, n) - 1] < target:
            prev = step
            step += int(n ** 0.5)
            if prev >= n:
                return -1
        for i in range(prev, min(step, n)):
            if arr[i] == target:
                return i
        return -1

    def interpolation_search(self, arr, target):
        """Interpolation Search"""
        # Time complexity: O(log n) in the best case, O(n) in the worst case
        # Space complexity: O(1)
        # Data type: Sorted
        low, high = 0, len(arr) - 1
        while low <= high and target >= arr[low] and target <= arr[high]:
            if low == high:
                if arr[low] == target:
                    return low
                return -1
            pos = low + ((target - arr[low]) * (high - low)) // (arr[high] - arr[low])
            if arr[pos] == target:
                return pos
            if arr[pos] < target:
                low = pos + 1
            else:
                high = pos - 1
        return -1

    def exponential_search(self, arr, target):
        """Exponential Search"""
        # Time complexity: O(log n)
        # Space complexity: O(1)
        # Data type: Sorted
        if arr[0] == target:
            return 0
        n = len(arr)
        index = 1
        while index < n and arr[index] <= target:
            index *= 2
        return self.binary_search(arr[min(index, n-1):], target)

    def fibonacci_search(self, arr, target):
        """Fibonacci Search"""
        # Time complexity: O(log n)
        # Space complexity: O(1)
        # Data type: Sorted
        n = len(arr)
        fib_m_2 = 0
        fib_m_1 = 1
        fib = fib_m_2 + fib_m_1
        while fib < n:
            fib_m_2 = fib_m_1
            fib_m_1 = fib
            fib = fib_m_2 + fib_m_1
        offset = -1
        while fib > 1:
            i = min(offset + fib_m_2, n-1)
            if arr[i] < target:
                fib = fib_m_1
                fib_m_1 = fib_m_2
                fib_m_2 = fib - fib_m_1
                offset = i
            elif arr[i] > target:
                fib = fib_m_2
                fib_m_1 -= fib_m_2
                fib_m_2 = fib - fib_m_1
            else:
                return i
        if fib_m_1 and arr[offset + 1] == target:
            return offset + 1
        return -1

    def ubiquitous_binary_search(self, arr, target):
        """The Ubiquitous Binary Search"""
        # Time complexity: O(log n)
        # Space complexity: O(1)
        # Data type: Sorted
        return self.binary_search(arr, target)




def main():
    # Example array
    arr_sorted = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    arr_unsorted = [10, 3, 7, 1, 15, 9, 17, 5, 13, 19]
    target = 7

    # Create SearchAlgorithms object
    search_algorithms = SearchAlgorithms()

    # Linear Search
    print("Linear Search:")
    result = search_algorithms.linear_search(arr_unsorted, target)
    print(f"Found target {target} at index: {result}")
    
    # Sentinel Linear Search
    print("\nSentinel Linear Search:")
    result = search_algorithms.sentinel_linear_search(arr_unsorted, target)
    print(f"Found target {target} at index: {result}")

    # Binary Search (on sorted array)
    print("\nBinary Search:")
    result = search_algorithms.binary_search(arr_sorted, target)
    print(f"Found target {target} at index: {result}")
    
    # Meta Binary Search (One-Sided Binary Search)
    print("\nMeta Binary Search (One-Sided Binary Search):")
    result = search_algorithms.one_sided_binary_search(arr_sorted, target)
    print(f"Found target {target} at index: {result}")

    # Ternary Search
    print("\nTernary Search:")
    result = search_algorithms.ternary_search(arr_sorted, target)
    print(f"Found target {target} at index: {result}")

    # Jump Search
    print("\nJump Search:")
    result = search_algorithms.jump_search(arr_sorted, target)
    print(f"Found target {target} at index: {result}")

    # Interpolation Search
    print("\nInterpolation Search:")
    result = search_algorithms.interpolation_search(arr_sorted, target)
    print(f"Found target {target} at index: {result}")

    # Exponential Search
    print("\nExponential Search:")
    result = search_algorithms.exponential_search(arr_sorted, target)
    print(f"Found target {target} at index: {result}")

    # Fibonacci Search
    print("\nFibonacci Search:")
    result = search_algorithms.fibonacci_search(arr_sorted, target)
    print(f"Found target {target} at index: {result}")

    # The Ubiquitous Binary Search
    print("\nThe Ubiquitous Binary Search:")
    result = search_algorithms.ubiquitous_binary_search(arr_sorted, target)
    print(f"Found target {target} at index: {result}")

if __name__ == "__main__":
    main()
