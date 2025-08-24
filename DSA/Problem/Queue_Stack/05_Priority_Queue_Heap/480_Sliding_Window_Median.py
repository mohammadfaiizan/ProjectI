"""
480. Sliding Window Median - Multiple Approaches
Difficulty: Hard

The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle values.

Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the median array for each window in the original array.
"""

from typing import List
import heapq
from collections import defaultdict

class SlidingWindowMedian:
    """Multiple approaches to find sliding window median"""
    
    def medianSlidingWindow_two_heaps(self, nums: List[int], k: int) -> List[float]:
        """
        Approach 1: Two Heaps with Lazy Deletion (Optimal)
        
        Use two heaps with lazy deletion for sliding window.
        
        Time: O(n log k), Space: O(k)
        """
        def balance_heaps():
            """Balance the two heaps"""
            # Remove invalid elements from tops
            while small and small[0] in to_remove and to_remove[small[0]] > 0:
                to_remove[small[0]] -= 1
                heapq.heappop(small)
            
            while large and large[0] in to_remove and to_remove[large[0]] > 0:
                to_remove[large[0]] -= 1
                heapq.heappop(large)
            
            # Balance sizes
            if len(small) > len(large) + 1:
                val = -heapq.heappop(small)
                heapq.heappush(large, val)
            elif len(large) > len(small) + 1:
                val = heapq.heappop(large)
                heapq.heappush(small, -val)
        
        def get_median():
            """Get median from balanced heaps"""
            balance_heaps()
            
            if k % 2 == 1:
                return float(-small[0] if len(small) > len(large) else large[0])
            else:
                return (-small[0] + large[0]) / 2.0
        
        small = []  # Max heap (negated values)
        large = []  # Min heap
        to_remove = defaultdict(int)  # Lazy deletion counter
        result = []
        
        # Initialize first window
        for i in range(k):
            heapq.heappush(small, -nums[i])
        
        # Balance initial window
        for _ in range(k // 2):
            val = -heapq.heappop(small)
            heapq.heappush(large, val)
        
        result.append(get_median())
        
        # Process remaining elements
        for i in range(k, len(nums)):
            # Remove outgoing element (lazy deletion)
            outgoing = nums[i - k]
            to_remove[outgoing] += 1
            
            # Add incoming element
            incoming = nums[i]
            if incoming <= -small[0]:
                heapq.heappush(small, -incoming)
            else:
                heapq.heappush(large, incoming)
            
            balance_heaps()
            result.append(get_median())
        
        return result
    
    def medianSlidingWindow_sorting(self, nums: List[int], k: int) -> List[float]:
        """
        Approach 2: Sorting Each Window
        
        Sort each window to find median.
        
        Time: O(n * k log k), Space: O(k)
        """
        result = []
        
        for i in range(len(nums) - k + 1):
            window = sorted(nums[i:i + k])
            
            if k % 2 == 1:
                result.append(float(window[k // 2]))
            else:
                result.append((window[k // 2 - 1] + window[k // 2]) / 2.0)
        
        return result
    
    def medianSlidingWindow_insertion_sort(self, nums: List[int], k: int) -> List[float]:
        """
        Approach 3: Maintain Sorted Window with Insertion Sort
        
        Use insertion sort to maintain sorted window.
        
        Time: O(n * k), Space: O(k)
        """
        def insert_sorted(arr: List[int], val: int) -> None:
            """Insert value in sorted position"""
            left, right = 0, len(arr)
            
            while left < right:
                mid = (left + right) // 2
                if arr[mid] < val:
                    left = mid + 1
                else:
                    right = mid
            
            arr.insert(left, val)
        
        def remove_sorted(arr: List[int], val: int) -> None:
            """Remove value from sorted array"""
            idx = arr.index(val)
            arr.pop(idx)
        
        result = []
        window = sorted(nums[:k])
        
        # First window median
        if k % 2 == 1:
            result.append(float(window[k // 2]))
        else:
            result.append((window[k // 2 - 1] + window[k // 2]) / 2.0)
        
        # Slide window
        for i in range(k, len(nums)):
            # Remove outgoing element
            remove_sorted(window, nums[i - k])
            
            # Add incoming element
            insert_sorted(window, nums[i])
            
            # Calculate median
            if k % 2 == 1:
                result.append(float(window[k // 2]))
            else:
                result.append((window[k // 2 - 1] + window[k // 2]) / 2.0)
        
        return result
    
    def medianSlidingWindow_multiset_simulation(self, nums: List[int], k: int) -> List[float]:
        """
        Approach 4: Multiset Simulation
        
        Simulate multiset behavior with sorted list.
        
        Time: O(n * k), Space: O(k)
        """
        from bisect import bisect_left, insort
        
        result = []
        window = []
        
        # Initialize first window
        for i in range(k):
            insort(window, nums[i])
        
        # First median
        if k % 2 == 1:
            result.append(float(window[k // 2]))
        else:
            result.append((window[k // 2 - 1] + window[k // 2]) / 2.0)
        
        # Slide window
        for i in range(k, len(nums)):
            # Remove outgoing element
            outgoing = nums[i - k]
            idx = bisect_left(window, outgoing)
            window.pop(idx)
            
            # Add incoming element
            insort(window, nums[i])
            
            # Calculate median
            if k % 2 == 1:
                result.append(float(window[k // 2]))
            else:
                result.append((window[k // 2 - 1] + window[k // 2]) / 2.0)
        
        return result


def test_sliding_window_median():
    """Test sliding window median algorithms"""
    solver = SlidingWindowMedian()
    
    test_cases = [
        ([1,3,-1,-3,5,3,6,7], 3, [1.0,-1.0,-1.0,3.0,5.0,6.0], "Example 1"),
        ([1,2,3,4,2,3,1,4,2], 3, [2.0,3.0,3.0,3.0,2.0,3.0,2.0], "Example 2"),
        ([1,4,2,3], 4, [2.5], "Single window"),
        ([1,2], 1, [1.0,2.0], "k=1"),
        ([1,2,3,4], 2, [1.5,2.5,3.5], "k=2"),
        ([2147483647,2147483647], 2, [2147483647.0], "Large numbers"),
    ]
    
    algorithms = [
        ("Two Heaps", solver.medianSlidingWindow_two_heaps),
        ("Sorting", solver.medianSlidingWindow_sorting),
        ("Insertion Sort", solver.medianSlidingWindow_insertion_sort),
        ("Multiset Simulation", solver.medianSlidingWindow_multiset_simulation),
    ]
    
    print("=== Testing Sliding Window Median ===")
    
    for nums, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Array: {nums}")
        print(f"k: {k}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums[:], k)  # Pass copy
                
                # Compare with tolerance for floating point
                def arrays_equal(a, b, tol=1e-9):
                    if len(a) != len(b):
                        return False
                    return all(abs(x - y) < tol for x, y in zip(a, b))
                
                status = "✓" if arrays_equal(result, expected) else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_two_heaps_approach():
    """Demonstrate two heaps approach step by step"""
    print("\n=== Two Heaps Approach Step-by-Step Demo ===")
    
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    
    print(f"Array: {nums}")
    print(f"Window size: {k}")
    print("Strategy: Use two heaps with lazy deletion for sliding window")
    
    # Simplified demonstration (without full lazy deletion implementation)
    for i in range(len(nums) - k + 1):
        window = nums[i:i + k]
        sorted_window = sorted(window)
        
        print(f"\nWindow {i+1}: {window}")
        print(f"  Sorted: {sorted_window}")
        
        if k % 2 == 1:
            median = float(sorted_window[k // 2])
            print(f"  Median (odd k): {sorted_window[k // 2]} = {median}")
        else:
            left_mid = sorted_window[k // 2 - 1]
            right_mid = sorted_window[k // 2]
            median = (left_mid + right_mid) / 2.0
            print(f"  Median (even k): ({left_mid} + {right_mid}) / 2 = {median}")


if __name__ == "__main__":
    test_sliding_window_median()
    demonstrate_two_heaps_approach()

"""
Sliding Window Median demonstrates advanced heap applications for
streaming median calculation with sliding windows, including lazy deletion
techniques and multiple approaches for dynamic median tracking.
"""
