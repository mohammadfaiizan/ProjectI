"""
215. Kth Largest Element in an Array - Multiple Approaches
Difficulty: Medium (but categorized as Easy in heap context)

Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

Can you solve it without sorting?
"""

from typing import List
import heapq
import random

class KthLargestElementInArray:
    """Multiple approaches to find kth largest element"""
    
    def findKthLargest_min_heap(self, nums: List[int], k: int) -> int:
        """
        Approach 1: Min Heap (Optimal for small k)
        
        Maintain min heap of size k.
        
        Time: O(n log k), Space: O(k)
        """
        heap = []
        
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)
        
        return heap[0]
    
    def findKthLargest_max_heap(self, nums: List[int], k: int) -> int:
        """
        Approach 2: Max Heap
        
        Build max heap and extract k elements.
        
        Time: O(n + k log n), Space: O(n)
        """
        # Build max heap (negate values)
        heap = [-num for num in nums]
        heapq.heapify(heap)
        
        # Extract k-1 elements
        for _ in range(k - 1):
            heapq.heappop(heap)
        
        return -heap[0]
    
    def findKthLargest_quickselect(self, nums: List[int], k: int) -> int:
        """
        Approach 3: QuickSelect (Optimal average case)
        
        Use quickselect algorithm for O(n) average time.
        
        Time: O(n) average, O(n²) worst, Space: O(1)
        """
        def quickselect(left: int, right: int, k_smallest: int) -> int:
            """Find k_smallest element in nums[left:right+1]"""
            if left == right:
                return nums[left]
            
            # Random pivot for better average performance
            pivot_idx = random.randint(left, right)
            nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
            
            # Partition
            pivot_idx = partition(left, right)
            
            if k_smallest == pivot_idx:
                return nums[k_smallest]
            elif k_smallest < pivot_idx:
                return quickselect(left, pivot_idx - 1, k_smallest)
            else:
                return quickselect(pivot_idx + 1, right, k_smallest)
        
        def partition(left: int, right: int) -> int:
            """Partition for descending order"""
            pivot = nums[right]
            i = left
            
            for j in range(left, right):
                if nums[j] >= pivot:  # Descending order
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
            
            nums[i], nums[right] = nums[right], nums[i]
            return i
        
        nums = nums[:]  # Make copy to avoid modifying original
        return quickselect(0, len(nums) - 1, k - 1)
    
    def findKthLargest_sorting(self, nums: List[int], k: int) -> int:
        """
        Approach 4: Sorting
        
        Sort array and return kth element.
        
        Time: O(n log n), Space: O(1)
        """
        nums.sort(reverse=True)
        return nums[k - 1]
    
    def findKthLargest_counting_sort(self, nums: List[int], k: int) -> int:
        """
        Approach 5: Counting Sort (for bounded input)
        
        Use counting sort when range is small.
        
        Time: O(n + range), Space: O(range)
        """
        min_val, max_val = min(nums), max(nums)
        
        # Count frequencies
        count = [0] * (max_val - min_val + 1)
        for num in nums:
            count[num - min_val] += 1
        
        # Find kth largest
        remaining = k
        for i in range(len(count) - 1, -1, -1):
            remaining -= count[i]
            if remaining <= 0:
                return i + min_val
        
        return -1  # Should not reach here
    
    def findKthLargest_bucket_sort(self, nums: List[int], k: int) -> int:
        """
        Approach 6: Bucket Sort
        
        Use bucket sort for better distribution.
        
        Time: O(n + k), Space: O(n)
        """
        min_val, max_val = min(nums), max(nums)
        
        if min_val == max_val:
            return min_val
        
        # Create buckets
        bucket_size = max(1, (max_val - min_val) // len(nums))
        bucket_count = (max_val - min_val) // bucket_size + 1
        buckets = [[] for _ in range(bucket_count)]
        
        # Distribute elements into buckets
        for num in nums:
            bucket_idx = (num - min_val) // bucket_size
            buckets[bucket_idx].append(num)
        
        # Sort buckets and find kth largest
        remaining = k
        for i in range(bucket_count - 1, -1, -1):
            if buckets[i]:
                buckets[i].sort(reverse=True)
                if remaining <= len(buckets[i]):
                    return buckets[i][remaining - 1]
                remaining -= len(buckets[i])
        
        return -1  # Should not reach here


def test_kth_largest_element_in_array():
    """Test kth largest element algorithms"""
    solver = KthLargestElementInArray()
    
    test_cases = [
        ([3,2,1,5,6,4], 2, 5, "Example 1"),
        ([3,2,3,1,2,4,5,5,6], 4, 4, "Example 2"),
        ([1], 1, 1, "Single element"),
        ([1,2], 1, 2, "Two elements, k=1"),
        ([1,2], 2, 1, "Two elements, k=2"),
        ([7,10,4,3,20,15], 3, 10, "Unsorted array"),
        ([1,1,1,1], 2, 1, "All same elements"),
        ([5,4,3,2,1], 3, 3, "Sorted descending"),
        ([1,2,3,4,5], 3, 3, "Sorted ascending"),
        ([-1,2,0], 1, 2, "With negative numbers"),
    ]
    
    algorithms = [
        ("Min Heap", solver.findKthLargest_min_heap),
        ("Max Heap", solver.findKthLargest_max_heap),
        ("QuickSelect", solver.findKthLargest_quickselect),
        ("Sorting", solver.findKthLargest_sorting),
        ("Counting Sort", solver.findKthLargest_counting_sort),
        ("Bucket Sort", solver.findKthLargest_bucket_sort),
    ]
    
    print("=== Testing Kth Largest Element in Array ===")
    
    for nums, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Array: {nums}")
        print(f"k: {k}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums[:], k)  # Pass copy
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:40]}")


def demonstrate_quickselect():
    """Demonstrate quickselect algorithm"""
    print("\n=== QuickSelect Algorithm Demonstration ===")
    
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    
    print(f"Array: {nums}")
    print(f"Finding {k}th largest element")
    print("QuickSelect partitions array around pivot")
    
    # Manual quickselect simulation
    arr = nums[:]
    target_idx = k - 1  # 0-indexed
    
    def partition_demo(arr, left, right):
        pivot = arr[right]
        print(f"  Pivot: {pivot}")
        
        i = left
        for j in range(left, right):
            if arr[j] >= pivot:  # Descending order
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        
        arr[i], arr[right] = arr[right], arr[i]
        print(f"  After partition: {arr}")
        print(f"  Pivot position: {i}")
        return i
    
    left, right = 0, len(arr) - 1
    step = 1
    
    while left <= right:
        print(f"\nStep {step}: Searching in {arr[left:right+1]}")
        
        pivot_idx = partition_demo(arr, left, right)
        
        if pivot_idx == target_idx:
            print(f"  Found! {k}th largest = {arr[pivot_idx]}")
            break
        elif target_idx < pivot_idx:
            print(f"  Target in left part")
            right = pivot_idx - 1
        else:
            print(f"  Target in right part")
            left = pivot_idx + 1
        
        step += 1


if __name__ == "__main__":
    test_kth_largest_element_in_array()
    demonstrate_quickselect()

"""
Kth Largest Element in an Array demonstrates multiple selection algorithms
including heap-based approaches, quickselect, and sorting techniques
for finding order statistics efficiently.
"""
