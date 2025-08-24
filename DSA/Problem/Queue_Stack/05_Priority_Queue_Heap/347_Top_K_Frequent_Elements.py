"""
347. Top K Frequent Elements - Multiple Approaches
Difficulty: Medium

Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.
"""

from typing import List, Dict
import heapq
from collections import Counter, defaultdict
import random

class TopKFrequentElements:
    """Multiple approaches to find top K frequent elements"""
    
    def topKFrequent_min_heap(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 1: Min Heap
        
        Use min heap to keep track of top K frequent elements.
        
        Time: O(n log k), Space: O(n + k)
        """
        # Count frequencies
        freq_map = Counter(nums)
        
        # Use min heap to keep top k elements
        heap = []
        
        for num, freq in freq_map.items():
            heapq.heappush(heap, (freq, num))
            if len(heap) > k:
                heapq.heappop(heap)
        
        return [num for freq, num in heap]
    
    def topKFrequent_max_heap(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 2: Max Heap
        
        Use max heap to get top K elements directly.
        
        Time: O(n log n), Space: O(n)
        """
        freq_map = Counter(nums)
        
        # Create max heap (negate frequencies for min heap to work as max heap)
        heap = [(-freq, num) for num, freq in freq_map.items()]
        heapq.heapify(heap)
        
        result = []
        for _ in range(k):
            freq, num = heapq.heappop(heap)
            result.append(num)
        
        return result
    
    def topKFrequent_bucket_sort(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 3: Bucket Sort
        
        Use bucket sort based on frequencies.
        
        Time: O(n), Space: O(n)
        """
        freq_map = Counter(nums)
        n = len(nums)
        
        # Create buckets for each possible frequency
        buckets = [[] for _ in range(n + 1)]
        
        # Place numbers in buckets based on their frequency
        for num, freq in freq_map.items():
            buckets[freq].append(num)
        
        # Collect top k elements from highest frequency buckets
        result = []
        for i in range(n, 0, -1):
            if buckets[i]:
                result.extend(buckets[i])
                if len(result) >= k:
                    break
        
        return result[:k]
    
    def topKFrequent_quickselect(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 4: Quickselect Algorithm
        
        Use quickselect to find kth most frequent element.
        
        Time: O(n) average, O(n²) worst case, Space: O(n)
        """
        freq_map = Counter(nums)
        unique_nums = list(freq_map.keys())
        
        def quickselect(left: int, right: int, k_smallest: int) -> None:
            """
            Sort a list within left..right till kth less frequent element
            takes its place.
            """
            # Base case: the list contains only one element
            if left == right:
                return
            
            # Select a random pivot_index
            pivot_index = left + random.randint(0, right - left)
            
            # Find the pivot position in a sorted list
            pivot_index = partition(left, right, pivot_index)
            
            # If the pivot is in its final sorted position
            if k_smallest == pivot_index:
                return
            # Go left
            elif k_smallest < pivot_index:
                quickselect(left, pivot_index - 1, k_smallest)
            # Go right
            else:
                quickselect(pivot_index + 1, right, k_smallest)
        
        def partition(left: int, right: int, pivot_index: int) -> int:
            pivot_frequency = freq_map[unique_nums[pivot_index]]
            
            # Move pivot to end
            unique_nums[pivot_index], unique_nums[right] = unique_nums[right], unique_nums[pivot_index]
            
            # Move all less frequent elements to the left
            store_index = left
            for i in range(left, right):
                if freq_map[unique_nums[i]] < pivot_frequency:
                    unique_nums[store_index], unique_nums[i] = unique_nums[i], unique_nums[store_index]
                    store_index += 1
            
            # Move pivot to its final place
            unique_nums[right], unique_nums[store_index] = unique_nums[store_index], unique_nums[right]
            
            return store_index
        
        n = len(unique_nums)
        # kth top frequent element is (n - k)th less frequent.
        # Do a partial sort: from less frequent to the most frequent, till
        # (n - k)th less frequent element takes its place (n - k) in a sorted array.
        # All elements on the left are less frequent.
        # All the elements on the right are more frequent.
        quickselect(0, n - 1, n - k)
        
        # Return top k frequent elements
        return unique_nums[n - k:]
    
    def topKFrequent_sorting(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 5: Sorting
        
        Sort by frequency and return top k.
        
        Time: O(n log n), Space: O(n)
        """
        freq_map = Counter(nums)
        
        # Sort by frequency in descending order
        sorted_items = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
        
        return [num for num, freq in sorted_items[:k]]
    
    def topKFrequent_counter_most_common(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 6: Counter.most_common()
        
        Use built-in Counter.most_common() method.
        
        Time: O(n log k), Space: O(n)
        """
        freq_map = Counter(nums)
        return [num for num, freq in freq_map.most_common(k)]
    
    def topKFrequent_trie_approach(self, nums: List[int], k: int) -> List[int]:
        """
        Approach 7: Trie-based Frequency Counting
        
        Use trie for frequency counting (educational approach).
        
        Time: O(n log n), Space: O(n)
        """
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.count = 0
                self.numbers = []
        
        # Build frequency map first
        freq_map = Counter(nums)
        
        # Build trie based on frequencies
        root = TrieNode()
        
        for num, freq in freq_map.items():
            current = root
            # Convert frequency to string and build path
            freq_str = str(freq)
            for char in freq_str:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
            current.numbers.append(num)
        
        # This approach is more complex than needed for this problem
        # Fallback to sorting approach
        return self.topKFrequent_sorting(nums, k)

def test_top_k_frequent():
    """Test top K frequent elements algorithms"""
    solver = TopKFrequentElements()
    
    test_cases = [
        ([1,1,1,2,2,3], 2, "Example 1"),
        ([1], 1, "Single element"),
        ([1,2], 2, "Two elements"),
        ([4,1,-1,2,-1,2,3], 2, "With negative numbers"),
        ([1,1,1,2,2,2,3,3,3], 3, "All same frequency"),
    ]
    
    algorithms = [
        ("Min Heap", solver.topKFrequent_min_heap),
        ("Max Heap", solver.topKFrequent_max_heap),
        ("Bucket Sort", solver.topKFrequent_bucket_sort),
        ("Quickselect", solver.topKFrequent_quickselect),
        ("Sorting", solver.topKFrequent_sorting),
        ("Counter most_common", solver.topKFrequent_counter_most_common),
    ]
    
    print("=== Testing Top K Frequent Elements ===")
    
    for nums, k, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"nums: {nums}, k: {k}")
        
        # Get expected result using Counter
        expected = [num for num, freq in Counter(nums).most_common(k)]
        print(f"Expected (one valid answer): {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums, k)
                # Check if result has correct length and contains valid elements
                freq_map = Counter(nums)
                result_freqs = [freq_map[num] for num in result]
                expected_freqs = [freq_map[num] for num in expected]
                
                # Sort both by frequency to compare
                result_freqs.sort(reverse=True)
                expected_freqs.sort(reverse=True)
                
                status = "✓" if len(result) == k and result_freqs == expected_freqs else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")

def benchmark_top_k_frequent():
    """Benchmark different approaches"""
    import time
    
    # Generate test data
    sizes = [1000, 5000, 10000]
    k_values = [10, 50, 100]
    
    algorithms = [
        ("Min Heap", TopKFrequentElements().topKFrequent_min_heap),
        ("Bucket Sort", TopKFrequentElements().topKFrequent_bucket_sort),
        ("Quickselect", TopKFrequentElements().topKFrequent_quickselect),
        ("Sorting", TopKFrequentElements().topKFrequent_sorting),
    ]
    
    print("\n=== Top K Frequent Elements Performance Benchmark ===")
    
    for size in sizes:
        for k in k_values:
            if k > size // 10:  # Skip if k is too large
                continue
                
            print(f"\n--- Array Size: {size}, K: {k} ---")
            # Generate array with some repeated elements
            nums = [random.randint(1, size // 2) for _ in range(size)]
            
            for alg_name, alg_func in algorithms:
                start_time = time.time()
                try:
                    result = alg_func(nums, k)
                    end_time = time.time()
                    print(f"{alg_name:15} | Time: {end_time - start_time:.4f}s")
                except Exception as e:
                    print(f"{alg_name:15} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_top_k_frequent()
    benchmark_top_k_frequent()

"""
Top K Frequent Elements demonstrates various approaches including
heap-based solutions, bucket sort, quickselect algorithm, and
their performance characteristics for different input sizes.
"""
