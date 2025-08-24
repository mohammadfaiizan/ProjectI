"""
373. Find K Pairs with Smallest Sums - Multiple Approaches
Difficulty: Medium

You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.

Define a pair (u, v) which consists of one element from the first array and one element from the second array.

Return the k pairs (u1, v1), (u2, v2), ..., (uk, vk) with the smallest sums.
"""

from typing import List, Tuple
import heapq

class FindKPairsWithSmallestSums:
    """Multiple approaches to find k pairs with smallest sums"""
    
    def kSmallestPairs_heap_approach(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        """
        Approach 1: Min Heap (Optimal)
        
        Use min heap to track pairs with smallest sums.
        
        Time: O(k log k), Space: O(k)
        """
        if not nums1 or not nums2:
            return []
        
        heap = []
        result = []
        
        # Initialize heap with first row pairs
        for i in range(min(k, len(nums1))):
            heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))
        
        while heap and len(result) < k:
            sum_val, i, j = heapq.heappop(heap)
            result.append([nums1[i], nums2[j]])
            
            # Add next pair from same row if exists
            if j + 1 < len(nums2):
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
        
        return result
    
    def kSmallestPairs_brute_force(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        """
        Approach 2: Brute Force with Sorting
        
        Generate all pairs and sort by sum.
        
        Time: O(mn log(mn)), Space: O(mn)
        """
        if not nums1 or not nums2:
            return []
        
        pairs = []
        
        for num1 in nums1:
            for num2 in nums2:
                pairs.append((num1 + num2, [num1, num2]))
        
        pairs.sort()
        
        return [pair[1] for pair in pairs[:k]]
    
    def kSmallestPairs_max_heap(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        """
        Approach 3: Max Heap of Size K
        
        Maintain max heap of k smallest pairs.
        
        Time: O(mn log k), Space: O(k)
        """
        if not nums1 or not nums2:
            return []
        
        max_heap = []  # Store (-sum, [num1, num2])
        
        for num1 in nums1:
            for num2 in nums2:
                current_sum = num1 + num2
                
                if len(max_heap) < k:
                    heapq.heappush(max_heap, (-current_sum, [num1, num2]))
                elif current_sum < -max_heap[0][0]:
                    heapq.heappop(max_heap)
                    heapq.heappush(max_heap, (-current_sum, [num1, num2]))
        
        result = [pair for _, pair in max_heap]
        result.sort(key=lambda x: x[0] + x[1])  # Sort by sum
        
        return result
    
    def kSmallestPairs_optimized_generation(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        """
        Approach 4: Optimized Pair Generation
        
        Generate pairs more efficiently using sorted property.
        
        Time: O(k log k), Space: O(k)
        """
        if not nums1 or not nums2:
            return []
        
        heap = [(nums1[0] + nums2[0], 0, 0)]
        visited = {(0, 0)}
        result = []
        
        while heap and len(result) < k:
            sum_val, i, j = heapq.heappop(heap)
            result.append([nums1[i], nums2[j]])
            
            # Add adjacent pairs
            if i + 1 < len(nums1) and (i + 1, j) not in visited:
                heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
                visited.add((i + 1, j))
            
            if j + 1 < len(nums2) and (i, j + 1) not in visited:
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
                visited.add((i, j + 1))
        
        return result
    
    def kSmallestPairs_two_pointers(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        """
        Approach 5: Two Pointers with Priority Queue
        
        Use two pointers approach with priority queue.
        
        Time: O(k log k), Space: O(k)
        """
        if not nums1 or not nums2:
            return []
        
        heap = []
        result = []
        
        # Start with pairs from first element of nums1
        for j in range(min(k, len(nums2))):
            heapq.heappush(heap, (nums1[0] + nums2[j], 0, j))
        
        while heap and len(result) < k:
            sum_val, i, j = heapq.heappop(heap)
            result.append([nums1[i], nums2[j]])
            
            # Add next pair from same column
            if i + 1 < len(nums1):
                heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
        
        return result


def test_find_k_pairs_with_smallest_sums():
    """Test find k pairs with smallest sums algorithms"""
    solver = FindKPairsWithSmallestSums()
    
    test_cases = [
        ([1,7,11], [2,4,6], 3, [[1,2],[1,4],[1,6]], "Example 1"),
        ([1,1,2], [1,2,3], 2, [[1,1],[1,1]], "Example 2"),
        ([1,2], [3], 3, [[1,3],[2,3]], "Example 3"),
        ([1], [1], 1, [[1,1]], "Single elements"),
        ([1,2,3], [1,2,3], 4, [[1,1],[1,2],[2,1],[1,3]], "Square matrix"),
        ([1,1,1], [1,1,1], 5, [[1,1],[1,1],[1,1],[1,1],[1,1]], "Duplicates"),
        ([1,7,11], [2,4,6], 10, [[1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]], "More than available"),
    ]
    
    algorithms = [
        ("Heap Approach", solver.kSmallestPairs_heap_approach),
        ("Brute Force", solver.kSmallestPairs_brute_force),
        ("Max Heap", solver.kSmallestPairs_max_heap),
        ("Optimized Generation", solver.kSmallestPairs_optimized_generation),
        ("Two Pointers", solver.kSmallestPairs_two_pointers),
    ]
    
    print("=== Testing Find K Pairs with Smallest Sums ===")
    
    for nums1, nums2, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"nums1: {nums1}")
        print(f"nums2: {nums2}")
        print(f"k: {k}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(nums1, nums2, k)
                # Sort both expected and result for comparison
                expected_sorted = sorted(expected, key=lambda x: (x[0] + x[1], x[0], x[1]))
                result_sorted = sorted(result, key=lambda x: (x[0] + x[1], x[0], x[1]))
                status = "✓" if result_sorted == expected_sorted else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_heap_approach():
    """Demonstrate heap approach step by step"""
    print("\n=== Heap Approach Step-by-Step Demo ===")
    
    nums1 = [1, 7, 11]
    nums2 = [2, 4, 6]
    k = 3
    
    print(f"nums1: {nums1}")
    print(f"nums2: {nums2}")
    print(f"k: {k}")
    
    print("\nAll possible pairs with sums:")
    for i, num1 in enumerate(nums1):
        for j, num2 in enumerate(nums2):
            print(f"  ({num1}, {num2}) -> sum = {num1 + num2}")
    
    print(f"\nUsing heap to find {k} smallest pairs:")
    
    heap = []
    result = []
    
    # Initialize with first row
    print(f"\nInitialize heap with first row pairs:")
    for i in range(min(k, len(nums1))):
        pair_sum = nums1[i] + nums2[0]
        heapq.heappush(heap, (pair_sum, i, 0))
        print(f"  Added ({nums1[i]}, {nums2[0]}) with sum {pair_sum}")
    
    print(f"Initial heap: {heap}")
    
    step = 1
    while heap and len(result) < k:
        print(f"\nStep {step}:")
        sum_val, i, j = heapq.heappop(heap)
        pair = [nums1[i], nums2[j]]
        result.append(pair)
        
        print(f"  Popped: sum={sum_val}, pair=({nums1[i]}, {nums2[j]})")
        
        # Add next pair from same row
        if j + 1 < len(nums2):
            next_sum = nums1[i] + nums2[j + 1]
            heapq.heappush(heap, (next_sum, i, j + 1))
            print(f"  Added next pair: ({nums1[i]}, {nums2[j + 1]}) with sum {next_sum}")
        
        print(f"  Current result: {result}")
        print(f"  Heap: {heap}")
        
        step += 1
    
    print(f"\nFinal result: {result}")


if __name__ == "__main__":
    test_find_k_pairs_with_smallest_sums()
    demonstrate_heap_approach()

"""
Find K Pairs with Smallest Sums demonstrates heap applications for
finding optimal pairs from sorted arrays, including multiple optimization
strategies and efficient pair generation techniques.
"""
