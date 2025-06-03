"""
Dynamic Programming - Subsequence Patterns
This module implements various DP problems on subsequences including Longest Increasing
Subsequence (LIS), bitonic subsequences, maximum sum subsequences, and counting problems.
"""

from typing import List, Dict, Tuple, Optional
import time
import bisect

# ==================== LONGEST INCREASING SUBSEQUENCE (LIS) ====================

class LongestIncreasingSubsequence:
    """
    Longest Increasing Subsequence Problems
    
    LIS is a fundamental DP problem with applications in many optimization
    problems including scheduling, patience sorting, and box stacking.
    """
    
    def lis_length_dp(self, nums: List[int]) -> int:
        """
        Find length of LIS using DP approach
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            nums: Input array
        
        Returns:
            Length of longest increasing subsequence
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n  # dp[i] = length of LIS ending at index i
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def lis_length_binary_search(self, nums: List[int]) -> int:
        """
        Find length of LIS using binary search optimization
        
        LeetCode 300 - Longest Increasing Subsequence
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        
        Args:
            nums: Input array
        
        Returns:
            Length of longest increasing subsequence
        """
        if not nums:
            return 0
        
        # tails[i] = smallest ending element of all increasing subsequences of length i+1
        tails = []
        
        for num in nums:
            # Find position to insert/replace using binary search
            pos = bisect.bisect_left(tails, num)
            
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num
        
        return len(tails)
    
    def lis_with_subsequence(self, nums: List[int]) -> Tuple[int, List[int]]:
        """
        Find LIS length and one actual LIS subsequence
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            nums: Input array
        
        Returns:
            Tuple of (length, actual_subsequence)
        """
        if not nums:
            return 0, []
        
        n = len(nums)
        dp = [1] * n
        parent = [-1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
        
        # Find the index with maximum LIS length
        max_length = max(dp)
        max_index = dp.index(max_length)
        
        # Reconstruct the LIS
        lis = []
        current = max_index
        
        while current != -1:
            lis.append(nums[current])
            current = parent[current]
        
        lis.reverse()
        return max_length, lis
    
    def lis_count(self, nums: List[int]) -> int:
        """
        Count number of different LIS
        
        LeetCode 673 - Number of Longest Increasing Subsequence
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            nums: Input array
        
        Returns:
            Number of different longest increasing subsequences
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n      # Length of LIS ending at i
        count = [1] * n   # Count of LIS ending at i
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        count[i] = count[j]
                    elif dp[j] + 1 == dp[i]:
                        count[i] += count[j]
        
        max_length = max(dp)
        return sum(count[i] for i in range(n) if dp[i] == max_length)
    
    def lis_non_decreasing(self, nums: List[int]) -> int:
        """
        Find LIS allowing equal elements (non-decreasing)
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        tails = []
        
        for num in nums:
            # Use bisect_right for non-decreasing (allows equal elements)
            pos = bisect.bisect_right(tails, num)
            
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num
        
        return len(tails)
    
    def lis_k_decreasing(self, nums: List[int], k: int) -> int:
        """
        Find LIS in k-decreasing array
        
        A k-decreasing array can be partitioned into at most k decreasing subsequences.
        
        Args:
            nums: Input array (k-decreasing)
            k: Maximum number of decreasing subsequences
        
        Returns:
            Length of LIS
        """
        # For k-decreasing array, we can use k different "tails" arrays
        tails_arrays = [[] for _ in range(k)]
        
        for num in nums:
            # Find the tails array with the largest ending element that's still < num
            best_array = -1
            best_val = float('-inf')
            
            for i in range(k):
                if tails_arrays[i] and tails_arrays[i][-1] >= best_val and tails_arrays[i][-1] < num:
                    best_array = i
                    best_val = tails_arrays[i][-1]
                elif not tails_arrays[i] and best_array == -1:
                    best_array = i
            
            if best_array != -1:
                # Binary search in the chosen array
                pos = bisect.bisect_left(tails_arrays[best_array], num)
                
                if pos == len(tails_arrays[best_array]):
                    tails_arrays[best_array].append(num)
                else:
                    tails_arrays[best_array][pos] = num
        
        return sum(len(tails) for tails in tails_arrays)

# ==================== LONGEST BITONIC SUBSEQUENCE ====================

class LongestBitonicSubsequence:
    """
    Longest Bitonic Subsequence Problems
    
    A bitonic subsequence first increases then decreases.
    Can be solved by combining LIS from left and LDS from right.
    """
    
    def longest_bitonic_subsequence(self, nums: List[int]) -> int:
        """
        Find length of longest bitonic subsequence
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            nums: Input array
        
        Returns:
            Length of longest bitonic subsequence
        """
        if not nums:
            return 0
        
        n = len(nums)
        
        # LIS ending at each position
        lis = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    lis[i] = max(lis[i], lis[j] + 1)
        
        # LDS starting at each position (computed backwards)
        lds = [1] * n
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                if nums[i] > nums[j]:
                    lds[i] = max(lds[i], lds[j] + 1)
        
        # Bitonic subsequence length at each position
        max_bitonic = 0
        for i in range(n):
            max_bitonic = max(max_bitonic, lis[i] + lds[i] - 1)
        
        return max_bitonic
    
    def longest_bitonic_subsequence_optimized(self, nums: List[int]) -> int:
        """
        Optimized version using binary search
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        
        # Compute LIS lengths using binary search
        lis = [0] * n
        tails = []
        
        for i in range(n):
            pos = bisect.bisect_left(tails, nums[i])
            if pos == len(tails):
                tails.append(nums[i])
            else:
                tails[pos] = nums[i]
            lis[i] = pos + 1
        
        # Compute LDS lengths using binary search (reverse array)
        lds = [0] * n
        tails = []
        
        for i in range(n - 1, -1, -1):
            pos = bisect.bisect_left(tails, -nums[i])  # Use negative for decreasing
            if pos == len(tails):
                tails.append(-nums[i])
            else:
                tails[pos] = -nums[i]
            lds[i] = pos + 1
        
        # Find maximum bitonic length
        return max(lis[i] + lds[i] - 1 for i in range(n))
    
    def bitonic_subsequence_with_peak(self, nums: List[int]) -> Tuple[int, int]:
        """
        Find longest bitonic subsequence and its peak element
        
        Args:
            nums: Input array
        
        Returns:
            Tuple of (length, peak_index)
        """
        if not nums:
            return 0, -1
        
        n = len(nums)
        
        lis = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    lis[i] = max(lis[i], lis[j] + 1)
        
        lds = [1] * n
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                if nums[i] > nums[j]:
                    lds[i] = max(lds[i], lds[j] + 1)
        
        max_length = 0
        peak_index = 0
        
        for i in range(n):
            bitonic_length = lis[i] + lds[i] - 1
            if bitonic_length > max_length:
                max_length = bitonic_length
                peak_index = i
        
        return max_length, peak_index

# ==================== MAXIMUM SUM INCREASING SUBSEQUENCE ====================

class MaximumSumIncreasingSubsequence:
    """
    Maximum Sum Increasing Subsequence Problems
    
    Instead of longest length, find the subsequence with maximum sum
    while maintaining increasing order.
    """
    
    def max_sum_increasing_subsequence(self, nums: List[int]) -> int:
        """
        Find maximum sum of increasing subsequence
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        
        Args:
            nums: Input array
        
        Returns:
            Maximum sum of increasing subsequence
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = nums[:]  # dp[i] = max sum of increasing subsequence ending at i
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + nums[i])
        
        return max(dp)
    
    def max_sum_increasing_subsequence_with_sequence(self, nums: List[int]) -> Tuple[int, List[int]]:
        """
        Find maximum sum and the actual subsequence
        
        Args:
            nums: Input array
        
        Returns:
            Tuple of (max_sum, subsequence)
        """
        if not nums:
            return 0, []
        
        n = len(nums)
        dp = nums[:]
        parent = [-1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i] and dp[j] + nums[i] > dp[i]:
                    dp[i] = dp[j] + nums[i]
                    parent[i] = j
        
        # Find index with maximum sum
        max_sum = max(dp)
        max_index = dp.index(max_sum)
        
        # Reconstruct subsequence
        subsequence = []
        current = max_index
        
        while current != -1:
            subsequence.append(nums[current])
            current = parent[current]
        
        subsequence.reverse()
        return max_sum, subsequence
    
    def max_sum_bitonic_subsequence(self, nums: List[int]) -> int:
        """
        Find maximum sum of bitonic subsequence
        
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        if not nums:
            return 0
        
        n = len(nums)
        
        # Max sum increasing subsequence ending at each position
        inc = nums[:]
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    inc[i] = max(inc[i], inc[j] + nums[i])
        
        # Max sum decreasing subsequence starting at each position
        dec = nums[:]
        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n):
                if nums[i] > nums[j]:
                    dec[i] = max(dec[i], dec[j] + nums[i])
        
        # Maximum bitonic sum
        return max(inc[i] + dec[i] - nums[i] for i in range(n))

# ==================== MINIMUM DELETIONS TO SORT ARRAY ====================

class MinimumDeletions:
    """
    Minimum Deletions to Sort Array
    
    Find minimum number of elements to delete to make array sorted.
    This is equivalent to finding LIS and deleting the rest.
    """
    
    def min_deletions_to_sort(self, nums: List[int]) -> int:
        """
        Find minimum deletions to make array sorted
        
        Key insight: min_deletions = n - LIS_length
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        
        Args:
            nums: Input array
        
        Returns:
            Minimum number of deletions needed
        """
        if not nums:
            return 0
        
        lis = LongestIncreasingSubsequence()
        lis_length = lis.lis_length_binary_search(nums)
        
        return len(nums) - lis_length
    
    def min_deletions_with_elements(self, nums: List[int]) -> Tuple[int, List[int]]:
        """
        Find minimum deletions and elements to delete
        
        Args:
            nums: Input array
        
        Returns:
            Tuple of (num_deletions, elements_to_delete)
        """
        if not nums:
            return 0, []
        
        lis = LongestIncreasingSubsequence()
        lis_length, lis_sequence = lis.lis_with_subsequence(nums)
        
        # Elements to keep (LIS)
        lis_set = set(lis_sequence)
        elements_to_delete = []
        
        # Track which elements we've used from LIS
        lis_index = 0
        
        for num in nums:
            if lis_index < len(lis_sequence) and num == lis_sequence[lis_index]:
                lis_index += 1
            else:
                elements_to_delete.append(num)
        
        return len(elements_to_delete), elements_to_delete
    
    def min_deletions_to_make_sorted_with_duplicates(self, nums: List[int]) -> int:
        """
        Handle case where array can have duplicates
        
        For non-decreasing order, we allow equal elements
        """
        if not nums:
            return 0
        
        lis = LongestIncreasingSubsequence()
        lis_length = lis.lis_non_decreasing(nums)
        
        return len(nums) - lis_length

# ==================== ADVANCED SUBSEQUENCE PROBLEMS ====================

class AdvancedSubsequenceProblems:
    """
    Advanced subsequence problems with additional constraints
    """
    
    def lis_with_difference_constraint(self, nums: List[int], max_diff: int) -> int:
        """
        LIS where consecutive elements differ by at most max_diff
        
        Args:
            nums: Input array
            max_diff: Maximum allowed difference between consecutive elements
        
        Returns:
            Length of constrained LIS
        """
        if not nums:
            return 0
        
        n = len(nums)
        dp = [1] * n
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i] and nums[i] - nums[j] <= max_diff:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        return max(dp)
    
    def lis_with_sum_constraint(self, nums: List[int], max_sum: int) -> int:
        """
        LIS where sum of elements doesn't exceed max_sum
        
        Args:
            nums: Input array
            max_sum: Maximum allowed sum
        
        Returns:
            Length of LIS with sum constraint
        """
        if not nums:
            return 0
        
        n = len(nums)
        # dp[i] = (length, sum) of best LIS ending at position i
        dp = [(1, nums[i]) for i in range(n)]
        
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    new_length = dp[j][0] + 1
                    new_sum = dp[j][1] + nums[i]
                    
                    if new_sum <= max_sum and new_length > dp[i][0]:
                        dp[i] = (new_length, new_sum)
        
        return max(length for length, sum_val in dp)
    
    def lis_in_matrix(self, matrix: List[List[int]]) -> int:
        """
        Find LIS in matrix where you can move right or down
        
        Args:
            matrix: 2D matrix
        
        Returns:
            Length of LIS in matrix
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        dp = [[1] * n for _ in range(m)]
        
        max_length = 1
        
        for i in range(m):
            for j in range(n):
                # From left
                if j > 0 and matrix[i][j - 1] < matrix[i][j]:
                    dp[i][j] = max(dp[i][j], dp[i][j - 1] + 1)
                
                # From top
                if i > 0 and matrix[i - 1][j] < matrix[i][j]:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j] + 1)
                
                max_length = max(max_length, dp[i][j])
        
        return max_length
    
    def wiggle_subsequence(self, nums: List[int]) -> int:
        """
        Find longest wiggle subsequence
        
        LeetCode 376 - Wiggle Subsequence
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if len(nums) < 2:
            return len(nums)
        
        up = down = 1
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                up = down + 1
            elif nums[i] < nums[i - 1]:
                down = up + 1
        
        return max(up, down)
    
    def longest_arithmetic_subsequence(self, nums: List[int]) -> int:
        """
        Find longest arithmetic subsequence
        
        LeetCode 1027 - Longest Arithmetic Subsequence
        
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        if len(nums) <= 2:
            return len(nums)
        
        n = len(nums)
        dp = [{} for _ in range(n)]
        max_length = 2
        
        for i in range(1, n):
            for j in range(i):
                diff = nums[i] - nums[j]
                
                if diff in dp[j]:
                    dp[i][diff] = dp[j][diff] + 1
                else:
                    dp[i][diff] = 2
                
                max_length = max(max_length, dp[i][diff])
        
        return max_length

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different LIS implementations"""
    print("=== Subsequence DP Performance Analysis ===\n")
    
    import random
    
    # Generate test data
    test_sizes = [100, 500, 1000]
    
    for size in test_sizes:
        nums = [random.randint(1, 1000) for _ in range(size)]
        
        print(f"Array size: {size}")
        
        lis = LongestIncreasingSubsequence()
        
        # DP approach
        start_time = time.time()
        dp_result = lis.lis_length_dp(nums)
        dp_time = time.time() - start_time
        
        # Binary search approach
        start_time = time.time()
        bs_result = lis.lis_length_binary_search(nums)
        bs_time = time.time() - start_time
        
        print(f"  DP O(n²): {dp_result} ({dp_time:.6f}s)")
        print(f"  Binary Search O(n log n): {bs_result} ({bs_time:.6f}s)")
        print(f"  Results match: {dp_result == bs_result}")
        print(f"  Speedup: {dp_time / bs_time:.2f}x")
        print()

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Subsequence DP Demo ===\n")
    
    # Longest Increasing Subsequence
    print("1. Longest Increasing Subsequence:")
    lis = LongestIncreasingSubsequence()
    
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    
    length_dp = lis.lis_length_dp(nums)
    length_bs = lis.lis_length_binary_search(nums)
    length_with_seq, sequence = lis.lis_with_subsequence(nums)
    count = lis.lis_count(nums)
    
    print(f"  Array: {nums}")
    print(f"  LIS length (DP): {length_dp}")
    print(f"  LIS length (Binary Search): {length_bs}")
    print(f"  One LIS: {sequence}")
    print(f"  Number of different LIS: {count}")
    
    # Non-decreasing
    nums_with_dups = [1, 3, 6, 7, 9, 4, 10, 5, 6]
    non_dec_length = lis.lis_non_decreasing(nums_with_dups)
    print(f"  Non-decreasing LIS in {nums_with_dups}: {non_dec_length}")
    print()
    
    # Longest Bitonic Subsequence
    print("2. Longest Bitonic Subsequence:")
    bitonic = LongestBitonicSubsequence()
    
    nums = [1, 11, 2, 10, 4, 5, 2, 1]
    bitonic_length = bitonic.longest_bitonic_subsequence(nums)
    bitonic_opt = bitonic.longest_bitonic_subsequence_optimized(nums)
    bitonic_with_peak, peak_idx = bitonic.bitonic_subsequence_with_peak(nums)
    
    print(f"  Array: {nums}")
    print(f"  Bitonic length: {bitonic_length}")
    print(f"  Optimized bitonic length: {bitonic_opt}")
    print(f"  Peak at index {peak_idx} (value {nums[peak_idx]})")
    print()
    
    # Maximum Sum Increasing Subsequence
    print("3. Maximum Sum Increasing Subsequence:")
    max_sum_lis = MaximumSumIncreasingSubsequence()
    
    nums = [1, 101, 2, 3, 100, 4, 5]
    max_sum = max_sum_lis.max_sum_increasing_subsequence(nums)
    max_sum_with_seq, max_sum_sequence = max_sum_lis.max_sum_increasing_subsequence_with_sequence(nums)
    
    print(f"  Array: {nums}")
    print(f"  Maximum sum: {max_sum}")
    print(f"  Subsequence: {max_sum_sequence}")
    
    # Bitonic sum
    bitonic_sum = max_sum_lis.max_sum_bitonic_subsequence(nums)
    print(f"  Maximum bitonic sum: {bitonic_sum}")
    print()
    
    # Minimum Deletions to Sort
    print("4. Minimum Deletions to Sort Array:")
    min_del = MinimumDeletions()
    
    nums = [6, 5, 4, 3, 2, 1, 9, 8, 7]
    deletions = min_del.min_deletions_to_sort(nums)
    deletions_with_elements, elements_to_delete = min_del.min_deletions_with_elements(nums)
    
    print(f"  Array: {nums}")
    print(f"  Minimum deletions needed: {deletions}")
    print(f"  Elements to delete: {elements_to_delete}")
    
    # With duplicates
    nums_dups = [6, 5, 4, 4, 3, 2, 1, 9, 8, 7]
    deletions_dups = min_del.min_deletions_to_make_sorted_with_duplicates(nums_dups)
    print(f"  With duplicates {nums_dups}: {deletions_dups} deletions")
    print()
    
    # Advanced Subsequence Problems
    print("5. Advanced Subsequence Problems:")
    advanced = AdvancedSubsequenceProblems()
    
    # LIS with difference constraint
    nums = [1, 3, 6, 7, 9, 4, 10, 5, 6]
    max_diff = 3
    constrained_lis = advanced.lis_with_difference_constraint(nums, max_diff)
    print(f"  LIS with max diff {max_diff} in {nums}: {constrained_lis}")
    
    # LIS with sum constraint
    max_sum = 15
    sum_constrained_lis = advanced.lis_with_sum_constraint(nums, max_sum)
    print(f"  LIS with max sum {max_sum}: {sum_constrained_lis}")
    
    # Wiggle subsequence
    nums_wiggle = [1, 7, 4, 9, 2, 5]
    wiggle_length = advanced.wiggle_subsequence(nums_wiggle)
    print(f"  Wiggle subsequence in {nums_wiggle}: {wiggle_length}")
    
    # Longest arithmetic subsequence
    nums_arith = [3, 6, 9, 12]
    arith_length = advanced.longest_arithmetic_subsequence(nums_arith)
    print(f"  Longest arithmetic subsequence in {nums_arith}: {arith_length}")
    
    # LIS in matrix
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    matrix_lis = advanced.lis_in_matrix(matrix)
    print(f"  LIS in matrix: {matrix_lis}")
    print()
    
    # Performance comparison
    performance_comparison()
    
    # Pattern Recognition Guide
    print("=== Subsequence DP Pattern Recognition ===")
    print("Common Subsequence DP Patterns:")
    print("  1. LIS: dp[i] = max(dp[j] + 1) for j < i where arr[j] < arr[i]")
    print("  2. Bitonic: Combine LIS from left and LDS from right")
    print("  3. Max Sum: Replace length with sum in transitions")
    print("  4. Counting: Add up possibilities instead of taking max")
    print("  5. Constraints: Add conditions to state transitions")
    
    print("\nOptimization Techniques:")
    print("  1. Binary Search: O(n²) → O(n log n) for basic LIS")
    print("  2. Coordinate Compression: For large value ranges")
    print("  3. Fenwick Tree/Segment Tree: For 2D LIS problems")
    print("  4. Patience Sorting: Alternative view of LIS algorithm")
    
    print("\nCommon Variations:")
    print("  1. Non-decreasing vs strictly increasing")
    print("  2. Longest vs count vs sum optimization")
    print("  3. Additional constraints (difference, sum, etc.)")
    print("  4. 2D/Matrix versions of subsequence problems")
    print("  5. Bitonic, wiggle, arithmetic progressions")
    
    print("\nReal-world Applications:")
    print("  1. Box stacking and scheduling problems")
    print("  2. Version control and diff algorithms")
    print("  3. Bioinformatics sequence alignment")
    print("  4. Financial trading strategies")
    print("  5. Game theory and optimal play")
    print("  6. Resource allocation optimization")
    
    print("\n=== Demo Complete ===") 